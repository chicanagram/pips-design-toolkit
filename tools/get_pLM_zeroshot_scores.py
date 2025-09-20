import os
import numpy as np
import pandas as pd
pd.set_option('display.max_columns',None)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import torch
from utils.utils import get_log_likelihood_ratios, flatten_2D_arr, mkDir
from utils.variables import aaList, aaList_with_X, data_folder, subfolders
from utils.pLM_pipeline_utils import get_plm_pipeline_inputs, initialize_plm_feature_variables, include_all_substitutions_for_mutated_positions
from feature_extraction.esm2 import ESM2


class PLMZero:
    def __init__(
            self,
            model_type,
            msa_fpath=None,
            data_folder=data_folder,
            data_subfolder='',
            get_mutations=True,
            get_embeddings = {'res_all': True, 'res_mut': True,'res_avg': True},
            deduplicate_sequences=True,
            skip_plm_processing=False,
            skip_logits_features=False
            ):
        self.PLM_dict = {
            'esm2': {'class':ESM2, 'model_name':'esm2_t33_650M_UR50D', 'repr_layers':[33], 'get_mutation_logits_probs':'mask'},
        }
        self.model_type = model_type
        self.msa_fpath = msa_fpath
        self.data_folder = data_folder
        self.data_subfolder = data_subfolder
        self.sequences_dir = data_folder + subfolders['sequences']
        self.embeddings_dir = data_folder + subfolders['protein_embeddings'] + data_subfolder + '/'
        self.conservation_analysis_dir = data_folder + subfolders['conservation_analysis'] + data_subfolder + '/'
        print('Embeddings dir:', self.embeddings_dir)
        print('Conservation analysis dir:', self.conservation_analysis_dir)
        mkDir(data_subfolder, data_folder + subfolders['conservation_analysis'], remove_existing_dir=False)
        mkDir(data_subfolder, data_folder + subfolders['protein_embeddings'], remove_existing_dir=False)
        self.get_mutations = get_mutations
        self.get_embeddings = get_embeddings
        self.deduplicate_sequences = deduplicate_sequences
        self.skip_plm_processing = skip_plm_processing
        self.skip_logits_features = skip_logits_features

    def get_inputs(
            self, 
            input_file, 
            batch_size=1,
            csv_name_col = 'protein_name',
            sep = '+',
            ):
        # define separator for combi mutants
        self.sep = sep
        # inputs
        if input_file.find('.csv')>-1:
            input_fpath = self.data_folder + input_file
        elif input_file.find('.fasta')>-1:
            input_fpath = self.sequences_dir + self.data_subfolder + '/' + input_file

        # model inputs
        [self.model_name, self.repr_layers, self.get_mutation_logits_probs] = [self.PLM_dict[self.model_type][key] for key in ['model_name','repr_layers','get_mutation_logits_probs']]
        if self.skip_logits_features:
            self.get_mutation_logits_probs = None
        print(self.model_type, self.model_name, self.repr_layers, self.get_mutation_logits_probs)
        self.layer_str = '-'.join([str(l) for l in self.repr_layers])
        print(self.model_name, self.repr_layers)

        # get input sequences
        subset_idx = None # specify index if we only want to process a subset of the inputs
        inputs_all = get_plm_pipeline_inputs(input_fpath, self.get_mutations, subset_idx=subset_idx, csv_name_col=csv_name_col, sep=self.sep)
        print(f"inputs_all['base'] {len(inputs_all['base']['sequence'])} sequences ({len(list(set(inputs_all['base']['sequence'])))} unique)")
        print(f"inputs_all['mut']  {len(inputs_all['mut']['sequence'])} sequences ({len(list(set(inputs_all['mut']['sequence'])))} unique)")
        from collections import Counter
        cnt = Counter(inputs_all['mut']['sequence'])
        print('Mutated sequences with repeats:', [k for k, v in cnt.items() if v > 1])

        # get number of tokens per batch
        self.tokens_per_batch = min(len(inputs_all['base']['name']), batch_size)

        # initialize variables and filepaths for storing results
        fname = f'{os.path.basename(input_file).split(".")[0]}_{self.model_type}-{self.layer_str}'
        self.emb_folder = self.embeddings_dir + '/' + self.model_type + '-' + self.layer_str + '/'
        self.llrsum_entropy_mutants_fpath = self.conservation_analysis_dir + fname + '_LLRsum_entropy.csv'
        self.mut_embeddings_fpath = self.embeddings_dir + fname + '_mut_embeddings.csv'
        self.seq_embeddings_fpath = self.embeddings_dir + fname + '_seq_embeddings.csv'
    
        return inputs_all
        
        
    def run_pLM_processing(self, inputs_all):
    
        # initialize PLM model
        plmClass = self.PLM_dict[self.model_type]['class']
        plm = plmClass(self.data_subfolder, self.data_folder, model_name=self.model_name, model_type=self.model_type)
    
        # deduplicate inputs and process with PLM to get features
        plm_inputs = {base_or_mut:{'name':[],'sequence':[], 'mutations':[]} for base_or_mut in ['base','mut']}

        for base_or_mut in ['base','mut']:
            print(f'Processing [{base_or_mut}] sequences...')
            inputs_base_or_mut = inputs_all[base_or_mut]
            # print(f'inputs_all ({base_or_mut})')
            # print(inputs_base_or_mut)
    
            # create list of base and mut sequences for PLM to process
            # deduplicate sequence inputs
            if self.deduplicate_sequences:
                for k, (seq_name, seq, mut) in enumerate(zip(inputs_base_or_mut['name'], inputs_base_or_mut['sequence'], inputs_base_or_mut['mutations'])):
                    # print(k, seq_name, mut)
    
                    if seq not in plm_inputs[base_or_mut]['sequence'] and seq is not None:
                        plm_inputs[base_or_mut]['sequence'].append(seq)
                        plm_inputs[base_or_mut]['name'].append(seq_name)
    
                        # aggregate list of mutations for that particular base sequence
                        if base_or_mut=='base':
                            mutations_seq_ = [inputs_base_or_mut['mutations'][i] for i, name in enumerate(inputs_base_or_mut['name']) if name==seq_name]
                            mutations_seq = []
                            for mut in mutations_seq_:
                                if mut not in mutations_seq:
                                    mutations_seq.append(mut)
                            print('Aggregated mutations:', seq_name, '# mutations_seq:', len(mutations_seq))
                            plm_inputs['base']['mutations'].append(mutations_seq)
                            # # MANUALLY MODIFY INPUTS_ALL MUTATIONS
                            # plm_inputs['base']['mutations'].append([10,12])
                        elif base_or_mut=='mut':
                            plm_inputs['mut']['mutations'].append(mut)
    
            # no need to deduplicate sequences
            else:
                plm_inputs[base_or_mut] = inputs_all[base_or_mut].copy()
                if self.get_mutations and base_or_mut=='base':
                    mutations_list = plm_inputs[base_or_mut]['mutations'].copy()
                    plm_inputs[base_or_mut]['mutations'] = [[mut] for mut in mutations_list]
    
            # expand mutations list to include all substitutions from positions mutated (so that probabilities & entropies can be calculated from the scores directly, without using logits)
            if base_or_mut=='base' and self.model_type in ['PoET', 'PoET2'] and self.get_mutation_logits_probs is not None:
                plm_inputs[base_or_mut] = include_all_substitutions_for_mutated_positions(plm_inputs[base_or_mut])
    
            print(base_or_mut, f'{len(plm_inputs[base_or_mut]["sequence"])} sequence(s), {sum(list(set([len(mut) for mut in plm_inputs[base_or_mut]["mutations"]])))} variant sequences total.')
            # print(f'plm_inputs ({base_or_mut})')
            # print(plm_inputs[base_or_mut])
    
    
            ####################
            # run PLM pipeline #
            ####################
            if len(plm_inputs[base_or_mut]['sequence'])>0 and \
                    ((self.get_embeddings['res_all'] or self.get_embeddings['res_mut'] or self.get_embeddings['res_avg']) or
                     self.get_mutation_logits_probs is not None):
    
                # skip processing mut samples for PoET
                if base_or_mut=='mut' and self.model_type in ['PoET','PoET2']:
                    print('Skip processing of mut group sequences with PoET/PoET2 PLMs.')
                    continue
                # continue processing under all other conditions
                else:
                    if not self.skip_plm_processing:
                        plm.run_prediction_pipeline(
                            plm_inputs[base_or_mut]['name'],
                            plm_inputs[base_or_mut]['sequence'],
                            plm_inputs[base_or_mut]['mutations'],
                            self.tokens_per_batch,
                            self.get_mutation_logits_probs if base_or_mut=='base' else None,
                            self.get_embeddings['res_all'] or self.get_embeddings['res_mut'],
                            self.get_embeddings['res_avg'],
                            self.repr_layers,
                            msa_fpath=self.msa_fpath
                        )

    def parse_plm_results(self, inputs_all, get_llr_entropy_plot=False):
        
        # parse results for each sequence in inputs_all
        seq_embs, mut_embs, llrsum_entropy_mutants, llr_entropy_dict_byseqbase = initialize_plm_feature_variables(self.get_embeddings, self.get_mutations, self.get_mutation_logits_probs, self.repr_layers)
        seq_base_processed = []
        missing_embeddings = []
        for i, (seq_name, seq_base, seq_mut, mut) in enumerate(zip(inputs_all['base']['name'], inputs_all['base']['sequence'], inputs_all['mut']['sequence'], inputs_all['mut']['mutations'])):
            mut_list = mut.split(self.sep)
            print(f'[{i}] {seq_name} ({len(seq_base)}) >> Mutations: {mut_list}; Sequence: {seq_mut}')

            ### LLR & entropy features ###
            # get LLR and entropy plots for each individual seq_base
            if self.get_mutation_logits_probs is not None and seq_base not in seq_base_processed:

                # fetch probability, entropy data for sequence
                mutprobs_fpath = f'{self.conservation_analysis_dir}{self.model_type}-{self.layer_str}/{seq_name}_{self.model_type}-{self.layer_str}_MutProbs.csv'
                df = pd.read_csv(mutprobs_fpath, index_col=0)
                # MutProbs sequence contains no gaps >> need to update pos_list with residue numbers reflecting gaps (and mutation position labels) in sequence, if found
                pos_list = np.array([i+1 for i, wt_aa in enumerate(seq_base) if wt_aa!='-'])
                df['RealPos'] = pos_list
                wt_aa_list = [seq_base[pos-1] for pos in pos_list]
                probs = df.set_index('RealPos').loc[:,[c for c in df.columns.tolist() if c not in ['RealPos','AA','entropy','pppl']]].transpose()
                entropy_values = df['entropy'].to_numpy()

                if 'X' in wt_aa_list:
                    aa_list = aaList_with_X
                else:
                    aa_list = aaList

                # calculate LLR and plot heatmap for LLR and linegraph for entropy
                savefig = f'{self.conservation_analysis_dir}{self.model_type}-{self.layer_str}/{seq_name}_{self.model_type}-{self.layer_str}'
                llr_map, res_w_positiveLLR = get_log_likelihood_ratios(probs, seq_base, plot_heatmap=get_llr_entropy_plot, savefig=savefig+'_LLR.png', seq_name=seq_name)
                plt.plot(pos_list, entropy_values); plt.title(f'Entropy for {seq_name} ({self.model_type})')
                plt.savefig(savefig+'_entropy.png'); plt.close()
                print('Obtained LLRs for all variants.')

                # convert LLR & entropies to flattened array --> save each individual sequence results as CSV
                llr_vect, llr_vect_labels = flatten_2D_arr(llr_map, seq_base, MT_aa=aa_list)
                entropy_vect = np.tile(entropy_values, (len(aa_list),1)).flatten('F')
                llr_entropy_seqbase = pd.DataFrame({
                    'name': [f'{seq_name}_{mut}' for mut in llr_vect_labels],
                    'protein_name':seq_name,
                    'mutations':llr_vect_labels,
                    'entropy': entropy_vect,
                    'LLR':llr_vect
                })

                # save as CSV
                llr_entropy_seqbase.to_csv(f'{self.conservation_analysis_dir}{self.model_type}-{self.layer_str}/{seq_name}_{self.model_type}-{self.layer_str}_LLR_entropy.csv')

                # update seq_base_processed
                seq_base_processed.append(seq_base)
                llr_entropy_dict_byseqbase[seq_base] = llr_entropy_seqbase

            # record entry for this particular mutation for overall dataset across all sequences
            if self.get_mutations and self.get_mutation_logits_probs is not None:
                llr_entropy_seqbase = llr_entropy_dict_byseqbase[seq_base]
                llrsum = 0
                entropy_avg = 0
                for mutation in mut_list:
                    print('Target mut:', mutation, '; WT aa:', seq_base[int(mutation[1:-1])-1]+mutation[1:-1])
                    # print(llr_entropy_seqbase)
                    llr_entropy_seqbase_mut = llr_entropy_seqbase[llr_entropy_seqbase['mutations']==mutation].iloc[0]
                    llrsum += llr_entropy_seqbase_mut['LLR']
                    entropy_avg += llr_entropy_seqbase_mut['entropy']
                entropy_avg /= len(mut_list)
                llrsum_entropy_mutants.append({'name': f'{seq_name}_{mut}', 'protein_name':seq_name, 'mutations':mut, 'LLR':round(llrsum,4), 'entropy':round(entropy_avg,4)})

            ### Embedding features ###
            try:
                if self.get_embeddings['res_avg'] or self.get_embeddings['res_mut']:
                    # load the saved files
                    fname_base = f"{seq_name}_{self.model_type}-{self.layer_str}"
                    fname_mut = f"{seq_name}_{mut}_{self.model_type}-{self.layer_str}"
                    ## try untruncated seq filename (if full embedding could be fetched without requiring sequence truncation)
                    try:
                        fname_suffix = ''
                        result_torch_base = torch.load(self.emb_folder + fname_base + fname_suffix + '.pt')
                        result_torch_mut = torch.load(self.emb_folder + fname_mut + fname_suffix + '.pt') if mut is not None else None
                    ## try truncated seq filename (if sequence had to be truncated while centered at mut_pos_center to fetch the embedding)
                    except Exception as e:
                        print(e)
                        mut_pos_center = int(np.mean(np.array([int(m[1:-1]) for m in mut.split(',')])))
                        fname_suffix = '_trunc' + str(mut_pos_center)
                        result_torch_base = torch.load(self.emb_folder + fname_base + fname_suffix + ".pt")
                        result_torch_mut = torch.load(self.emb_folder + fname_mut + fname_suffix + ".pt") if mut is not None else None
                    print(f'Loaded embeddings from {fname_base + fname_suffix + ".pt"}, {fname_mut + fname_suffix + ".pt"}.')

                    # get position offsets
                    if 'seq_start_offset' in result_torch_base:
                        seq_start_offset_base = result_torch_base['seq_start_offset']
                    else:
                        seq_start_offset_base = 0
                    if 'seq_start_offset' in result_torch_mut:
                        seq_start_offset_mut = result_torch_mut['seq_start_offset']
                    else:
                        seq_start_offset_mut = 0
                    print(f'Position offsets for residue-level embeddings: [base] {seq_start_offset_base}, [mut] {seq_start_offset_mut}')

                    # get embeddings for each layer
                    for layer in self.repr_layers:

                        # get residue-level embeddings for mutated positions, base and mut sequence
                        if self.get_embeddings['res_mut'] and result_torch_mut is not None:
                            # get full embeddings
                            res_embeddings_base = result_torch_base['full_representations'][layer].numpy()
                            res_embeddings_mut = result_torch_mut['full_representations'][layer].numpy()
                            # initialize arrays and lists
                            res_embeddings_concat = []
                            res_embeddings_base_mutpos = []
                            res_embeddings_mut_mutpos = []
                            # iterate through mutations present (e.g. if combi)
                            for mutation in mut_list:
                                mut_pos = int(mutation[1:-1])
                                # append residue embeddings for base sequence
                                res_embeddings_base_mutpos.append(list(res_embeddings_base[mut_pos-seq_start_offset_base-1,:]))
                                # append residue embeddings for mutated sequence
                                res_embeddings_mut_mutpos.append(list(res_embeddings_mut[mut_pos-seq_start_offset_mut-1,:]))
                            # update mut_embs data array with concatenated (averaged) mutated position vectors from base and mut
                            res_embeddings_concat += list(np.mean(np.array(res_embeddings_base_mutpos), axis=0))
                            res_embeddings_concat += list(np.mean(np.array(res_embeddings_mut_mutpos), axis=0))
                            mut_embs[layer].append(res_embeddings_concat)

                        # get GLOBAL AVERAGE embedding, base and mut sequence
                        if self.get_embeddings['res_avg']:
                            seq_embeddings_concat = []
                            # append global average embeddings for base sequence
                            seq_embeddings_base = result_torch_base['mean_representations'][layer].numpy()
                            seq_embeddings_concat += list(seq_embeddings_base)
                            # append global average embeddings for base sequence
                            if result_torch_mut is not None:
                                seq_embeddings_mut = result_torch_mut['mean_representations'][layer].numpy()
                                seq_embeddings_concat += list(seq_embeddings_mut)
                            # update seq_embs data array
                            seq_embs[layer].append(seq_embeddings_concat)

            # unable to process embedding for this mutation
            except Exception as e:
                print(e)
                print('ERROR obtaining embedding for:', (i, seq_name, mut))
                missing_embeddings.append((i, seq_name, mut))

        print('\n', 'Missing embeddings:')
        for missing_embedding in missing_embeddings: print(missing_embedding)
        print()

        #######################
        # save results to CSV #
        #######################
        if self.get_mutations and self.get_mutation_logits_probs is not None:
            llrsum_entropy_mutants = pd.DataFrame(llrsum_entropy_mutants)
            llrsum_entropy_mutants.to_csv(self.llrsum_entropy_mutants_fpath)

        if self.get_embeddings['res_mut']:
            for layer in self.repr_layers:
                # WT/MT concat
                mut_embs_arr = np.array(mut_embs[layer])
                mut_emb_cols = [f'{self.model_type}-{layer}_{i}' for i in list(range(1, len(seq_embeddings_base)+1))]
                if len(mut_emb_cols)<mut_embs_arr.shape[1]:
                    mut_emb_cols = [c+'_WT' for c in mut_emb_cols] + [c+'_MT' for c in mut_emb_cols]
                mut_embs_layer = pd.DataFrame(mut_embs_arr, index=inputs_all['base']['name'], columns=mut_emb_cols)
                # MT only
                mut_embs_layer_MT = mut_embs_layer.loc[:, [c for c in mut_emb_cols if c.find('MT')>-1]]
                # DIFF df
                p = int(mut_embs_layer.shape[1]/2)
                mut_embs_diff = mut_embs_layer.iloc[:,p:].to_numpy() - mut_embs_layer.iloc[:,:p].to_numpy()
                mut_embs_diff = pd.DataFrame(mut_embs_diff, columns=[f'{self.model_type}-{layer}_{i}_diff' for i in list(range(1, p+1))])
                # save dataframes
                mut_embs_layer.insert(0, 'mutations', inputs_all['mut']['mutations'])
                mut_embs_layer.to_csv(self.mut_embeddings_fpath)
                mut_embs_layer_MT.insert(0, 'mutations', inputs_all['mut']['mutations'])
                mut_embs_layer_MT.to_csv(self.mut_embeddings_fpath.replace('.csv','_MT.csv'))
                mut_embs_diff.insert(0, 'mutations', inputs_all['mut']['mutations'])
                mut_embs_diff.to_csv(self.mut_embeddings_fpath.replace('.csv','_diff.csv'))
                print('Mutation embeddings:', mut_embs_layer.shape)

        if self.get_embeddings['res_avg']:
            for layer in self.repr_layers:
                # WT/MT concat
                seq_embs_arr = np.array(seq_embs[layer])
                seq_emb_cols = [f'{self.model_type}-{layer}_{i}' for i in list(range(1, len(seq_embeddings_base)+1))]
                if len(seq_emb_cols)<seq_embs_arr.shape[1]:
                    seq_emb_cols = [c+'_WT' for c in seq_emb_cols] + [c+'_MT' for c in seq_emb_cols]
                seq_embs_layer = pd.DataFrame(seq_embs_arr, index=inputs_all['base']['name'], columns=seq_emb_cols)
                # MT only
                seq_embs_layer_MT = seq_embs_layer.loc[:, [c for c in seq_emb_cols if c.find('MT')>-1]]
                # DIFF df
                p = int(seq_embs_layer.shape[1]/2)
                seq_embs_diff = seq_embs_layer.iloc[:,p:].to_numpy() - seq_embs_layer.iloc[:,:p].to_numpy()
                seq_embs_diff = pd.DataFrame(seq_embs_diff, columns=[f'{self.model_type}-{layer}_{i}_diff' for i in list(range(1, p+1))])
                # save dataframes
                seq_embs_layer.insert(0, 'mutations', inputs_all['mut']['mutations'])
                seq_embs_layer.to_csv(self.seq_embeddings_fpath)
                seq_embs_layer_MT.insert(0, 'mutations', inputs_all['mut']['mutations'])
                seq_embs_layer_MT.to_csv(self.seq_embeddings_fpath.replace('.csv','_MT.csv'))
                seq_embs_diff.insert(0, 'mutations', inputs_all['mut']['mutations'])
                seq_embs_diff.to_csv(self.seq_embeddings_fpath.replace('.csv','_diff.csv'))
                print('Sequence embeddings:', seq_embs_layer.shape)
        
    
def get_pLM_zeroshot_scores(
        model_type, 
        data_subfolder,
        input_file,
        msa_fpath,
        get_mutations,
        get_embeddings, 
        deduplicate_sequences=True,
        skip_plm_processing=False,
        skip_logits_features=False,
        batch_size=1,
        csv_name_col='protein_name',
        sep='+'
):
    
    # initialize PLMZero class
    plm_zeroshot = PLMZero(
        model_type,
        msa_fpath,
        data_folder,
        data_subfolder,
        get_mutations,
        get_embeddings,
        deduplicate_sequences,
        skip_plm_processing,
        skip_logits_features
        )

    # get PLM inputs
    inputs_all = plm_zeroshot.get_inputs(input_file, batch_size, csv_name_col, sep)

    # process through pLM
    plm_zeroshot.run_pLM_processing(inputs_all)

    # parse results for each sequence in inputs_all
    plm_zeroshot.parse_plm_results(inputs_all, get_llr_entropy_plot=False)



if __name__=='__main__':
    
    # set filepaths
    model_type = 'esm2'
    data_subfolder = '' # 'GOh1052_mutagenesis' 
    input_file = 'expdata/GOh1052_mutagenesis/GOh1052_mutagenesis.csv' 
    msa_fpath = None # data_folder + subfolders['msa'] + 'UPO_aligned_clustalo.fasta' #
    
    # set parameters
    get_mutations=True
    get_embeddings = {'res_all': True, 'res_mut': True,'res_avg': True}
    deduplicate_sequences = True
    skip_plm_processing=False
    skip_logits_features=False
    batch_size = 1
    csv_name_col = 'protein_name'
    sep = '+'

    # run pipeline
    get_pLM_zeroshot_scores(
        model_type, 
        data_subfolder,
        input_file,
        msa_fpath,
        get_mutations,
        get_embeddings, 
        deduplicate_sequences,
        skip_plm_processing,
        skip_logits_features,
        batch_size,
        csv_name_col,
        sep
    )


