import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.width',1000)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from utils.utils import fetch_sequences_from_fasta, list_all_mutations, variant_scores_vect_to_matrix, sort_variants_by_position, get_mutated_sequence
from utils.variables import data_folder, subfolders, aaList, amino_acid_groups, variant_proposal_scores_base


aa_to_grp_mapping = {}
for aa_grp_label, aa_grp_list in amino_acid_groups.items():
    for aa in aa_grp_list:
        aa_to_grp_mapping[aa] = aa_grp_label

class VariantSelection:
    def __init__(
            self,
            data_folder,
            data_fbase,
            fname_suffix,
            seq_fname,
            sequences_subfolder='sequences/',
            mutagenesis_proposal_subfolder='mutagenesis_proposal/',
            plm_dir='../prot-embeddings/',
            remove_entries_with_nan=False

    ):
        # set folder names
        self.data_folder = data_folder
        self.data_fbase = data_fbase
        self.fname_suffix = fname_suffix
        self.sequences_subfolder = sequences_subfolder
        self.mutagenesis_proposal_subfolder = mutagenesis_proposal_subfolder
        self.plm_dir = plm_dir

        # import PLM inference code
        if self.plm_dir not in sys.path:
            sys.path.append(self.plm_dir)

        # get WT sequence & initialize dataframe
        seq, seq_name, _ = fetch_sequences_from_fasta(self.data_folder+self.sequences_subfolder+seq_fname)
        self.seq_base = seq[0]
        self.seq_name_base = seq_name[0].replace('_WT','')

        # other settings
        self.remove_entries_with_nan = remove_entries_with_nan

    def combine_scores(self, features_to_get, plot_variant_heatmap=False, calc_diff_wrt_WT=True):

        # get all possible single-site mutations
        mut_all = list_all_mutations(self.seq_base, ignore_mutations_to_WT=True)
        pos_list = [int(mut[1:-1]) for mut in mut_all]
        wt_aa = [mut[0] for mut in mut_all]
        mt_aa = [mut[-1] for mut in mut_all]
        wt_aa_grp = [aa_to_grp_mapping[aa] for aa in wt_aa]
        mt_aa_grp = [aa_to_grp_mapping[aa] for aa in mt_aa]
        scores_all = pd.DataFrame({'Position':pos_list, 'mutations': mut_all, 'wt_aa':wt_aa, 'mt_aa':mt_aa, 'wt_aa_grp':wt_aa_grp, 'mt_aa_grp':mt_aa_grp})

        for i, f in enumerate(features_to_get):
            print(i,f)
            feature_name = f['name']
            fpath = f['fpath']
            feature_col = f['feature_col']
            mut_col = f['mut_col']
            wt_inc, invert_scores = f['wt_inc'], f['invert_scores']
            df = pd.read_csv(fpath, index_col=False)

            # entropy scores by position
            if mut_col in ['RealPos', 'Position']:
                if 'Position' not in df:
                    df = df.rename(columns={'RealPos':'Position'})
                if isinstance(feature_col, str):
                    feature_col = [feature_col]
                scores_all = scores_all.merge(df[['Position']+feature_col], on='Position', how='left')

            # variant scores
            else:
                if wt_inc:
                    score_wt = df[df[mut_col] == 'WT'][feature_col].mean(axis=0)
                else:
                    score_wt = 0
                df_muts = df[df[mut_col] != 'WT']
                scores = np.concatenate([np.array([score_wt]), df_muts[feature_col].to_numpy()], axis=0)
                if invert_scores:
                    scores = -scores
                # calculate scores wrt WT
                scores_vs_WT = scores
                if calc_diff_wrt_WT:
                    scores_vs_WT = scores - score_wt
                variants = ['WT'] + df_muts[mut_col].tolist()
                # append scores to main dataframe
                df_append = pd.DataFrame({feature_name: scores_vs_WT})
                # df_append = pd.DataFrame({feature_col: scores_vs_WT})
                df_append.insert(0, 'mutations', variants)
                scores_all = scores_all.merge(df_append, on='mutations', how='left')

                # plot variant heatmap
                if plot_variant_heatmap:
                    savefig = fpath.replace('.csv', '.jpg')
                    figtitle = f'{feature_name} scores vs WT for all single-site variants'
                    scores_matrix, diff_scores_matrix = variant_scores_vect_to_matrix(scores, self.seq_base, variants,
                                                                                      remove_nan_pos=False, plot_heatmap=savefig,
                                                                                      figtitle=figtitle)
                    print(f'# of NaN scores: {len(np.where(np.isnan(diff_scores_matrix))[0])}/{len(self.seq_base) * 20}')

        # remove rows with nan
        if self.remove_entries_with_nan:
            scores_all = scores_all.dropna(ignore_index=True)

        return scores_all


    def filt_df_by_feature(self, df_filt, feature_name, pos_filter_dict):
        n_pre = len(df_filt)
        (cutoff, filt_direction, remove_by_thres_or_fraction) = pos_filter_dict[feature_name]
        
        # get removal threshold
        if remove_by_thres_or_fraction == 'thres':
            thres = cutoff
        elif remove_by_thres_or_fraction=='frac':
            min_num_mut_to_remove = int(cutoff*n_pre)
            thres = df_filt.sort_values(by=feature_name, ascending=True).iloc[min_num_mut_to_remove-1][feature_name]
        
        # perform filtering
        if filt_direction=='>':
            df_filt = df_filt[df_filt[feature_name] > thres]
        elif filt_direction=='<':
            df_filt = df_filt[df_filt[feature_name] < thres]
        n_post = len(df_filt)
        n_removed = n_pre-n_post
        print(f"Removed {n_removed}/{n_pre} ({round(n_removed/n_pre*100,2)}%) of mutations by filtering on the score: {feature_name} {filt_direction} {thres}")
        
        return df_filt
        

    def filter_scores_by_position(self, df, pos_filter_dict):
        
        df_pos_deduped = df.drop_duplicates(subset=['Position'])
        pos_list_init = df_pos_deduped['Position'].tolist()
        num_pos_init = len(pos_list_init)

        df_filt = df.copy()
        for feature_name in pos_filter_dict:
            if feature_name in df_filt:
                df_filt = self.filt_df_by_feature(df_filt, feature_name, pos_filter_dict)
                
        # tally remaining positions at the end
        pos_list_filt = list(set(df_filt['Position'].tolist()))
        pos_removed = [pos for pos in pos_list_init if pos not in pos_list_filt]
        print(f'{len(pos_removed)}/{num_pos_init} ({round(len(pos_removed)/num_pos_init*100,2)}%) positions removed after filtering on conservation & structural scores:', pos_removed)
        print(f'{len(pos_list_filt)}/{num_pos_init} positions remaining:', ', '.join([str(pos) for pos in pos_list_filt]))
        print()
        return df_filt, pos_list_filt


    def filter_scores_by_variant(self, df, variant_filter_dict=None, max_variants_per_position=2, col_to_sort_by='avg_PLM_score'):
        mut_list_init = list(set(df['mutations'].tolist()))
        df_filt = df.copy()

        for feature_name in variant_filter_dict:
            if feature_name in df_filt:
                df_filt = self.filt_df_by_feature(df_filt, feature_name, variant_filter_dict)

        # tally remaining mutations at the end
        mut_list = df_filt['mutations'].tolist()
        pos_list = list(set([int(mut[1:-1]) for mut in mut_list]))
        mut_list_removed = [mut for mut in mut_list_init if mut not in mut_list]
        print(f'{len(mut_list_removed)}/{len(mut_list_init)} ({round(len(mut_list_removed)/len(mut_list_init)*100,2)}%) variants removed after filtering by {", ".join(list(variant_filter_dict.keys()))}.')
        print(f'{len(mut_list)}/{len(mut_list_init)} mutations remaining:', *mut_list)
        print(f'# of positions mutated: {len(pos_list)}:', *pos_list)

        # sort variants by position; remove if exceed max number per position
        mut_list_removed = []
        pos_mut_dict = sort_variants_by_position(mut_list)
        pos_mut_dict_filt = {}
        pos_mut_avg_PLM_scores = {}
        for pos, mut_list_bypos in pos_mut_dict.items():
            if len(mut_list_bypos)>max_variants_per_position:
                mut_list_removed += mut_list_bypos[max_variants_per_position:]
                pos_mut_dict_filt[pos] = mut_list_bypos[:max_variants_per_position]
            else:
                pos_mut_dict_filt[pos] = mut_list_bypos
            pos_mut_avg_PLM_scores[pos] = float(df_filt.loc[df_filt['mutations'].isin(mut_list_bypos), col_to_sort_by].mean())
            print(f'[{pos}]', *pos_mut_dict_filt[pos])

        # update mut list and df_filt
        mut_list = [mut for mut in mut_list if mut not in mut_list_removed]
        pos_list = list(set([int(mut[1:-1]) for mut in mut_list]))
        df_filt = df_filt[~df_filt['mutations'].isin(mut_list_removed)].reset_index(drop=True)
        
        # update with column for average PLM scores for mutants selected for each position
        df_filt['avg_PLM_score_bypos'] = np.nan
        for pos, avg_plm_score_bypos in pos_mut_avg_PLM_scores.items():
            df_filt.loc[df_filt['Position']==pos, 'avg_PLM_score_bypos'] = avg_plm_score_bypos
        df_filt = df_filt.sort_values(by='avg_PLM_score_bypos', ascending=True)
        df_filt = df_filt.round(4)
        # print results
        print(f'{len(mut_list_removed)}/{len(mut_list_init)} ({round(len(mut_list_removed)/len(mut_list_init)*100,2)}%) variants removed after limiting the number of mutations per position to {max_variants_per_position}.')
        print(f'{len(mut_list)}/{len(mut_list_init)} mutations remaining:', ', '.join(mut_list))
        print(f'# of positions mutated: {len(pos_list)}:', pos_list)
        print()
        return df_filt, mut_list, pos_mut_dict_filt

    def get_plm_scores(self, mut_combi_list, model_name='PoET2'):

        # get full sequence for all combi variants
        _, sequence_names, sequences, _ = get_mutated_sequence(self.seq_base, mut_combi_list, seq_name_base=self.seq_name_base, write_to_fasta=None)
        sequence_names = [self.seq_name_base+'_WT'] + sequence_names
        sequences = [self.seq_base] + sequences

        # get PLM scores
        if model_name=='PoET2':
            from get_PoET2_outputs import PoET2
            plm = PoET2(data_fbase, data_folder, num_ensemble_prompts=5, random_seed=46)

        # score sequences
        scores = plm.score_sequences(
            sequence_names,
            sequences,
            msa_seed=None,
            divide_score_by_seqlen=True,
            get_avg_norm_score=True,
            save_csv=None
        )
        return scores

    def remove_excess_mutation_occurrences(self, selection_bycombi, num_variants_bycombi, max_mutation_occurrences, all_mutations, mut_combi_list_ordered):
        """
         replace combis with mutations that occur above the threshold number of times
         """
        # get combis to reject
        final_selection_bycombi = []
        selection_bycombi_filt = []
        rejected_combis = []
        mutation_occurrence_count = {mut:0 for mut in all_mutations}
        for combi in selection_bycombi:
            if len(final_selection_bycombi) < num_variants_bycombi:
                # check if max_mutation_occurrences has not been reached yet for all mutations in this combi
                if all([mutation_occurrence_count[mut]<max_mutation_occurrences for mut in combi]):
                    # accept combi
                    final_selection_bycombi.append(combi)
                    # update counts
                    for mut in combi:
                        mutation_occurrence_count[mut] += 1
                else:
                    rejected_combis.append(combi)
            else:
                break
        combis_to_sample_from = [combi for combi in mut_combi_list_ordered if (combi not in final_selection_bycombi and combi not in rejected_combis)]
        print('# rejected_combis:', len(rejected_combis))
        print('# target variants:', num_variants_bycombi)
        print('# final_selection_bycombi:', len(final_selection_bycombi))
        print('# of variants to resample:', num_variants_bycombi-len(final_selection_bycombi))
        print('# combis_to_sample_from:', len(combis_to_sample_from))

        # resample combis from remaining list to plug shortage due to rejected selections
        while len(final_selection_bycombi) < num_variants_bycombi and len(combis_to_sample_from)>0:
            for combi in combis_to_sample_from:
                if all([mutation_occurrence_count[mut] < max_mutation_occurrences for mut in combi]):
                    # accept combi
                    print(f'{combi} added to selection.')
                    final_selection_bycombi.append(combi)
                    # update counts
                    for mut in combi:
                        mutation_occurrence_count[mut] += 1
                    print(f'Added {combi}.', end=' ')
                else:
                    rejected_combis.append(combi)
                # update list of combis to sample from
                combis_to_sample_from.remove(combi)
        mutations_not_selected = [mut for mut in all_mutations if mutation_occurrence_count[mut] == 0]
        print('\n', f'Resampled combis from remaining list. # final_selection_bycombi: {len(final_selection_bycombi)}; # additional variants to sample: {num_variants_bycombi-len(final_selection_bycombi)}.')
        print(f'{len(mutations_not_selected)} mutations not selected:', *mutations_not_selected)

        # if still not enough selections, then sample from reject list


        print('mutation_occurrence_count:')
        print(pd.DataFrame(mutation_occurrence_count, index=['count']))
        return final_selection_bycombi, max_mutation_occurrences, mutations_not_selected, combis_to_sample_from

    def get_weights_from_scores(self, scores_w_WT, power=1):
        # convert averaged normalized scores to weights -- weights for all possible variants should add to 1
        scores_wo_WT = scores_w_WT[1:]
        # perform min-max normalization (between nonzero_offset & 1)
        nonzero_offset = np.diff(np.sort(scores_wo_WT)[:2])[0] / 2 # half of difference between two minimum scores --> so that lowest score >0
        scores_w_WT -= scores_wo_WT.min()
        scores_w_WT += nonzero_offset
        scores_w_WT /= scores_wo_WT.max()
        scores_w_WT[0] = 0
        print(f'scores_w_WT: [MIN] {scores_w_WT.min()}; [MAX] {scores_w_WT.max()}')

        # apply power to transform relative scores
        scores_w_WT_power = np.power(scores_w_WT, power)
        scores_wo_WT_power = scores_w_WT_power[1:]
        print(f'scores_w_WT_power: [MIN] {scores_w_WT_power.min()}; [MAX] {scores_w_WT_power.max()}')

        # divide by sum of all scores
        weights_w_WT = scores_w_WT / scores_wo_WT.sum()
        weights_w_WT_power = scores_w_WT_power / scores_wo_WT_power.sum()
        return weights_w_WT, weights_w_WT_power


    def run_ss_pipeline(self,
                     features_to_get,
                     pos_filter_dict,
                     variant_filter_dict,
                     max_variants_per_position=2,
                     weighting_method='uniform',
                     plot_variant_heatmap = False,
                     plot_histogram_scores = False,
                     remove_mutations_or_positions = []
                     ):

        #######################################
        # GET SCORES FOR INDIVIDUAL MUTATIONS #
        # perform filtering #
        #######################################
        # combine scores
        print('Combining scores from different calculations...')
        scores_all = self.combine_scores(features_to_get, plot_variant_heatmap)
        scores_all_fpath = f"{self.data_folder}{subfolders['mutagenesis_proposal']}{self.data_fbase}_AllScores.csv"
        scores_all.to_csv(scores_all_fpath)
        print('Saved normalized scores to:', scores_all_fpath)

        # get normalized scores
        column_types = scores_all.dtypes.tolist()
        variant_score_first_feature_idx = [i for i,coltype in enumerate(column_types) if coltype=='float64'][0]
        max_abs_scores = np.nanmax(np.abs(scores_all.iloc[:, variant_score_first_feature_idx:].to_numpy()), axis=0)
        scores_norm = scores_all.copy()
        scores_norm.iloc[:, variant_score_first_feature_idx:] = scores_all.iloc[:, variant_score_first_feature_idx:] / max_abs_scores

        # get average PLM score and save csv
        scores_norm['avg_PLM_score'] = scores_norm[[plm for plm in ['ESM-2_LLR(-ive)', 'ProtT5_LLR(-ve)', 'PoET_LLR(-ve)', 'PoET2_LLR(-ve)'] if plm in scores_norm]].mean(axis=1).round(4)
        scores_norm = scores_norm.round(4)
        scores_norm_fpath = f"{self.data_folder}{subfolders['mutagenesis_proposal']}{self.data_fbase}_AllScoresNorm.csv"
        scores_norm.to_csv(scores_norm_fpath)
        print('Saved normalized scores to:', scores_norm_fpath)
        print()

        # plot average PLM score
        avg_PLM_scores = [0] + scores_norm['avg_PLM_score'].tolist()
        variants = ['WT'] + scores_norm['mutations'].tolist()
        savefig = f"{self.data_folder}{subfolders['mutagenesis_proposal']}{self.data_fbase}_avgPLMscore.png"
        figtitle = f'Avg PLM score for all single-site variants'
        _, _ = variant_scores_vect_to_matrix(np.array(avg_PLM_scores), self.seq_base, variants, remove_nan_pos=False, plot_heatmap=savefig, figtitle=figtitle)
        plt.close()

        # plot distribution of scores
        if plot_histogram_scores:
            hist = scores_norm.hist(
                column=['avg_PLM_score', 'Pythia_DDGstability', 'FoldX_DDGstability', 'YASARA_DDGbind', 'Tango_AggVsRef',
                        'Waltz_AggVsRef'], bins=100)
            plt.show()

        # filter by position
        print('Filtering positions...')
        scores_norm_filt, pos_list_filt = self.filter_scores_by_position(scores_norm, pos_filter_dict)

        # filter by variant score
        print('Filtering mutations...')
        scores_norm_filt, mut_list, pos_mut_dict_filt = self.filter_scores_by_variant(scores_norm_filt, variant_filter_dict, max_variants_per_position, col_to_sort_by='avg_PLM_score')

        # add column num_mutations
        scores_norm_filt.insert(0, 'num_mutation_sites', 1)
        scores_norm_filt = scores_norm_filt[[c for c in variant_proposal_scores_base if c in scores_norm_filt]]

        # remove mutations from list
        if len(remove_mutations_or_positions) > 0:
            mutations_to_remove = []
            for mut_or_pos in remove_mutations_or_positions:
                if isinstance(mut_or_pos,  int):
                    wt_aa = self.seq_base[mut_or_pos-1]
                    mutations_to_remove += [wt_aa + str(mut_or_pos) + aa for aa in aaList if aa!=wt_aa]
                elif isinstance(mut_or_pos, str):
                    mutations_to_remove.append(mut_or_pos)
            
            scores_norm_filt = scores_norm_filt[~scores_norm_filt['mutations'].isin(mutations_to_remove)]
            final_mutations = list(set(scores_norm_filt["mutations"].tolist()))
            final_positions_mutated = list(set(scores_norm_filt["Position"].tolist()))
            final_positions_mutated.sort()
            print('\n', f'After removing mutations: {len(final_mutations)} mutations remaining across {len(final_positions_mutated)} positions.')
            print('Mutations:', *final_mutations)
            print('Positions:', *final_positions_mutated)

        # save filtered single-site results
        scores_norm_filt.reset_index(drop=True).to_csv(f"{self.data_folder}{subfolders['mutagenesis_proposal']}{self.data_fbase}{self.fname_suffix}_singlesite_SELECTED.csv")

        return scores_norm_filt



if __name__=='__main__':
    data_fbase = 'GOh1052_R1'
    fname_suffix = '_ss'
    seq_fname = 'GOh1052.fasta'
    get_ss_variants = True
    plot_variant_heatmap = False
    plot_histogram_scores = False
    remove_mutations_or_positions = [217, 268] #
    remove_entries_with_nan = False
    
    features_to_get = [
        {'name': 'Distance', 'fpath': data_folder + subfolders['conservation_and_distance'] + 'GOh1052_distance.csv', 'mut_col': 'Position', 'feature_col': ['min_distance_to_UNK', 'avg_distance_UNK'], 'wt_inc': False, 'invert_scores': False},
        {'name': 'SIFT_norm', 'fpath': data_folder + subfolders['conservation_and_distance'] + 'GOh1052_sift.csv', 'mut_col': 'Position', 'feature_col': 'sift_avg', 'wt_inc': False, 'invert_scores': False},
        {'name': 'ShannonEntropy', 'fpath': data_folder + subfolders['conservation_and_distance'] + 'GOh1052_shanms.csv', 'mut_col': 'Position', 'feature_col': ['shanID', 'shanMS'], 'wt_inc': False, 'invert_scores': False},
        {'name': 'ESM-2_LLR(-ive)', 'fpath': data_folder + subfolders['conservation_and_distance'] + 'GOh1052_esm2-33_LLRsum_entropy.csv', 'mut_col': 'mutations', 'feature_col': 'LLR', 'wt_inc': False, 'invert_scores': True},
        # {'name': 'Tango_AggVsRef', 'fpath': data_folder + subfolders['aggregation'] + 'AggregationScore_GOh1052_tango.csv', 'mut_col': 'Enzyme/Mutation', 'feature_col': 'Aggregation', 'wt_inc': True, 'invert_scores': False},
        # {'name': 'Waltz_AggVsRef', 'fpath': data_folder + subfolders['aggregation'] + 'AggregationScore_GOh1052_waltz.csv', 'mut_col': 'Enzyme/Mutation', 'feature_col': 'Aggregation', 'wt_inc': True, 'invert_scores': False},
        # {'name': 'YASARA_DDGbind', 'fpath': data_folder + subfolders['yasara'] + 'Output/' + data_fbase + '/' + 'DDGbind_GOh1052.csv', 'mut_col': 'mutations', 'feature_col': 'ebindDDG', 'wt_inc': False, 'invert_scores': False},
        # {'name': 'FoldX_DDGstability', 'fpath': data_folder + subfolders['stability'] + data_fbase + '/' + 'Output/' + data_fbase + '/' + 'DDGstability_GOh1052.csv', 'mut_col': 'Mutation', 'feature_col': 'DDG', 'wt_inc': False, 'invert_scores': False},
    ]
    
    weighting_method = 'plm'  # 'uniform' #
    pos_filter_dict = {
        'sift_avg': (0.15, '>', 'frac'),
        # 'min_distance_to_UNK': (5, '<', 'thres'),
    }
    max_variants_per_position = 5
    variant_filter_dict = {
        'avg_PLM_score': (0.15, '<', 'thres'), 
        # 'Pythia_DDGstability': (0.1, '<', 'thres')
        # 'FoldX_DDGstability': (0.05, '<', 'thres'), 
        # 'Tango_AggVsRef': (0.05, '<', 'thres'),
        # 'Waltz_AggVsRef': (0.05, '<', 'thres'),
        # 'YASARA_DDGbind': (0.05, '<', 'thres'),
    }


    # initialize class
    variant_select = VariantSelection(data_folder, data_fbase, fname_suffix, seq_fname, remove_entries_with_nan=remove_entries_with_nan)
    # run selection pipeline
    selected_variants = variant_select.run_ss_pipeline(
        features_to_get,
        pos_filter_dict,
        variant_filter_dict,
        max_variants_per_position,
        weighting_method,
        plot_variant_heatmap,
        plot_histogram_scores,
        remove_mutations_or_positions,
    )