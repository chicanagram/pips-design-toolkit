import pandas as pd
from utils.variables import aaList
from utils.utils import fetch_sequences_from_fasta, list_all_mutations, get_mutated_sequence, mkDir

def get_inputs_from_fasta(input_fpath, get_mutations, sep='+'):
    inputs_all = {'base': {}, 'mut': {}}
    sequence_list, seq_name_list, _ = fetch_sequences_from_fasta(input_fpath)
    # if fasta file contains only 1 base sequence (without variants specified), get all possible single site mutations
    if len(sequence_list)==1:
        seq_base = sequence_list[0]
        seq_name_base = seq_name_list[0]
        mutations_all = list_all_mutations(seq_base, ignore_mutations_to_WT=True)
        print('mutations_all', len(mutations_all), mutations_all)
        mutations_list, seq_name_list, sequence_list, _ = get_mutated_sequence(seq_base, mutations_all, seq_name_base=seq_name_base, write_to_fasta=None, sep=sep)

    # set inputs
    inputs_all['base']['sequence'] = sequence_list
    if get_mutations:
        mutations_list = [seq_name.split('_')[-1] for seq_name in seq_name_list]
        base_name_list = [seq_name.split('_')[0] for seq_name in seq_name_list]
        inputs_all['mut'].update({
            'name': base_name_list,
            'mutations': mutations_list,
            'sequence': sequence_list,
        })
        inputs_all['base']['name'] = base_name_list

        # Generate base sequences (reverse mutation from mutated sequence) and mutations
        print('mutations_list', len(mutations_list), mutations_list)
        if len(inputs_all['mut']['mutations']) > 0:
            # MODIFY TO CATER TO MULTIPLE SEQUENCES
            seq_base_list = []
            for seq_mut, mutstr in zip(sequence_list, mutations_list):
                muts = mutstr.split(sep)
                muts_rev = [mut[-1]+mut[1:-1]+mut[0] for mut in muts]
                seq_base = get_mutated_sequence(seq_mut, muts_rev, sep=sep)[2][0]
                seq_base_list.append(seq_base)
            inputs_all['base']['sequence'] = seq_base_list
            inputs_all['base']['mutations'] = mutations_list
    else:
        inputs_all['base']['mutations'] = [None] * len(inputs_all['base']['name'])
        inputs_all['mut'].update({
            'mutations': [None] * len(inputs_all['base']['name']),
            'sequence': [None] * len(inputs_all['base']['name']),
            'name': [None] * len(inputs_all['base']['name'])
        })
    return inputs_all

def get_inputs_from_csv(input_fpath, get_mutations, csv_name_col='name'):
    inputs_all = {'base': {}, 'mut': {}}
    input_df = pd.read_csv(input_fpath)
    base_name_list = input_df[csv_name_col].tolist()
    mutations_list = input_df['mutations'].tolist()
    print(f'CSV mutations list: {len(mutations_list)} mutants ({len(list(set(mutations_list)))} unique)')
    if get_mutations:
        # get base sequences
        inputs_all['base']['name'] = base_name_list
        inputs_all['base']['sequence'] = input_df['sequence_base'].tolist()
        inputs_all['base']['mutations'] = mutations_list
        # get mutated sequences
        inputs_all['mut'].update({
            'name': base_name_list,
            'mutations': mutations_list,
            'sequence': input_df['sequence'].tolist(),
        })
    else:
        # get base sequences
        inputs_all['base']['name'] = base_name_list
        inputs_all['base']['sequence'] = input_df['sequence'].tolist()
        inputs_all['base']['mutations'] = [None] * len(inputs_all['base']['name'])
        # no mut sequences
        inputs_all['mut'].update({
            'mutations': [None] * len(inputs_all['base']['name']),
            'sequence': [None] * len(inputs_all['base']['name']),
            'name': [None] * len(inputs_all['base']['name'])
        })
    return inputs_all

def get_plm_pipeline_inputs(input_fpath, get_mutations, subset_idx=None, csv_name_col='name', sep='+'):
    # get sequences from fasta file
    if input_fpath.find('.fasta')>-1:
        print('Obtaining inputs from fasta file...')
        inputs_all = get_inputs_from_fasta(input_fpath, get_mutations, sep=sep)
    # get sequences from csv file
    else:
        print('Obtaining inputs from csv file...')
        inputs_all = get_inputs_from_csv(input_fpath, get_mutations, csv_name_col=csv_name_col)

    # set mut to None for WT sequences
    print('# of WT sequences:', len([mut for mut in inputs_all['mut']['mutations'] if mut=='WT']))
    inputs_all['mut']['mutations'] = [mut if mut!='WT' else None for mut in inputs_all['mut']['mutations']]

    # filter to process only a subset of samples
    if subset_idx is not None:
        for base_or_mut in ['base', 'mut']:
            for k in ['name', 'mutations', 'sequence']:
                inputs_all[base_or_mut][k] = inputs_all[base_or_mut][k][subset_idx:]
    print('Obtained inputs.')
    return inputs_all

def include_all_substitutions_for_mutated_positions(inputs_all_base_or_mut):
    """
    Expand mutations list for each position mutated to include all possible substitution
    """
    # iterate through base sequences
    for i, (seq_base, mutations) in enumerate(zip(inputs_all_base_or_mut['sequence'], inputs_all_base_or_mut['mutations'])):
        # get mutated positions
        if mutations is None:
            mutations_all = list_all_mutations(seq_base)
            inputs_all_base_or_mut['mutations'][i] = mutations_all
        else:
            pos_list = list(set(list([int(mut[1:-1]) for mut in mutations])))
            if 'WT' in pos_list:
                pos_list.remove('WT')
            mutations_expanded = []
            for pos in pos_list:
                WT_aa = seq_base[pos-1]
                mutations_expanded += [WT_aa + str(pos) + aa for aa in aaList if aa!=WT_aa]
            inputs_all_base_or_mut['mutations'][i] = mutations_expanded
    return inputs_all_base_or_mut

def initialize_plm_feature_variables(get_embeddings, get_mutations, get_mutation_logits_probs, repr_layers):
    seq_embs, mut_embs, llrsum_entropy_mutants, llr_entropy_dict_byseqbase = None, None, None, None

    if get_embeddings['res_avg']:
        seq_embs = {layer:[] for layer in repr_layers}

    if get_embeddings['res_mut']:
        mut_embs = {layer:[] for layer in repr_layers}

    if get_mutation_logits_probs is not None:
        llr_entropy_dict_byseqbase = {}
        # record sum of LLR and entropy for all mutants encountered in input file
        if get_mutations:
            llrsum_entropy_mutants = []

    # return res_embeddings_fpath, seq_embeddings_fpath, seq_embs, mut_embeddings_fpath, mut_embs, mutprob_fpath, llrsum_entropy_mutants, llrsum_entropy_mutants_fpath, llr_entropy_dict_byseqbase
    return seq_embs, mut_embs, llrsum_entropy_mutants, llr_entropy_dict_byseqbase
