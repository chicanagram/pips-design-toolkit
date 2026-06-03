#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 07:58:44 2024

@author: charmainechia
"""
import os
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from utils.variables import data_folder, subfolders
from ml_prediction.model_features import feature_names


def apply_thresholds(value):
    if value <= 0.8:
        return -1
    elif value < 1.2:
        return 0
    else:
        return 1


def get_mut_diff(mutations, label1, label2):
    diff_mut_21 = [mut for mut in mutations[label1] if mut not in mutations[label2]]
    diff_mut_12 = [mut for mut in mutations[label2] if mut not in mutations[label1]]
    if len(diff_mut_21)>0:       
        print(f'diff mut ({label2} <> {label1}):', len(diff_mut_21))
    if len(diff_mut_12)>0:       
        print(f'diff mut ({label1} <> {label2}):', len(diff_mut_12))


def aggregate_features(feature_types_to_include, data_folder, data_subfolder, output_csv_fname, output_csv_dir, deduplicate_entries, WT_or_MT='MT', sep='+', fname_prefix=''):

    i = 0
    features_all = []
    mutations = {}

    ##############
    ## Y LABELS ##
    ##############
    if 'ylabel' in feature_types_to_include:
        dir = data_folder + subfolders['expdata']
        fname = feature_types_to_include['ylabel'][0]
        ylabel_list = feature_types_to_include['ylabel'][1]
        metadata_list = feature_types_to_include['ylabel'][2]
        # get grid-formatted (enzyme X substrate) results
        df = pd.read_csv(dir+fname)
        df = df[[c for c in metadata_list + ylabel_list if c in df.columns]]
        # add num_mutations if missing
        mutations_list = df['mutations'].tolist()
        mutations['ylabel'] = mutations_list
        if 'num_mutations' not in df:
            num_mutations_list = [len(mut.split(sep)) for mut in mutations_list]
            df['num_mutations'] = num_mutations_list
        print('expdata:', df.shape)
        # add Position
        positions = []
        for mutstr in mutations_list: 
            muts = mutstr.split(sep)
            pos = [mut[1:-1] for mut in muts]
            pos = ','.join(pos)
            if len(muts)==1:
                pos = int(pos)
            positions.append(pos)
        df['Position'] = positions

        # deduplicate entries
        if deduplicate_entries:
            from ml_prediction.datacleaning_utils import deduplicate_data_average_labels
            data_all = deduplicate_data_average_labels(df, 'mutations', ylabel_list, metadata_list, get_classification_label=False)
            print('expdata, deduped:', data_all.shape)

            # re-label averaged values of non-label columns
            col_nonlabel = [c for c in ylabel_list if c.find('label')==-1]
            for col in col_nonlabel:
                print(col)
                data_all[col.replace('foldchange','label')] =  data_all[col].apply(apply_thresholds)
        else:
            data_all = df.copy()
        features_all += metadata_list + ylabel_list
        i += 1

    #######################
    ## SEQUENCE FEATURES ##
    #######################
    if 'sequences' in feature_types_to_include:
        from utils.utils import fetch_sequences_from_fasta, get_mutated_sequence
        dir = data_folder + subfolders['sequences']
        fname = feature_types_to_include['sequences']

        seqs, seq_names, _ = fetch_sequences_from_fasta(dir + fname)
        seq_base = seqs[0]
        seq_name_base = seq_names[0]
        mutations_list = data_all['mutations'].tolist()
        _, _, sequence_list, _ = get_mutated_sequence(seq_base, mutations_list, seq_name_base=None, write_to_fasta=None, sep=sep)
        data_all['sequence'] = sequence_list
        data_all['sequence_base'] = [seq_base]*len(sequence_list)
        data_all.insert(0, 'protein_name', seq_name_base)
        data_all.insert(1, 'name', [f'{seq_name_base}_{mut}' for mut in mutations_list])
        features_all += ['protein_name', 'name', 'sequence', 'sequence_base']


    #############################
    ## YASARA BINDING FEATURES ##
    #############################
    if 'binding_yasara' in feature_types_to_include:
        # get feature names
        if WT_or_MT=='WT':
            cols = ['enz_name', 'sequence'] + feature_names['binding_nomut']
            features_all += feature_names['binding_nomut']
        elif WT_or_MT=='MT':
            cols = ['mutations'] + feature_names['binding_mut']
            features_all += feature_names['binding_mut']

        # load data
        dir = data_folder + subfolders['feature_extraction'] 
        fname = feature_types_to_include['binding_yasara']
        df = pd.read_csv(dir+fname)[cols]
        print('binding_yasara:', df.shape)

        # merge data
        if i==0:
            data_all = df.copy()
        else:
            data_all = data_all.merge(df, on='mutations', how='outer')
        i += 1
        print('Processed "binding_yasara" features >>', data_all.shape)

        mutations['yasara'] = df['mutations'].tolist()
        get_mut_diff(mutations, 'yasara', 'ylabel')

    ########################
    ## STABILITY FEATURES ##
    ########################
    if 'stability_foldx' in feature_types_to_include:
        # get feature names
        if WT_or_MT=='WT':
            cols = ['enz_name', 'sequence'] + feature_names['stability_foldx'] # to update
        elif WT_or_MT=='MT':
            cols = ['mutations'] + feature_names['stability_foldx']

        # load data
        dir = data_folder + subfolders['feature_extraction'] + data_subfolder + '/'
        fname = feature_types_to_include['stability_foldx']
        df = pd.read_csv(dir+fname)[cols]
        print('stability_foldx:', df.shape)

        # merge data
        if i==0:
            data_all = df.copy()
        else:
            data_all = data_all.merge(df, on='mutations', how='outer')
        i += 1
        features_all += feature_names['stability_foldx']
        print('Processed "stability_foldx" features >>', data_all.shape)

        mutations['stability'] = df['mutations'].tolist()
        get_mut_diff(mutations, 'stability', 'ylabel')

    if 'stability_pythia' in feature_types_to_include:
        # get feature names
        cols = ['mutations'] + feature_names['stability_pythia']

        # load data
        dir = data_folder + subfolders['feature_extraction'] + data_subfolder + '/'
        fname = feature_types_to_include['stability_pythia']
        df = pd.read_csv(dir+fname)[cols]
        print('stability_pythia:', df.shape)

        # merge data
        if i==0:
            data_all = df.copy()
        else:
            data_all = data_all.merge(df, on='mutations', how='outer')
        i += 1
        features_all += feature_names['stability_pythia']
        print('Processed "stability_pythia" features >>', data_all.shape)


    ####################
    ## TANGO FEATURES ##
    ####################
    if 'tango' in feature_types_to_include:
        cols = ['mutations'] + feature_names['tango'] + feature_names['tango_vs_ref']
        dir = data_folder + subfolders['feature_extraction']
        fname = feature_types_to_include['tango']
        df = pd.read_csv(dir + fname, index_col=False)
        df = df.rename(columns={'Aggregation':'Aggregation_tango', 'Aggregation_vs_ref': 'Aggregation_vs_ref_tango', 'Enzyme/Mutation': 'mutations'})
        df = df[[c for c in cols if c in df]]
        print('tango:', df.shape)

        # merge data
        if i == 0:
            data_all = df.copy()
        else:
            data_all = data_all.merge(df, on='mutations', how='outer')
        i += 1
        features_all += feature_names['tango'] + feature_names['tango_vs_ref']
        print('Processed "tango" features >>', data_all.shape)

        mutations['tango'] = df['mutations'].tolist()
        get_mut_diff(mutations, 'tango', 'ylabel')

    ####################
    ## WALTZ FEATURES ##
    ####################
    if 'waltz' in feature_types_to_include:
        cols = ['mutations'] + feature_names['waltz'] + feature_names['waltz_vs_ref']
        dir = data_folder + subfolders['feature_extraction']
        fname = feature_types_to_include['waltz']
        df = pd.read_csv(dir + fname, index_col=False)
        df = df.rename(columns={'Aggregation':'Aggregation_waltz', 'Aggregation_vs_ref':'Aggregation_vs_ref_waltz', 'Enzyme/Mutation': 'mutations'})
        df = df[[c for c in cols if c in df]]
        print('waltz:', df.shape)

        # merge data
        if i == 0:
            data_all = df.copy()
        else:
            data_all = data_all.merge(df, on='mutations', how='outer')
        i += 1
        features_all += feature_names['waltz'] + feature_names['waltz_vs_ref']
        print('Processed "waltz" features >>', data_all.shape)

        mutations['waltz'] = df['mutations'].tolist()
        get_mut_diff(mutations, 'waltz', 'ylabel')

    ######################
    ## PLM LLR FEATURES ##
    ######################
    if 'esm2_llr_entropy' in feature_types_to_include:
        protein_embeddings_subfolder = subfolders['conservation_and_distance']
        df = pd.read_csv(f'{data_folder}{protein_embeddings_subfolder}{data_subfolder}/{fname_prefix}esm2-33_LLRsum_entropy.csv')

        # merge data
        if i == 0:
            data_all = df.copy()
        else:
            data_all = data_all.merge(df, on='mutations', how='outer')
        i += 1
        features_all += feature_names['esm2_llr_entropy']
        print('Processed pLM LLR and entropy features >>', data_all.shape)

    ########################
    ## EMBEDDING FEATURES ##
    ########################
    if 'esm2_embeddings' in feature_types_to_include:
        protein_embeddings_subfolder = subfolders['feature_extraction']
        df = pd.read_csv(f'{data_folder}{protein_embeddings_subfolder}{data_subfolder}/{fname_prefix}esm2_embeddings.csv')
        df['enz_name'] = list(df.index)

        # merge data
        if i == 0:
            data_all = df.copy()
        else:
            data_all = data_all.merge(df, on='mutations', how='outer')
        i += 1
        features_all += feature_names['esm2_embeddings']
        print('Processed pLM embedding features >>', data_all.shape)

    ########################
    ## CHEMISTRY FEATURES ##
    ########################
    if 'chemistry' in feature_types_to_include:
        pass


    #####################
    ## REMOVE BAD DATA ##
    #####################
    # drop NA data
    data_all = data_all.dropna().reset_index(drop=True)
    # drop mutations with '*'
    idx_to_keep = []
    for i, mut in enumerate(data_all['mutations'].tolist()):
        if mut.find('*')==-1:
            idx_to_keep.append(i)
    data_all = data_all.iloc[idx_to_keep,:].reset_index(drop=True)

    #################
    ## SPLIT INDEX ##
    #################
    if 'split_index' in feature_types_to_include:
        split_type = feature_types_to_include['split_index'][0]
        if split_type=='random':
            n_splits = feature_types_to_include['split_index'][1]
            fold_idx_list = []
            for k in range(len(data_all)):
                fold_idx_list.append(k%n_splits)
            data_all['fold_random_5'] = fold_idx_list

    # save dataset as CSV
    if not os.path.isdir(output_csv_dir):
        os.makedirs(output_csv_dir)
    data_all.to_csv(output_csv_dir + output_csv_fname)
    print('final data (non-null):', data_all.shape)

    ##############
    ## SAVE CSV ##
    ##############
    output_csv_fpath = output_csv_dir + output_csv_fname
    output_csv_fpath_abs = os.path.abspath(output_csv_fpath)
    data_all.to_csv(output_csv_fpath)
    print('Saved aggregated data to:', output_csv_fpath, output_csv_fpath_abs)
    
    return data_all, features_all

    
if __name__ == "__main__":  # confirms that the code is under main function
    data_subfolder =  '' 
    feature_types_to_include = {
        'ylabel': ('GOh1052_mutagenesis.csv',
                   ['CategoryV3', 'Fold (Max Abs)'], ['mutations','fold_random_5']),
        'sequences': 'GOh1052.fasta',
        'binding_yasara': 'DDGbind_GOh1052.csv',
        'stability_foldx': 'DDGstability_GOh1052.csv',
        'tango': 'AggregationScore_GOh1052_tango.csv',
        'waltz': 'AggregationScore_GOh1052_waltz.csv'
    }
    output_csv_fname = 'GOh1052mut.csv'
    sep = '+'
    output_csv_dir = data_folder + subfolders['expdata'] + data_subfolder + '/'
    deduplicate_entries = False

    # compile dataset
    data_all, features_all = aggregate_features(feature_types_to_include, data_folder, data_subfolder, output_csv_fname, output_csv_dir, deduplicate_entries=deduplicate_entries, WT_or_MT='MT', sep=sep)