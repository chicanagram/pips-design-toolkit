# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 19:38:58 2022

@author: Jhoann
"""

import pandas as pd
import numpy as np
import os, sys, shutil
if os.path.basename(os.getcwd()) != 'feature_extraction': os.chdir('./feature_extraction/')
import subprocess
from utils.utils import fetch_sequences_from_fasta, write_sequence_to_fasta, get_mutations, get_mutated_sequence

def AggWaltz(mutations, seq_base, seq_name, inPath, outPath, output_csv_fname):
    waltz_path = "aggregation/Waltz.pl"

    # generate sequence of mutants based on seq_base
    if mutations is not None:
        mutations_list, seq_name_list, sequence_list, fasta_list = get_mutated_sequence(seq_base, mutations, seq_name,
                                                                                        write_to_fasta=inPath)
    else:
        if isinstance(seq_name, str):
            seq_name_list = [seq_name]
            sequence_list = [seq_base]
        else:
            seq_name_list = seq_name
            sequence_list = seq_base
            # generate fasta file for each sequence
            fasta_list = []
            for sequence, seq_name in zip(sequence_list, seq_name_list):
                fasta_file = write_sequence_to_fasta(sequence, seq_name, seq_name, inPath)
                fasta_list.append(fasta_file)

    # deduplicate sequences
    seq_name_list_deduped = []
    fasta_list_deduped = []
    for seq_name, fasta in zip(seq_name_list, fasta_list):
        if seq_name not in seq_name_list_deduped:
            seq_name_list_deduped.append(seq_name)
            fasta_list_deduped.append(fasta)
    print('# of Sequences Before deduplication:', len(seq_name_list), '; After deduplication:', len(seq_name_list_deduped))

    # iterate through sequences
    for i, (seq_name, fasta_file) in enumerate(zip(seq_name_list_deduped, fasta_list_deduped)):
        # set waltz args
        waltz_args = [
            "perl",
            waltz_path,
            fasta_file,
            '77',
        ]
        log_path = os.path.join(outPath + 'waltz/', seq_name + '.txt')
        try:
            logfile = open(log_path, 'w')
            # run waltz
            res = subprocess.run(waltz_args, stdout=logfile)
            logfile.close()
            print(f'Finished running Waltz on {seq_name} ({i+1}/{len(seq_name_list_deduped)})')
            # delete fasta file generated
            if os.path.exists(fasta_file):
                os.remove(fasta_file)
        except:
            print('Could not add results for ' + seq_name)

    # consolidate results
    df_agg = []
    df_agg_cols = ['Enzyme/Mutation', 'Aggregation']
    f_list = [os.path.basename(f) for f in os.listdir(outPath+'waltz/') if f.endswith('.txt')]
    f_list.sort()
    for f in f_list:
        seq_name = f.replace('.txt','')
        if seq_name in seq_name_list:
            log_path = outPath + 'waltz/' + f
            if mutations is not None:
                mut = seq_name.split('_')[-1]
            else:
                mut = None
            try:
                with open(log_path) as f:
                    df = pd.read_csv(log_path, sep='\t')
                    mylist = f.read().split("\t")
                    newlist = [float(el) for el in mylist[1:-1]]
                # get sum of each column
                row = {
                    'Enzyme/Mutation': mut if mutations is not None else seq_name,
                    'Aggregation': np.sum(np.array(newlist))
                }
                df_agg.append(row)
                print('Added results for ' + seq_name)
            except:
                print('Could not add results for ' + seq_name)

    df_agg = pd.DataFrame(df_agg)[df_agg_cols]

    # check if there's a reference sequence
    df_agg['is_ref'] = df_agg['Enzyme/Mutation'].str.contains('WT')
    df_ref = df_agg[df_agg['is_ref']].drop(columns='is_ref')
    df_agg = df_agg.drop(columns='is_ref')
    if len(df_ref) > 0:
        df_agg[[col + '_vs_ref' for col in df_agg_cols[1:]]] = df_agg.iloc[:, 1:].to_numpy() - df_ref.iloc[0, 1:].to_numpy().reshape(1,-1)

    final_path = os.path.join(outPath, f"{output_csv_fname}_waltz.csv")
    df_agg.to_csv(final_path, index=False)
    print(f'Saved aggregated results to {final_path}.')


def AggTango(mutations, seq_base, seq_name, inPath, outPath, output_csv_fname):

    # generate sequence of mutants based on seq_base
    if mutations is not None:
        mutations_list, seq_name_list, sequence_list, _ = get_mutated_sequence(seq_base, mutations, seq_name, write_to_fasta=None)
    else:
        if isinstance(seq_name, str):
            seq_name_list, sequence_list = [seq_name], [seq_base]
        else:
            seq_name_list, sequence_list = seq_name, seq_base

    encoding = sys.getdefaultencoding()
    cwd = os.getcwd()
    print('Current working directory:', cwd)
    tango_path = os.path.abspath('./aggregation/Tango.exe')
    tango_dir = os.path.dirname(tango_path)
    os.chdir(tango_dir)

    # iterate through sequences
    missed_seq_names = []
    seq_name_list_deduped = []
    sequence_list_deduped = []

    # deduplicate sequences
    for seq_name, sequence in zip(seq_name_list, sequence_list):
        if seq_name not in seq_name_list_deduped:
            seq_name_list_deduped.append(seq_name)
            sequence_list_deduped.append(sequence)
    print('# of Sequences Before deduplication:', len(seq_name_list), '; After deduplication:', len(seq_name_list_deduped))

    for i, (seq_name, sequence) in enumerate(zip(seq_name_list_deduped, sequence_list_deduped)):
        tango_args = [
            tango_path,
            seq_name,
            'nt=N',
            'ct=N',
            'ph=7',
            'te=298',
            'io=0.05',
            'tf=0',
            'stab=-10',
            'seq=' + sequence]

        print('Running Tango with args:')
        print(' '.join(tango_args))
        relative_outPath = os.path.relpath(cwd, tango_dir) + f'/{outPath}tango/'
        log_path = os.path.join(relative_outPath, seq_name + '.log')

        try:
            logfile = open(log_path, 'w')
            res = subprocess.run(tango_args, stdout=logfile, stderr=logfile, encoding=encoding)
            logfile.close()
            # move results .txt file from Tango dir to data dir
            shutil.copy(f'{seq_name}.txt', f'{relative_outPath}{seq_name}.txt')
            # remove log file and seq file
            os.remove(log_path)
            os.remove(f'{seq_name}.txt')
            print(f'Finished running Tango on {seq_name} ({i+1}/{len(seq_name_list_deduped)})')
        except:
            missed_seq_names.append(seq_name)
            print(f'Unable to finish processing {seq_name} with Tango.')
    os.chdir(cwd)

    # consolidate results
    df_agg = []
    df_agg_cols = ['Enzyme/Mutation', 'Aggregation', 'Conc-Stab_Aggregation', 'Beta', 'Turn', 'Helix']
    f_list = [os.path.basename(f) for f in os.listdir(outPath+'tango/') if f.endswith('.txt')]
    f_list.sort()
    for f in f_list:
        seq_name = f.replace('.txt','')
        if seq_name in seq_name_list:
            log_path = outPath + 'tango/' + f
            if mutations is not None:
                mut = seq_name.split('_')[-1]
            else:
                mut = None
            try:
                with open(log_path) as f:
                    df = pd.read_csv(log_path, sep='\t')
                # get sum of each column
                row = {
                    'Enzyme/Mutation': mut if mutations is not None else seq_name,
                    'Aggregation': sum(df['Aggregation']),
                    'Conc-Stab_Aggregation': sum(df['Conc-Stab_Aggregation']),
                    'Beta': sum(df['Beta']),
                    'Turn': sum(df['Turn']),
                    'Helix': sum(df['Helix'])
                }
                # print(row)
                df_agg.append(row)
                print('Added results for ' + seq_name)
            except:
                print('Could not add results for ' + seq_name)
    df_agg = pd.DataFrame(df_agg)[df_agg_cols]

    # check if there's a reference sequence
    df_agg['is_ref'] = df_agg['Enzyme/Mutation'].str.contains('WT')
    df_ref = df_agg[df_agg['is_ref']].drop(columns='is_ref')
    df_agg = df_agg.drop(columns='is_ref')
    if len(df_ref) > 0:
        df_agg[[col + '_vs_ref' for col in df_agg_cols[1:]]] = df_agg.iloc[:, 1:].to_numpy() - df_ref.iloc[0, 1:].to_numpy().reshape(1,-1)

    # save final dataframe
    final_path = os.path.join(outPath, f"{output_csv_fname}_tango.csv")
    df_agg.to_csv(final_path, index=False)
    print(f'Saved aggregated results to {final_path}.')
    print('Missed sequences:', missed_seq_names)


def get_aggregation_features():

    # setup
    waltz_or_tango = ['waltz', 'tango']
    inPath = '../data/feature_extraction/Input/'
    outPath = '../data/feature_extraction/'
    sequence_fname = 'GOh1052.fasta'

    # specify base sequence(s)
    sequence_fpath = '../data/sequences/' + sequence_fname
    seq_base, seq_name, seq_description = fetch_sequences_from_fasta(sequence_fpath)
    output_csv_fname = f'AggregationScore_{sequence_fname.replace(".fasta","")}'

    # get mutations
    seq_base = seq_base[0]
    seq_name = seq_name[0]
    mutatePos = [aa+str(i+1) for i,aa in enumerate(seq_base)]
    mutations = [None] + get_mutations(mutatePos)

    ## RUN WALTZ ##
    if 'waltz' in waltz_or_tango:
        AggWaltz(mutations, seq_base, seq_name, inPath, outPath, output_csv_fname)
    ## RUN TANGO ##
    if 'tango' in waltz_or_tango:
        AggTango(mutations, seq_base, seq_name, inPath, outPath, output_csv_fname)


if __name__ == '__main__':
    get_aggregation_features()