#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 07:58:44 2024

@author: charmainechia
"""
try:
    import pandas as pd
    pandas_imported = True
except ImportError as e:
    pandas_imported = False
import os
import numpy as np
import platform
from utils.variables import aaList, aaList_with_X
opsys = platform.system()

def sort_list(lst):
    lst.sort()
    return lst

def get_mutstr(mutation):
    if isinstance(mutation, list):
        mutstr = '+'.join(mutation)
    else:
        mutstr = mutation
        mutation = [mutation]
    return mutstr, mutation

def get_mutated_sequence(seq_base, mutations, seq_name=None, write_to_fasta=None):
    mutations_list = []
    seq_name_list = []
    sequence_list = []
    fasta_list = []

    # get mutated sequences
    for mut in mutations:
        if mut is not None:
            wildtype_aa, position, mutant_aa = split_wildtype(mut)

            if seq_base[position-1] == wildtype_aa:
                if seq_name is None:
                    seq_name_wmut = mut
                else:
                    seq_name_wmut = seq_name + '_' + mut
            else:
                seq_name_wmut = seq_name + '_' + mut
            list_seq_base = list(seq_base)
            list_seq_base[position-1] = mutant_aa
            seq_mutate = "".join(list_seq_base)

        else:
            mut = 'WT'
            seq_mutate = seq_base
            seq_name_wmut = seq_name + '_' + mut

        mutations_list.append(mut)
        sequence_list.append(seq_mutate)
        seq_name_list.append(seq_name_wmut)

        # write to fasta
        if write_to_fasta is not None:
            fasta_file = write_sequence_to_fasta(seq_mutate, seq_name_wmut, seq_name_wmut, write_to_fasta)
            fasta_list.append(fasta_file)

    return mutations_list, seq_name_list, sequence_list, fasta_list

def mkDir(res, output_dir, remove_existing_dir=True):
    import shutil
    # making new directory
    new_dir = (output_dir + res)
    if os.path.exists(new_dir):
        # remove if directory exists, and make new directory
        if remove_existing_dir:
            shutil.rmtree(new_dir)
            os.makedirs(new_dir)
    else:
        os.makedirs(new_dir)
    return new_dir

def findProcess(process_name):
    if opsys=='Windows':
        return [int(item.split()[1]) for item in os.popen('tasklist').read().splitlines()[4:] if process_name in item.split()]
    elif opsys=='Linux' or opsys=='Darwin':
        return [int(pid) for pid in os.popen('pidof '+process_name).read().strip(' \n').split(' ') if pid!='']

def exit_program(pid):
    import signal
    print("Sending SIGINT to self...")
    os.kill(pid, signal.SIGINT)
    print('Exited program', pid)


def save_dict_as_csv(datadict, cols, log_fpath, csv_suffix ='', multiprocessing_proc_num=None):
    # save results as CSV
    csv_txt = ''
    # get csv_suffix if running multiprocessing
    if multiprocessing_proc_num is not None:
        csv_suffix += '_' + str(multiprocessing_proc_num)

    # check if file exists yet
    log_fpath_full = log_fpath + csv_suffix + '.csv'
    if not os.path.exists(log_fpath_full):
        # if not, start a new file with headers
        write_mode = 'w'
        csv_txt += ','.join(cols) + '\n'
    else:
        write_mode = 'a'

    # convert dict of lists to list of dicts
    if isinstance(datadict[cols[0]], list):
        num_rows = len(datadict[cols[0]])
        datadict_byrow = []
        for row_idx in range(num_rows):
            row = []
            for col in cols:
                row.append(datadict[col][row_idx])
            datadict_byrow.append(row)
    else:
        row = []
        for col in cols:
            row.append(datadict[col])
        datadict_byrow = [row]

    # add data to csv file
    for row in datadict_byrow:
        csv_txt += ','.join([str(el) for el in row])
        csv_txt += '\n'
    # save the changes
    with open(log_fpath_full, write_mode) as f:
        f.write(csv_txt)
    return csv_txt, log_fpath_full, write_mode

def combine_csv_files(log_fpath_list, output_dir, output_fname, remove_combined_files=True):
    # combine files spawned
    txt_all_list = []
    missing_data = []
    # fetch logged result
    for i, log_fpath in enumerate(log_fpath_list):
        if os.path.exists(log_fpath):
            with open(log_fpath, 'r') as f:
                if i==0:
                    txt_all_list += f.readlines()
                else:
                    txt_all_list += f.readlines()[1:]
        else:
            missing_data.append(log_fpath)
            print(i, log_fpath)

    # update combined results
    if os.path.exists(output_dir + output_fname + '.csv'):
        write_mode = 'a'
        txt_all_list = txt_all_list[1:]
    else:
        write_mode = 'w'
    # get text string to write
    txt_all = '\n'.join(txt_all_list)
    txt_all = txt_all.replace('\n\n', '\n').replace(',\n', '\n')
    # update or save file
    with open(output_dir + output_fname + '.csv', write_mode) as f:
        f.write(txt_all)

    # record missing files
    if len(missing_data)>0:
        if os.path.exists(output_dir + 'missing_data.txt'):
            write_mode = 'a'
        else:
            write_mode = 'w'
        with open(output_dir + 'missing_data.txt', write_mode) as f:
            missing_txt = '\n'.join(missing_data) + '\n'
            f.write(missing_txt)

    # remove combined files
    if remove_combined_files:
        for log_fpath in [f for f in log_fpath_list if f not in missing_data]:
            os.remove(log_fpath)
    return missing_data

def split_mutation(mutation, aa_letter_representation=False):
    # Convert point mutation to wildtype residue, muted residue and mutation position
    mutation = list(mutation)
    WT_res = mutation[0]
    MUT_res = mutation[-1]
    if not aa_letter_representation:
        WT_res = mapping[WT_res]
        MUT_res = mapping[MUT_res]
    MUT_pos = mutation[1:len(mutation)-1]
    MUT_pos = int(''.join(MUT_pos))
    return WT_res, MUT_pos, MUT_res

def split_wildtype(mutation):
    # Convert point mutation to wildtype residue, muted residue and mutation position
    WT_res = mutation[0]
    MUT_pos = int(mutation[1:len(mutation)-1])
    MT_res = mutation[-1]
    return WT_res, MUT_pos, MT_res

def get_mutations(wildtype_list):
    # get amino acid list to perform mutations to
    aaList = ['A', 'H', 'Y', 'R', 'T', 'K', 'M', 'D', 'N', 'C', 'Q', 'E', 'G', 'I', 'L', 'F', 'P', 'S', 'W', 'V']
    # get all mutations to run
    mutations = []
    for wt in wildtype_list:
        wtAA = wt[0]
        for aa in aaList:
            if aa != wtAA:
                mt = wt + aa
                mutations.append(mt)
    print('mutants:', mutations)
    return mutations
    
def get_mutation_list_from_inputfile(input_fname, input_dir):
    # get mutations
    # input is a list of positions to mutate
    res_mut_dict = {}
    with open(input_dir + input_fname) as f:
        mutations = [mut.replace('\n','') for mut in f.readlines()]
        # only WT positions specified, not mutations
        if mutations[0][-1].isdigit():
            wildtype_list = mutations.copy()
            # mutate to all possible residues, if not specified
            for wt in wildtype_list:
                res_mut_dict[wt] = [wt + aa for aa in aaList if wt[0] != aa]
            mutations = [item for sublist in list(res_mut_dict.values()) for item in sublist]
        # both WT and MT specified
        elif mutations[0][-1].isalpha():
            wildtype_list = []
            for mut in mutations:
                wt = mut[:-1]
                if wt not in wildtype_list:
                    wildtype_list.append(wt)
                    res_mut_dict[wt] = []
                res_mut_dict[wt].append(mut)
    return mutations,  res_mut_dict

def list_all_mutations(seq, ignore_mutations_to_WT=True):
    mut_all = []
    for i in range(len(seq)):
        pos = str(i+1)
        wt_aa = seq[i]
        if ignore_mutations_to_WT:
            mut_pos = [wt_aa+pos+aa for aa in aaList if aa!=wt_aa]
        else:
            mut_pos = [wt_aa + pos + aa for aa in aaList]
        mut_all += mut_pos
    return mut_all
    
def fetch_sequences_from_fasta(sequence_fpath):
    from Bio import SeqIO
    sequence_names = []
    sequence_list = []
    sequence_descriptions = []
    for j, record in enumerate(SeqIO.parse(sequence_fpath, "fasta")):
        sequence_names.append(record.id)
        sequence_list.append(str(record.seq))
        sequence_descriptions.append(record.description)
    return sequence_list, sequence_names, sequence_descriptions
    
def write_sequence_to_fasta(sequences, seq_names, filename, fasta_dir):
    fasta_file = fasta_dir + filename + '.fasta'
    if isinstance(sequences, str) and isinstance(seq_names, str):
        sequences = [sequences]
        seq_names = [seq_names]
    with open(fasta_file, 'w') as f:
        for i, (sequence, seq_name) in enumerate(zip(sequences, seq_names)):
            f.write('> ' + seq_name + '\n')
            if i==len(sequences)-1:
                f.write(sequence)
            else:
                f.write(sequence+'\n')
    print('Saved fasta file to ' + fasta_file)
    return fasta_file

def get_ref_seq_idxs_aa_from_msa(msa_path, ref_seq_name_list, zero_indexed=False):
    # get ref_seq_name_list found in MSA
    ref_seq_name_list_inmsa = []
    ref_seq_idxs_list_inmsa = []
    ref_seq_list_inmsa = []
    # check that all ref seqs are in the MSA
    msa_seqs, msa_names, _ = fetch_sequences_from_fasta(msa_path)
    for ref_seq_name in ref_seq_name_list:
        ref_seq_inmsa = ref_seq_name in msa_names
        print(ref_seq_name + ' is in MSA: ' + str(ref_seq_inmsa))
        if ref_seq_name in msa_names:
            ref_seq_name_list_inmsa.append(ref_seq_name)
            msa_idx = msa_names.index(ref_seq_name)
            msa_seq = msa_seqs[msa_idx]
            seq_filt = ''
            idx_filt = []
            for i, letter in enumerate(list(msa_seq)):
                if letter != '-':
                    seq_filt += letter
                    if zero_indexed:
                        idx_filt.append(i)
                    else:
                        idx_filt.append(i+1)
            ref_seq_idxs_list_inmsa.append(idx_filt)
            ref_seq_list_inmsa.append(seq_filt)
    return ref_seq_name_list_inmsa, ref_seq_list_inmsa, ref_seq_idxs_list_inmsa

def compute_entropy(probs_matrix):
    """
    Computes the entropy for each position in the given probability matrix.
    """
    from scipy.stats import entropy

    # Initialize an empty list to store the entropy values
    entropy_values = []

    # Iterate over the columns of probs_matrix
    for i in range(probs_matrix.shape[1]):
        # Compute the entropy for the probabilities at the current position
        H = entropy(probs_matrix[:, i], base=2)
        entropy_values.append(H)

    # Convert entropy_values to a numpy array for convenience
    entropy_values = np.array(entropy_values)
    return entropy_values

def get_log_likelihood_ratios(probs_matrix, seq, plot_heatmap=True, seq_name=None, savefig=None, print_positive_LLR_mutations=False):

    if not isinstance(probs_matrix, np.ndarray):
        pos_list = probs_matrix.columns.tolist()
        probs_matrix = probs_matrix.to_numpy()
    else:
        pos_list = list(range(1,len(seq)+1))
    wt_aa_pos = [seq[pos-1] for pos in pos_list]

    # get amino acid list to plot
    if 'X' in wt_aa_pos:
        aa_list = aaList_with_X
    else:
        aa_list = aaList
    probs_matrix = probs_matrix[:len(aa_list), :]

    # get N_res_per_heatmap_row
    if len(pos_list) < 150: N_res_per_heatmap_row = len(pos_list)
    else: N_res_per_heatmap_row = 100

    # get LLRs
    diff_logprob_heatmap = np.zeros_like(probs_matrix)
    diff_logprob_heatmap[:] = np.nan
    pos_w_positiveLLR = {}
    for i, pos in enumerate(pos_list):
        probs_byres = probs_matrix[:,i]
        wt_aa = seq[pos-1]
        if wt_aa!='-':
            wt_idx = aa_list.index(wt_aa)
            prob_wt = probs_byres[wt_idx]
            diff_logprob = np.round(np.log(probs_byres) - np.log(prob_wt),3)
            diff_logprob_heatmap[:,i] = diff_logprob
            # get positions with positive LLR vs. WT
            posLLR_aa_idx = np.where(diff_logprob>0)[0]
            if len(posLLR_aa_idx) > 0:
                mut_llr_list = [(wt_aa+str(pos)+aa_list[k], diff_logprob[k]) for k in posLLR_aa_idx]
                pos_w_positiveLLR.update({pos: mut_llr_list})
                if print_positive_LLR_mutations:
                    print(pos, end=':  ')
                    for (mut, llr) in mut_llr_list:
                        print(f'{mut} ({llr})', end='; ')
                    print()
        else:
            pass
    diff_logprob_heatmap = pd.DataFrame(diff_logprob_heatmap, columns=pos_list)
    if plot_heatmap:
        plot_variant_heatmap(diff_logprob_heatmap, seq, N_res_per_heatmap_row, aa_list, seq_name=seq_name, savefig=savefig, figtitle='Predicted Effects of Mutations on Protein Sequence (LLR)')

    return diff_logprob_heatmap, pos_w_positiveLLR

def flatten_2D_arr(arr2D, seq, MT_aa=aaList):
    """
    arr2D is a 2-dimensional matrix
        axis 0 (vertical): 20 amino acids along axis 0
        axis 1 (horizontal): sequence positions
    """
    if not isinstance(arr2D, np.ndarray):
        WT_res = [seq[pos-1]+str(pos) for pos in arr2D.columns.tolist()]
        arr2D = arr2D.to_numpy()
    else:
        WT_res = [seq[pos - 1] + str(pos) for pos in list(range(1, arr2D.shape[1] + 1))]

    arr1D = arr2D.flatten('F')
    mutations = [wt + mt for wt in WT_res for mt in MT_aa]
    return arr1D, mutations


def compute_pppl(probs, sequence, pos_list=None):
    """
    Compute the pseudo-perplexity for a given sequence given the full probability matrix for all possible substitutions
    Input probs matrix has amino acid substitutions along axis 0, and sequence positions along axis 1
    """
    if pos_list is None:
        pos_list = list(range(1,len(sequence)+1))
    log_probs_wt = []
    for i, pos in enumerate(pos_list):
        wt_aa = sequence[pos-1]
        log_probs_wt.append(np.log(probs[aaList_with_X.index(wt_aa),i]))
    log_probs_wt = [logprob for logprob in log_probs_wt if ~np.isnan(logprob)]
    pppl = -sum(log_probs_wt)/len(log_probs_wt)
    print('\nPPPL:', pppl)
    return pppl


def compose_prob_entropy_PPPL_outputs(probs_matrix_list, seqs, seq_names, out_dir=None, fname_suffix='_MutProbs'):
    """
    Give the probability matrix for all substitutions for a set of positions, calculate entropy and pppl
    Input probability matrix has: axis 0 -> amino acids; axis 1: positions in sequence
    """
    if not isinstance(probs_matrix_list, list):
        probs_matrix_list = [probs_matrix_list]
        seqs = [seqs]
        seq_names = [seq_names]

    pppl_list = []
    df_list = []
    for i, (probs, seq, seq_name) in enumerate(zip(probs_matrix_list, seqs, seq_names)):
        print(f'Composing CSV for {seq_name}...')
        # convert pandas dataframe to numpy array if needed
        if not isinstance(probs, np.ndarray):
            pos_list = probs.columns.tolist()
            probs = probs.to_numpy()
        else:
            pos_list = list(range(1,len(seq)+1))
        wt_aa_pos = [seq[pos-1] for pos in pos_list]
        print('# of wt_aa_pos:', len(wt_aa_pos))

        # get alphabet
        if 'X' in wt_aa_pos:
            aa_list = aaList_with_X
        else:
            aa_list = aaList
        probs = probs[:len(aa_list), :]
        probs /= np.sum(probs, axis=0)

        # calculate PPPL
        pppl = compute_pppl(probs, seq, pos_list)
        pppl_list.append(pppl)

        # calculate entropy and plot
        entropy_values = compute_entropy(probs)
        print('Obtained entropies for each residue.')

        # save probabilities, entropy, PPPL to CSV file
        df_cols = ['RealPos', 'AA', 'entropy', 'pppl'] + aa_list
        df_vals = np.zeros((len(pos_list), len(df_cols)))
        df_vals[:, 4:] = np.transpose(probs)
        df = pd.DataFrame(df_vals, columns=df_cols)
        df['RealPos'] = pos_list
        df['RealPos'] = df['RealPos'].astype(int)
        df['AA'] = wt_aa_pos
        df['entropy'] = entropy_values
        df['pppl'] = pppl
        # remove rows where all AA prob values are NaN
        df = df.dropna(subset=aa_list, how="all")
        df_list.append(df)

        # save CSV
        if out_dir is not None:
            csv_fpath = f'{out_dir}{seq_name}{fname_suffix}.csv'
            df[df_cols].reset_index(drop=True).to_csv(csv_fpath)
            print(f'Saved MutProbs CSV for {seq_names}: {csv_fpath}')

    if len(df_list)==1:
        df_list = df_list[0]
    return df_list
