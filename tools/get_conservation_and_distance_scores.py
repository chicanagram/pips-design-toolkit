#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 07:58:44 2024

@author: charmainechia
"""
import os
import numpy as np
import pandas as pd
from utils.utils import data_folder, mapping_inv, fetch_sequences_from_fasta, write_sequence_to_fasta, get_ref_seq_idxs_aa_from_msa, compute_entropy

def insert_position_col_offset(df, offset, pos_col='RealPos'):
    pos = df[pos_col].to_numpy()
    pos_woffset = pos + offset
    df.insert(0,'Position', pos_woffset)
    return df

def ShanEntropy(msa_fname, output_fname, ref_seq_idxs, ref_seq, position_offset=None):
    from conservation_and_distance.alfa2cons import alfa2cons
    csv = alfa2cons(data_folder+'msa/'+msa_fname+'.fasta', data_folder+'conservation_and_distance/'+output_fname+'.csv', save_csv=False)
    csv_filt = csv[csv['Position'].isin(ref_seq_idxs)].copy()
    ref_seq_parsed = list(ref_seq) if ref_seq is not None else None
    if ref_seq_parsed is not None:
        csv_filt['AA'] = ref_seq_parsed
    csv_filt['RealPos'] = list(np.arange(len(ref_seq_parsed)) + 1)
    csv_filt = csv_filt.rename(columns={'Position': 'PositionMSA'})
    csv_filt = csv_filt[['AA', 'RealPos', 'PositionMSA', 'shanID', 'shanMS']]
    if position_offset is not None:
        csv_filt = insert_position_col_offset(csv_filt, position_offset)
    # write dataframe to file
    output_fpath = data_folder+'conservation_and_distance/'+output_fname+'_shanms.csv'
    csv_filt.to_csv(output_fpath)
    print(f'Saved Shannon Entropy scores to: {output_fpath}')

def SIFT(msa_fname, output_fname, ref_seq_name, position_offset=None):
    from conservation_and_distance import access_sift_webserver
    msa_path = data_folder+'msa/'+msa_fname+'.fasta'
    msa_seqs, msa_names, _ = fetch_sequences_from_fasta(msa_path)

    msa_idx = msa_names.index(ref_seq_name)
    msa_names_rearranged = [msa_names[msa_idx]] + msa_names[:msa_idx] + msa_names[msa_idx+1:]
    msa_seqs_rearranged = [msa_seqs[msa_idx]] + msa_seqs[:msa_idx] + msa_seqs[msa_idx+1:]
    msa_path_rearranged = data_folder+'msa/'+msa_fname + f'_{ref_seq_name}.fasta'
    write_sequence_to_fasta(msa_seqs_rearranged, msa_names_rearranged, msa_fname+f'_{ref_seq_name}', data_folder+'msa/')
    csv = access_sift_webserver.main(
      {'-a': os.path.abspath(msa_path_rearranged),
       '--fname': data_folder+'conservation_and_distance/'+msa_fname+f'_sift.csv'
       }
    )
    # get probabilities pre-normalization
    probs_matrix_norm = np.transpose(csv.iloc[:, -20:].to_numpy()).astype(float)
    csv.iloc[:,-20:] = np.transpose(probs_matrix_norm)
    prob_col = csv['prob'].to_numpy()
    probs_matrix = probs_matrix_norm * prob_col
    # get average sift score for each residue
    probs_mean = np.mean(probs_matrix_norm, axis=0)
    csv.insert(2, 'sift', list(probs_mean))
    # calculate entropy for each position
    ent = compute_entropy(probs_matrix)
    csv.insert(2, 'entropy', list(ent))
    # write dataframe to file
    csv = csv.rename(columns={'pos': 'RealPos', 'wt': 'AA'})
    if position_offset is not None:
        csv = insert_position_col_offset(csv, position_offset)
    output_fpath = data_folder+'conservation_and_distance/'+output_fname+'_sift.csv'
    csv.to_csv(output_fpath)
    os.remove(msa_path_rearranged)
    os.remove(data_folder+'conservation_and_distance/'+msa_fname+f'_sift.csv')
    print(f'Saved SIFT scores to: {output_fpath}')

def DistResSub(sce_fname, output_fname, position_offset=None):
    import yasara
    print('Started YASARA')
    # start yasara
    yasara.info.mode = 'txt'
    yasara.info.licenseshown = 0
    yasara.Console('Off')
    # load structure
    yasara.CleanAll()
    yasara.LoadSce(data_folder+'sce/'+sce_fname)
    print('Loaded docked structure')
    yasara.Console('Off')
    # get number of residues
    num_res = yasara.CountRes('Protein')
    print('# of residues:', num_res)
    # iterate through each residue and get distance to substrate
    res = []
    for res_idx in range(1, num_res+1):
        wildtype = yasara.NameRes(res_idx)
        aa = mapping_inv[wildtype[0]]
        selection1 = 'Res '+str(res_idx)
        selection2 = 'Obj 2'
        dist = yasara.GroupDistance(selection1, selection2)
        res.append([aa, res_idx, dist])
    res = pd.DataFrame(res, columns=['AA', 'RealPos', 'distance'])
    if position_offset is not None:
        res = insert_position_col_offset(res, position_offset)
    output_fpath = data_folder+'conservation_and_distance/'+output_fname+'_distance.csv'
    res.to_csv(output_fpath)
    print(f'Saved distance scores to: {output_fpath}')

def get_conservation_and_distance_scores():
    msa_fname = 'GOh1052_msa'
    ref_seq_name = 'FGGALOX'
    ref_seq = None
    output_fname = 'GOh1052'
    sce_fname = 'S152_1GOG_GOh1001b_postOpt'  # "S152_preOpt_1GOG_GOh1001b_preOpt.sce"
    position_offset = 23 # 0

    if ref_seq is None:
        _, ref_seq, ref_seq_idxs = get_ref_seq_idxs_aa_from_msa(data_folder+'msa/'+msa_fname+'.fasta', ref_seq_name)

    # run ShanEntropy
    ShanEntropy(msa_fname, output_fname, ref_seq_idxs, ref_seq, position_offset)
    # run SIFT
    SIFT(msa_fname, output_fname, ref_seq_name, position_offset)
    # run YASARA distance calculation
    DistResSub(sce_fname, output_fname, position_offset)

if __name__ == "__main__":
    get_conservation_and_distance_scores()