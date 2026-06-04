#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def load_data_allsuffixes(data_fname, data_folder, data_suffix_list=None, fmt='.csv'):

    if data_suffix_list is None:
        data_suffix_list = ['']
    # load csv data
    if fmt=='.csv':
        for suffix_idx, suffix in enumerate(data_suffix_list):
            df_suffix = pd.read_csv(data_folder + data_fname + suffix + fmt)
            if suffix_idx==0:
                df = df_suffix.copy()
            else:
                df = pd.concat([df, df_suffix], axis=0)
    # load npy data (sparse matrix of OHE encoding)
    elif fmt=='.npz':
        import scipy.sparse as sp
        for suffix_idx, suffix in enumerate(data_suffix_list):
            df_suffix = sp.load_npz(data_folder + data_fname + suffix + fmt).todense()
            if suffix_idx==0:
                df = df_suffix.copy()
            else:
                df = np.concatenate((df, df_suffix), axis=0)
    elif fmt=='.npy':
        for suffix_idx, suffix in enumerate(data_suffix_list):
            df_suffix = np.load(data_folder + data_fname + suffix + fmt)
            if suffix_idx==0:
                df = df_suffix.copy()
            else:
                df = np.concatenate((df, df_suffix), axis=0)
    print('Loaded data:', data_folder + data_fname + fmt, data_suffix_list, df.shape)
    return df

def round_to_classification_label(df_row, cols):
    for col in cols:
        val_initial = float(df_row[col].iloc[0])
        if val_initial > 0.5:
            val = 1
        elif val_initial > -0.5 and val_initial <= 0.5:
            val = 0
        elif val_initial <= -0.5:
            val = -1
        df_row[col] = val
    return df_row

def deduplicate_data_average_labels(df, dedupe_on, cols_to_avg, extra_cols_to_get, get_classification_label=False):
    idx_list = list(set(df[dedupe_on].tolist()))
    for k, idx in enumerate(idx_list):
        # filter for duplicate rows
        df_i = df.loc[df[dedupe_on]==idx, cols_to_avg+extra_cols_to_get].reset_index(drop=True)
        # if no duplicates, skip
        if len(df_i)<=1:
            df_i_avg = df_i[extra_cols_to_get + cols_to_avg]
        # if there are duplicates, get the average
        else:
            df_i_avg = df_i[cols_to_avg].mean(axis=0).to_frame().transpose()
            # for classification labels, round to appropriate label
            if get_classification_label:
                df_i_avg = round_to_classification_label(df_i_avg, cols_to_avg)
            # add additional metadata columns
            df_i_avg = pd.concat([df_i.loc[:0,extra_cols_to_get], df_i_avg], axis=1)
        # append to full dataset
        if k == 0:
            df_deduped = df_i_avg.copy()
        else:
            df_deduped = pd.concat([df_deduped, df_i_avg], axis=0, ignore_index=True)
    df_deduped = df_deduped.reset_index(drop=True)
    return df_deduped
