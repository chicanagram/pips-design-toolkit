#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 07:58:44 2024

@author: charmainechia
"""
import os
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option('display.width', 1000)
from utils.variables import data_folder, subfolders
from ml_prediction.model_features import feature_names, get_feature_combinations, get_featureset_dir
from ml_prediction.metrics import get_regressor_scoring
from ml_prediction.splits import get_protein_splits, get_mutres_modulo_splits, get_custom_split
from ml_prediction.datacleaning_utils import load_data_allsuffixes, deduplicate_data_average_labels
from ml_prediction.sklearn_models import fit_sklearn_regressor_kfold


def evaluate_regression(
    data_folder,
    data_subfolder,
    labels_dir,
    data_suffix_list,
    fname_prefix,
    labels_actual_fname,
    ylabel_list,
    splits_to_evaluate,
    extra_cols_to_get,
    output_fname,
    custom_test_fname=None,
    filter_out_data={'mutations': ['WT','NC','X']},
    filter_in_data={},
    merge_on='mutations',
    include_nonplm_featuresets=True,
    scale_data=False,
    deduplicate_data=True,
    n_splits=5,
    metrics_list=['r2', 'SpearmanR', 'PearsonR'],
    models_to_eval_list=['xgb'],
    plm_name_list=['esm2-33'],
    plot_scatter=False,
    show_only_test_results=True,
    save_y_predictions=False,

):
    # metrics list
    res_cols=['ylabel', 'Featureset', 'p', 'n', 'Model', 'split_type', 'train_or_test']+metrics_list
    res_cols_summary = ['ylabel', 'Featureset', 'p', 'n', 'Model', 'split_type', 'train_or_test'] + metrics_list + [c for c_pair in [[metric+'_avg', metric+'_std'] for metric in metrics_list] for c in c_pair]
    # get PLM combination features
    feature_combinations = get_feature_combinations(plm_name_list, feature_combinations={}, include_nonplm_featuresets=include_nonplm_featuresets)

    for split_type in splits_to_evaluate:
        print('SPLIT TYPE:', split_type)
        # initialize results
        res = []
        count = 0

        # load label data and filter rows
        print('Fetching labels...')
        df0 = load_data_allsuffixes(labels_actual_fname, labels_dir, data_suffix_list, fmt='.csv')
        for col, val in filter_out_data.items():
            df0 = df0[~df0[col].isin(val)]
        for col, val in filter_in_data.items():
            df0 = df0[df0[col].isin(val)]
        print('labels dataframe shape:', df0.shape)

        ###########################
        # load features & predict #
        ###########################
        # iterate through all featuresets to test
        for i, (featureset_label, component_featureset_list) in enumerate(feature_combinations.items()):
            print('\n********************************')
            print(i, featureset_label)
            print('********************************')
            df = df0.copy()
            print('Component featureset list:', component_featureset_list)
            feature_list_all = []

            # get each component featureset
            for component_featureset in component_featureset_list:
                fname_base = component_featureset
                if fname_base.find('LLRsum')>-1 and fname_base.find('entropy')==-1:
                    fname_base = fname_base.replace('_LLRsum', '_LLRsum_entropy')
                subfolder = get_featureset_dir(fname_base)
                features_fname = subfolders[subfolder] + data_subfolder + '/' + fname_prefix + fname_base
                print(features_fname)
                feature_list = feature_names[component_featureset]

                # if features and labels are in separate files, merge them
                print(f'Parsing features for {component_featureset}...')
                if feature_list is not None:
                    df_features = load_data_allsuffixes(features_fname, data_folder, data_suffix_list, fmt='.csv')
                else:
                    arr = load_data_allsuffixes(features_fname, data_folder, data_suffix_list, fmt='.npz')
                    feature_list = range(arr.shape[1])
                    df_features = pd.DataFrame(arr, columns=feature_list)
                    df_features.insert(0,'name', df['name'].tolist())
                # get 'name' column in df_features
                if 'name' not in df_features:
                    if 'protein_name' in df_features and 'mutations' in df_features:
                        df_features['name'] = [prot+'_'+mut for prot,mut in zip(df_features['protein_name'].tolist(),df_features['mutations'].tolist())]
                    elif 'Unnamed: 0' in df_features:
                        df_features = df_features.rename(columns={'Unnamed: 0':'name'})

                # modify colnames if there are duplicates
                if 'embeddings' in component_featureset and len([fs for fs in component_featureset_list if 'embeddings' in fs])>1:
                    embedding_prefix = component_featureset[:component_featureset.find('_')+1]
                    feature_list = [f.replace(embedding_prefix, component_featureset[:component_featureset.find('embeddings')]) for f in feature_list]
                    df_features.columns = [f.replace(embedding_prefix, component_featureset[:component_featureset.find('embeddings')]) for f in df_features.columns.tolist()]
                feature_list_all += feature_list

                # merge features with labels
                df_features = df_features.drop_duplicates()
                print('df_features.shape:', df_features.shape)
                if features_fname != labels_actual_fname:
                    if len(df)==len(df_features):
                        # if not AggStabBind feature set
                        # if component_featureset.find('AggStabBind')==-1:
                        if component_featureset.find('ohe') > -1:
                            df = pd.concat([df.reset_index(drop=True), df_features[feature_list].reset_index(drop=True)], axis=1)
                        else:
                            df = df.merge(df_features[[merge_on]+feature_list], on=merge_on, how='left')
                        # df = df.merge(df_features[[merge_on] + feature_list], on=merge_on, how='left')
                        print('Concatenated features with dataframe:', df.shape)
                    else:
                        df = df.merge(df_features[[merge_on]+feature_list], on=merge_on, how='left')
                        print('Merged features with dataframe:', df.shape)

            # deduplicate data
            if deduplicate_data:
                df = deduplicate_data_average_labels(df, merge_on, feature_list_all+ylabel_list, extra_cols_to_get)
                print('After deduplicating:', df.shape)

            # get split indices and shuffle data, if needed
            df = df[extra_cols_to_get + feature_list_all + ylabel_list]
            df = df.dropna(axis=0, ignore_index=True)
            print('FINAL XY DATAFRAME SIZE:', df.shape)

            shuffle_idx = None
            if split_type=='random':
                split_idxs_list = None
                idxs = np.arange(len(df))
                shuffle_idx = []
                for k in range(n_splits):
                    shuffle_idx += list(idxs[k::n_splits])
                assert len(shuffle_idx)==len(idxs)
                # shuffle_idx = np.arange(len(df))
                # np.random.seed(seed=0)
                # np.random.shuffle(idxs)
            elif split_type=='mutres-modulo':
                split_idxs_list, _ = get_mutres_modulo_splits(df, n_splits=n_splits)
            elif split_type=='protein':
                split_idxs_list, _ = get_protein_splits(df, n_splits=n_splits)
            elif split_type=='custom':
                split_idxs_list = get_custom_split(df, data_folder+custom_test_fname)

            for ylabel in ylabel_list:
                # get XY data columns only
                XY = df[extra_cols_to_get + feature_list_all + [ylabel]]

                # shuffle data
                if shuffle_idx is not None:
                    XY = XY.iloc[shuffle_idx, :]
                    print('Shuffled the data.')
                num_samples = XY.shape[0]
                num_features = len(feature_list_all)
                print('num_samples:', num_samples, '; num_features:', num_features)

                for k, model_type in enumerate(models_to_eval_list):
                    print('MODEL TYPE:', model_type)
                    model_dict = {'model_type': model_type}

                    ##########################
                    # perform classification #
                    ##########################

                    # perform k-fold train test, and get full set of predictions from out-of-training split
                    if 'ohe' in featureset_label:
                        scale_data = False

                    # get results for the n_splits
                    print('Fitting regressor on k-folds', split_type)
                    if split_type=='random': print('# of splits:', n_splits)
                    if model_type in ['lasso', 'xgb', 'plsr', 'randomforest']:
                        X = XY[feature_list_all].to_numpy()
                        y = XY[ylabel].to_numpy()
                        metrics_kfold, ypred = fit_sklearn_regressor_kfold(X, y, model_dict, n_splits, scale_data, split_idxs_list, plot_scatter)
                    # elif model_type in ['autogluon']:
                    #     model_dict.update({ 'save_model': data_folder+'/ml_prediction/trained_models/', 'load_model': None, 'model_settings': ag_model_settings, 'num_bag_folds': ag_num_bag_folds, 'num_stack_levels': ag_num_stack_levels})
                    #     metrics_kfold, ypred = fit_autogluon_regressor_kfold(XY, ylabel, model_dict, n_splits, split_idxs_list, metrics_list, multiclass_average_method)

                    # get average of k-folds
                    metrics_kfold = pd.DataFrame(metrics_kfold)
                    metrics_kfold_summary = pd.DataFrame([
                        metrics_kfold.loc[metrics_kfold['train_or_test']=='train', metrics_list].mean().to_dict(),
                        metrics_kfold.loc[metrics_kfold['train_or_test']=='test', metrics_list].mean().to_dict()
                    ])
                    metrics_kfold_summary_stdev = pd.DataFrame([
                        metrics_kfold.loc[metrics_kfold['train_or_test'] == 'train', metrics_list].std().to_dict(),
                        metrics_kfold.loc[metrics_kfold['train_or_test'] == 'test', metrics_list].std().to_dict()
                    ])
                    metrics_kfold_summary_stdev = metrics_kfold_summary_stdev.rename(
                        columns={c: c + '_std' for c in metrics_kfold_summary_stdev.columns})
                    metrics_kfold_summary = pd.concat([metrics_kfold_summary, metrics_kfold_summary_stdev], axis=1)
                    metrics_kfold_summary = metrics_kfold_summary.rename(columns={c: c + '_avg' for c in metrics_list})

                    # get y vs ypred overall
                    predictions = XY[['protein_name', 'mutations', ylabel]].copy()
                    predictions[ylabel + '_pred'] = ypred
                    metrics_all = get_regressor_scoring(ypred, y, model_name=None)
                    metrics_all = {k: float(round(v, 4)) if isinstance(v, float) else v for k, v in metrics_all.items()}
                    print('metrics_all:', metrics_all)
                    if save_y_predictions:
                        predictions.to_csv(
                            data_folder + 'ml_prediction/Output/' + data_subfolder + '/' + output_fname + '_' + split_type + 'ypred_all.csv')
                    # append overall metrics calculations
                    for metric in metrics_list:
                        metrics_kfold_summary[metric] = metrics_all[metric]

                    # append metadata cols
                    metrics_kfold['ylabel'] = ylabel
                    metrics_kfold_summary['ylabel'] = ylabel
                    metrics_kfold['Featureset'] = featureset_label
                    metrics_kfold_summary['Featureset'] = featureset_label
                    metrics_kfold['p'] = num_features
                    metrics_kfold_summary['p'] = num_features
                    metrics_kfold['n'] = num_samples
                    metrics_kfold_summary['n'] = num_samples
                    metrics_kfold['Model'] = model_dict['model_type']
                    metrics_kfold_summary['Model'] = model_dict['model_type']
                    metrics_kfold['split_type'] = split_type
                    metrics_kfold_summary['split_type'] = split_type
                    metrics_kfold_summary.insert(0, 'train_or_test', ['train','test'])
                    metrics_kfold = metrics_kfold[res_cols]
                    metrics_kfold_summary = metrics_kfold_summary[res_cols_summary]
                    print('K-Fold summary')
                    print(metrics_kfold_summary)

                    # update K-fold results
                    if count==0:
                        metrics_kfold_all = metrics_kfold.round(3).copy()
                        metrics_kfold_summary_all = metrics_kfold_summary.round(3).copy()
                    else:
                        metrics_kfold_all = pd.concat([metrics_kfold_all, metrics_kfold.round(3)], axis=0)
                        metrics_kfold_summary_all = pd.concat([metrics_kfold_summary_all, metrics_kfold_summary.round(3)], axis=0)
                    count+=1

        # save kfold results
        if show_only_test_results:
            metrics_kfold_summary_all = metrics_kfold_summary_all[metrics_kfold_summary_all['train_or_test']=='test']
            metrics_kfold_all = metrics_kfold_all[metrics_kfold_all['train_or_test'] == 'test']

        # save kfold results
        metrics_kfold_summary_all = metrics_kfold_summary_all.sort_values(by=['ylabel', 'p'])
        out_summary_fpath = data_folder+'ml_prediction/Output/' + data_subfolder + '/' + output_fname + '_' + split_type + '_summary.csv'
        metrics_kfold_summary_all.to_csv(out_summary_fpath)
        out_fpath = data_folder + 'ml_prediction/Output/' + data_subfolder + '/' + output_fname + '_' + split_type + '.csv'
        metrics_kfold_all.to_csv(out_fpath)

        print('Saved results to:', out_summary_fpath)

if __name__ == "__main__":

    # settings
    data_subfolder = ''
    labels_dir = data_folder + subfolders['expdata']
    data_suffix_list = [''] 
    fname_prefix = 'GOh1052_'
    labels_actual_fname = 'GOh1052_mutagenesis'
    ylabel_list = ['Fold (Max Abs)']
    splits_to_evaluate =  ['random']
    extra_cols_to_get = ['protein_name', 'mutations']
    output_fname = 'regression_metrics_kfold' 

    # evaluate regression
    evaluate_regression(    
        data_folder,
        data_subfolder,
        labels_dir,
        data_suffix_list,
        fname_prefix,
        labels_actual_fname,
        ylabel_list,
        splits_to_evaluate,
        extra_cols_to_get,
        output_fname,
    )
