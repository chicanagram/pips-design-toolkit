#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification evaluation utilities for the pips-design-toolkit datasets.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import numpy as np

# Allow direct "run file" execution from an IDE while keeping package imports
# working for notebooks and `python -m tools...`.
if __package__ in (None, ''):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)

from tools.utils.variables import data_folder as DEFAULT_DATA_FOLDER, subfolders
from tools.ml_prediction.model_features import get_feature_combinations
from tools.ml_prediction.classification_eval_utils import (
    REFERENCE_METRICS,
    get_filter_thresholds,
    subset_train_by_frac,
    summarize_split_metrics,
    get_result_columns,
    flush_classification_results,
)
from tools.ml_prediction.metrics import get_classification_bootstrap_ci
from tools.ml_prediction.splits import (
    get_protein_splits,
    get_mutres_modulo_splits,
    get_custom_split,
    get_split_idxs_from_col,
    get_random_split_idxs,
)
from tools.ml_prediction.dataset_builder import (
    as_path,
    assemble_featureset_dataset,
    infer_dataset_fbase,
)
from tools.ml_prediction.sklearn_models import sklearn_classifier
from tools.conservation_and_distance.filter_data import filter_by_score


def _get_split_indices(df, ylabel, split_type, n_splits, labels_dir, custom_test_fname, use_precomputed_folds, stratify, random_state):
    if split_type == 'random':
        split_idxs_list, _ = get_random_split_idxs(
            df,
            n_splits=n_splits,
            y_col=ylabel,
            use_precomputed_folds=use_precomputed_folds,
            stratify=stratify,
            random_state=random_state,
        )
    elif split_type == 'mutres-modulo':
        split_col = f'fold_modulo_{n_splits}'
        if split_col in df:
            split_idxs_list = get_split_idxs_from_col(df, split_col, n_splits=n_splits)
        else:
            split_idxs_list, _ = get_mutres_modulo_splits(df, n_splits=n_splits)
    elif split_type == 'protein':
        split_col = f'fold_protein_{n_splits}'
        if split_col in df:
            split_idxs_list = get_split_idxs_from_col(df, split_col, n_splits=n_splits)
        else:
            split_idxs_list, _ = get_protein_splits(df, n_splits=n_splits)
    elif split_type == 'custom':
        split_idxs_list = get_custom_split(df, str(Path(labels_dir) / custom_test_fname))
    else:
        raise ValueError(f'Unsupported split_type: {split_type}')
    return split_idxs_list


def _build_predictions_df(XY, ylabel, ypred, fold_ids, featureset_label, model_type, split_type, data_frac):
    prediction_cols = [c for c in ['protein_name', 'mutations', 'name'] if c in XY.columns]
    predictions = XY[prediction_cols + [ylabel]].copy()
    predictions = predictions.rename(columns={ylabel: 'y_true'})
    predictions['y_pred'] = ypred
    predictions['fold_id'] = fold_ids
    predictions['ylabel'] = ylabel
    predictions['Featureset'] = featureset_label
    predictions['Model'] = model_type
    predictions['split_type'] = split_type
    predictions['data_frac'] = data_frac
    return predictions


def _get_prediction_output_path(data_folder, data_subfolder, output_fname, split_type, featureset_label, ylabel):
    predictions_dir = as_path(data_folder) / 'ml_prediction' / 'Predictions'
    if data_subfolder:
        predictions_dir = predictions_dir / data_subfolder
    predictions_dir.mkdir(parents=True, exist_ok=True)
    fname = f'{output_fname}_{split_type}_{featureset_label}_ypred_{ylabel}.csv'
    return predictions_dir / fname


def _drop_invalid_feature_rows(XY, feature_cols, split_name):
    X = XY[feature_cols].to_numpy()
    valid_mask = ~np.any(np.isinf(X) | np.isnan(X), axis=1)
    n_removed = int((~valid_mask).sum())
    if n_removed > 0:
        print(f'{split_name} invalid-feature rows removed: {n_removed}')
    return XY.loc[valid_mask].reset_index(drop=True)


def _fit_sklearn_classifier_kfold_df(
    XY,
    feature_cols,
    ylabel,
    model_dict,
    class_labels,
    split_idxs_list,
    data_frac,
    scale_data,
    average_method,
    filt_by,
    filt_thres,
    filter_stage,
    dataset_fbase,
    featureset_label,
    split_type,
):
    split_metrics = []
    predictions_test = []

    for split_idx, (train_index, test_index) in enumerate(split_idxs_list):
        XY_train = XY.iloc[train_index].reset_index(drop=True)
        XY_test = XY.iloc[test_index].reset_index(drop=True)

        if filt_by and filter_stage == 'train_only':
            XY_train, _ = filter_by_score(XY_train, filt_by, filt_thres, dataset_fbase)

        XY_train = subset_train_by_frac(XY_train, data_frac)
        XY_train = _drop_invalid_feature_rows(XY_train, feature_cols, 'train')
        XY_test = _drop_invalid_feature_rows(XY_test, feature_cols, 'test')

        train_n = len(XY_train)
        test_n = len(XY_test)
        print(f'Split {split_idx + 1}/{len(split_idxs_list)}')

        metrics, _, ypred_test = sklearn_classifier(
            XY_train[feature_cols].to_numpy(),
            XY_train[ylabel].to_numpy(),
            XY_test[feature_cols].to_numpy(),
            XY_test[ylabel].to_numpy(),
            model_dict,
            class_labels=class_labels,
            print_res=True,
            scale_data=scale_data,
            multiclass_average_method=average_method,
        )

        for metric_row in metrics:
            metric_row = dict(metric_row)
            metric_row['split_label'] = split_idx
            metric_row['train_n'] = train_n
            metric_row['test_n'] = test_n
            split_metrics.append(metric_row)

        preds = _build_predictions_df(
            XY_test,
            ylabel,
            np.asarray(ypred_test),
            np.full(test_n, split_idx),
            featureset_label,
            model_dict['model_type'],
            split_type,
            data_frac,
        )
        predictions_test.append(preds)

    return pd.DataFrame(split_metrics), pd.concat(predictions_test, axis=0, ignore_index=True)


def evaluate_classification(
    data_folder,
    data_subfolder,
    labels_dir,
    data_suffix_list,
    fname_prefix,
    labels_actual_fname,
    ylabel_list,
    feature_combinations,
    splits_to_evaluate,
    extra_cols_to_get,
    output_fname,
    class_labels=None,
    data_frac_list=None,
    custom_test_fname=None,
    filter_out_data=None,
    filter_in_data=None,
    merge_on='mutations',
    scale_data=False,
    deduplicate_data=False,
    n_splits=5,
    metrics_list=None,
    models_to_eval_list=None,
    average_method='macro',
    show_only_test_results=True,
    save_y_predictions=False,
    train_model_on_fulldata=False,
    filt_by='',
    filt_shanms=(0.5, 1.5),
    filt_sift=(0.1, 0.45),
    filt_dist=(20, 50),
    filter_stage='train_only',
    n_boot=1000,
    ci=0.95,
    bootstrap_random_state=42,
    use_precomputed_folds=True,
    stratify=False,
    random_state=42,
):
    data_folder = as_path(data_folder)
    labels_dir = as_path(labels_dir)
    if class_labels is None:
        class_labels = [-1, 0, 1]
    if data_frac_list is None:
        data_frac_list = [1]
    if filter_out_data is None:
        filter_out_data = {'mutations': ['WT', 'NC', 'X']}
    if filter_in_data is None:
        filter_in_data = {}
    if metrics_list is None:
        metrics_list = list(REFERENCE_METRICS)
    if models_to_eval_list is None:
        models_to_eval_list = ['xgb']
    dataset_fbase = infer_dataset_fbase(fname_prefix, labels_actual_fname)
    dataset_label = f'{dataset_fbase}_{filt_by}-filt' if filt_by else dataset_fbase
    filt_thres = get_filter_thresholds(filt_by, filt_sift, filt_shanms, filt_dist) if filt_by else None
    required_extra_cols = list(extra_cols_to_get)
    if filt_by and 'Position' not in required_extra_cols:
        required_extra_cols.append('Position')

    res_cols, res_cols_summary = get_result_columns(metrics_list)

    for split_type in splits_to_evaluate:
        print('SPLIT TYPE:', split_type)
        metrics_kfold_all = pd.DataFrame()
        metrics_kfold_summary_all = pd.DataFrame()
        output_dir = data_folder / 'ml_prediction' / 'Output'
        if data_subfolder:
            output_dir = output_dir / data_subfolder
        output_dir.mkdir(parents=True, exist_ok=True)
        out_summary_fpath = output_dir / f'{output_fname}_{split_type}_summary.csv'
        out_fpath = output_dir / f'{output_fname}_{split_type}.csv'

        for i, (featureset_label, component_featureset_list) in enumerate(feature_combinations.items()):
            print('\n********************************')
            print(i, featureset_label)
            print('********************************')
            print('Component featureset list:', component_featureset_list)
            df, feature_list_all = assemble_featureset_dataset(
                data_folder=data_folder,
                data_subfolder=data_subfolder,
                labels_dir=labels_dir,
                data_suffix_list=data_suffix_list,
                fname_prefix=fname_prefix,
                labels_actual_fname=labels_actual_fname,
                ylabel_list=ylabel_list,
                component_featureset_list=component_featureset_list,
                extra_cols_to_get=required_extra_cols,
                filter_out_data=filter_out_data,
                filter_in_data=filter_in_data,
                merge_on=merge_on,
                deduplicate_data=deduplicate_data,
                get_classification_label=True,
            )
            if filt_by and filter_stage == 'dataset':
                df, _ = filter_by_score(df, filt_by, filt_thres, dataset_fbase)
                print('FINAL XY DATAFRAME SIZE AFTER FILTER:', df.shape)
            print('FINAL XY DATAFRAME SIZE:', df.shape)

            for ylabel in ylabel_list:
                XY = df[[c for c in required_extra_cols + feature_list_all + [ylabel] if c in df.columns]]
                split_idxs_list = _get_split_indices(
                    XY,
                    ylabel,
                    split_type,
                    n_splits,
                    labels_dir,
                    custom_test_fname,
                    use_precomputed_folds,
                    stratify,
                    random_state,
                )
                num_samples = XY.shape[0]
                num_features = len(feature_list_all)
                print('num_samples:', num_samples, '; num_features:', num_features)

                for data_frac in data_frac_list:
                    print('\n>>> Processing data fraction:', data_frac)
                    for model_type in models_to_eval_list:
                        print('MODEL TYPE:', model_type)
                        model_dict = {'model_type': model_type}

                        if 'ohe' in featureset_label:
                            scale_data = False

                        print('Fitting classifier on k-folds', split_type)
                        metrics_kfold, predictions = _fit_sklearn_classifier_kfold_df(
                            XY,
                            feature_list_all,
                            ylabel,
                            model_dict,
                            class_labels,
                            split_idxs_list,
                            data_frac,
                            scale_data,
                            average_method,
                            filt_by,
                            filt_thres,
                            filter_stage,
                            dataset_fbase,
                            featureset_label,
                            split_type,
                        )

                        metrics_kfold_summary = summarize_split_metrics(metrics_kfold, metrics_list)

                        metrics_all = get_classification_bootstrap_ci(
                            predictions['y_true'].to_numpy(),
                            predictions['y_pred'].to_numpy(),
                            metrics_list=metrics_list,
                            class_labels=class_labels,
                            average_method=average_method,
                            n_boot=n_boot,
                            ci=ci,
                            random_state=bootstrap_random_state,
                        )
                        metrics_all = {k: float(round(v, 4)) if pd.notna(v) else np.nan for k, v in metrics_all.items()}
                        print('metrics_all:', metrics_all)

                        for metric in metrics_list:
                            metrics_kfold_summary[f'{metric}_pooled'] = np.nan
                            metrics_kfold_summary[f'{metric}_ci_low'] = np.nan
                            metrics_kfold_summary[f'{metric}_ci_high'] = np.nan
                            metrics_kfold_summary.loc[metrics_kfold_summary['train_or_test'] == 'test', f'{metric}_pooled'] = metrics_all[metric]
                            metrics_kfold_summary.loc[metrics_kfold_summary['train_or_test'] == 'test', f'{metric}_ci_low'] = metrics_all[f'{metric}_ci_low']
                            metrics_kfold_summary.loc[metrics_kfold_summary['train_or_test'] == 'test', f'{metric}_ci_high'] = metrics_all[f'{metric}_ci_high']

                        if save_y_predictions:
                            predictions.to_csv(
                                _get_prediction_output_path(
                                    data_folder,
                                    data_subfolder,
                                    output_fname,
                                    split_type,
                                    featureset_label,
                                    ylabel,
                                ),
                                index=False,
                            )

                        metrics_kfold['data_frac'] = data_frac
                        metrics_kfold_summary['data_frac'] = data_frac
                        metrics_kfold['dataset'] = dataset_label
                        metrics_kfold_summary['dataset'] = dataset_label
                        metrics_kfold['ylabel'] = ylabel
                        metrics_kfold_summary['ylabel'] = ylabel
                        metrics_kfold['Featureset'] = featureset_label
                        metrics_kfold_summary['Featureset'] = featureset_label
                        metrics_kfold['p'] = num_features
                        metrics_kfold_summary['p'] = num_features
                        metrics_kfold['Model'] = model_dict['model_type']
                        metrics_kfold_summary['Model'] = model_dict['model_type']
                        metrics_kfold['split_type'] = split_type
                        metrics_kfold_summary['split_type'] = split_type
                        metrics_kfold['filter'] = filt_by
                        metrics_kfold_summary['filter'] = filt_by
                        metrics_kfold['filter_stage'] = filter_stage
                        metrics_kfold_summary['filter_stage'] = filter_stage
                        metrics_kfold = metrics_kfold[[c for c in res_cols if c in metrics_kfold.columns]]
                        metrics_kfold_summary = metrics_kfold_summary[[c for c in res_cols_summary if c in metrics_kfold_summary.columns]]
                        print('K-Fold summary')
                        print(metrics_kfold_summary)

                        if metrics_kfold_all.empty:
                            metrics_kfold_all = metrics_kfold.copy()
                            metrics_kfold_summary_all = metrics_kfold_summary.copy()
                        else:
                            metrics_kfold_all = pd.concat([metrics_kfold_all, metrics_kfold], axis=0, ignore_index=True)
                            metrics_kfold_summary_all = pd.concat([metrics_kfold_summary_all, metrics_kfold_summary], axis=0, ignore_index=True)

                        metrics_kfold_all_current = metrics_kfold_all.copy()
                        metrics_kfold_summary_all_current = metrics_kfold_summary_all.copy()
                        if show_only_test_results:
                            metrics_kfold_all_current = metrics_kfold_all_current[metrics_kfold_all_current['train_or_test'] == 'test']
                            metrics_kfold_summary_all_current = metrics_kfold_summary_all_current[metrics_kfold_summary_all_current['train_or_test'] == 'test']
                        metrics_kfold_summary_all_current = metrics_kfold_summary_all_current.dropna(axis=1, how='all')
                        metrics_kfold_summary_all_current = metrics_kfold_summary_all_current.sort_values(by=['ylabel', 'p'])
                        flush_classification_results(
                            metrics_kfold_all_current,
                            metrics_kfold_summary_all_current,
                            out_fpath,
                            out_summary_fpath,
                            digits=4,
                        )

                        if train_model_on_fulldata:
                            print('Training classifier on full dataset...')
                            import joblib
                            XY_train_full = XY.copy()
                            if filt_by:
                                XY_train_full, _ = filter_by_score(XY_train_full, filt_by, filt_thres, dataset_fbase)
                            XY_train_full = _drop_invalid_feature_rows(XY_train_full, feature_list_all, 'train_full')
                            _, model, _ = sklearn_classifier(
                                XY_train_full[feature_list_all].to_numpy(),
                                XY_train_full[ylabel].to_numpy(),
                                XY_train_full[feature_list_all].to_numpy(),
                                XY_train_full[ylabel].to_numpy(),
                                model_dict,
                                class_labels=class_labels,
                                print_res=True,
                                scale_data=scale_data,
                                multiclass_average_method=average_method,
                            )
                            model_dir = data_folder / 'ml_prediction' / 'trained_models'
                            if data_subfolder:
                                model_dir = model_dir / data_subfolder
                            model_dir.mkdir(parents=True, exist_ok=True)
                            model_fpath = model_dir / f'{model_type}_{ylabel}_{featureset_label}_{data_frac}.joblib'
                            joblib.dump(model, model_fpath)
                            print('Saved trained model:', model_fpath)

        if show_only_test_results:
            metrics_kfold_summary_all = metrics_kfold_summary_all[metrics_kfold_summary_all['train_or_test'] == 'test']
            metrics_kfold_all = metrics_kfold_all[metrics_kfold_all['train_or_test'] == 'test']

        metrics_kfold_summary_all = metrics_kfold_summary_all.dropna(axis=1, how='all')
        metrics_kfold_summary_all = metrics_kfold_summary_all.sort_values(by=['ylabel', 'p'])
        flush_classification_results(
            metrics_kfold_all,
            metrics_kfold_summary_all,
            out_fpath,
            out_summary_fpath,
            digits=4,
        )

        print('Saved results to:', out_summary_fpath)
        print('Saved split metrics to:', out_fpath)

    return metrics_kfold_all, metrics_kfold_summary_all


if __name__ == "__main__":
    data_folder = DEFAULT_DATA_FOLDER
    data_subfolder = 'GOh1052'
    labels_actual_fname = 'GOh1052_mutagenesis'
    labels_dir = Path(data_folder) / subfolders['expdata']
    data_suffix_list = ['']
    fname_prefix = 'GOh1052_'
    ylabel_list = ['CategoryV3']
    class_labels = [-1, 0, 1]
    plm_name_list = ['esm2-33']
    include_nonplm_featuresets = True
    splits_to_evaluate = ['random']
    extra_cols_to_get = ['protein_name', 'mutations', 'name', 'fold_random_5']
    output_fname = 'classification_metrics'
    data_frac_list = [1.0]
    scale_data = False
    deduplicate_data = False
    train_model_on_fulldata = False
    models_to_eval_list = ['xgb', 'ridge']
    custom_test_fname = None
    save_y_predictions = True
    filt_by = ''
    filt_shanms = [0.5, 1.5]
    filt_sift = [0.1, 0.45]
    filt_dist = [20, 50]
    filter_stage = 'train_only'
    use_precomputed_folds = False
    stratify = True
    random_state = 42

    feature_combinations = get_feature_combinations(
        plm_name_list,
        feature_combinations={},
        include_nonplm_featuresets=include_nonplm_featuresets,
    )

    evaluate_classification(
        data_folder,
        data_subfolder,
        labels_dir,
        data_suffix_list,
        fname_prefix,
        labels_actual_fname,
        ylabel_list,
        feature_combinations,
        splits_to_evaluate,
        extra_cols_to_get,
        output_fname,
        class_labels=class_labels,
        data_frac_list=data_frac_list,
        custom_test_fname=custom_test_fname,
        save_y_predictions=save_y_predictions,
        scale_data=scale_data,
        deduplicate_data=deduplicate_data,
        train_model_on_fulldata=train_model_on_fulldata,
        models_to_eval_list=models_to_eval_list,
        filt_by=filt_by,
        filt_shanms=filt_shanms,
        filt_sift=filt_sift,
        filt_dist=filt_dist,
        filter_stage=filter_stage,
        use_precomputed_folds=use_precomputed_folds,
        stratify=stratify,
        random_state=random_state,
    )
