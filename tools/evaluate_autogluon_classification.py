#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train and evaluate AutoGluon classification models with random CV and pooled metrics.
"""
from __future__ import annotations

import warnings
import os
from pathlib import Path
import sys
from types import SimpleNamespace

warnings.simplefilter(action='ignore', category=FutureWarning)

DEFAULT_NUM_THREADS = '1'
THREAD_ENV_VARS = [
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS',
]

# Limit native library thread counts to reduce OpenMP / BLAS runtime conflicts
# during AutoGluon training and inference on macOS.
for env_var in THREAD_ENV_VARS:
    os.environ.setdefault(env_var, DEFAULT_NUM_THREADS)

import numpy as np
import pandas as pd

# Allow direct "run file" execution from an IDE while keeping package imports
# working for notebooks and `python -m tools...`.
if __package__ in (None, ''):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

from autogluon.tabular import TabularDataset

from tools.ml_prediction.dataset_builder import (
    as_path,
    assemble_featureset_dataset,
    load_precompiled_dataset,
)
from tools.ml_prediction.model_features import resolve_component_featuresets
from tools.ml_prediction.classification_eval_utils import (
    REFERENCE_METRICS,
    get_filter_thresholds,
    subset_train_by_frac,
    summarize_split_metrics,
    get_result_columns,
    flush_classification_results,
)
from tools.ml_prediction.metrics import (
    get_classification_metrics,
    get_classification_bootstrap_ci,
)
from tools.ml_prediction.splits import get_random_split_idxs
from tools.ml_prediction.autogluon_models import autogluon_classifier
from tools.conservation_and_distance.filter_data import filter_by_score
from tools.utils.variables import data_folder as DEFAULT_DATA_FOLDER, subfolders

CLASS_LABELS = [-1, 0, 1]
AUTOGLUON_LEADERBOARD_METRICS = ['mcc', 'accuracy', 'recall_0', 'recall_1', 'recall_2', 'precision_0', 'precision_1', 'precision_2']


def _default_ml_prediction_dir():
    return str(Path(DEFAULT_DATA_FOLDER) / 'ml_prediction') + '/'


def _default_project_data_dir():
    return str(Path(DEFAULT_DATA_FOLDER)) + '/'


def parse_args():
    import argparse
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_folder', default=_default_ml_prediction_dir(), help='Path to the precompiled ml_prediction directory')
        parser.add_argument('--project_data_folder', default=_default_project_data_dir(), help='Path to the project data directory used for assembled inputs')
        parser.add_argument('--input_mode', default='assembled', choices=['assembled', 'precompiled'], help='Load a precompiled dataset CSV or assemble labels and features from source files')
        parser.add_argument('--train_input_fname', default='GOh1052', help='Dataset stem or CSV path for the training / base input')
        parser.add_argument('--test_input_fname', default='', help='Dataset stem or CSV path for the custom-mode test input')
        parser.add_argument('--labels_actual_fname', default='', help='Optional labels CSV stem override for assembled random-CV mode')
        parser.add_argument('--input_subfolder', default='', help='Optional feature subfolder to search before the top-level feature directories in assembled mode')
        parser.add_argument('--n_splits', default=5, type=int, help='Number of CV folds')
        parser.add_argument('--num_bag_folds', default=8, type=int, help='Number of AutoGluon bagging folds')
        parser.add_argument('--num_stack_levels', default=1, type=int, help='Number of AutoGluon stack levels')
        parser.add_argument('--filt_by', default='', type=str, help='Filter by SIFT, ShanMS, or distance score')
        parser.add_argument('--filt_shanms', default='0.5,1.5', type=str, help='Filter by ShanMS score')
        parser.add_argument('--filt_sift', default='0.1,0.45', type=str, help='Filter by SIFTnorm score')
        parser.add_argument('--filt_dist', default='20,50', type=str, help='Filter by distance to substrate')
        parser.add_argument('--featureset', default='AggStabBind-mut', type=str, help='Feature set name')
        parser.add_argument('--ylabel', default='CategoryV3', type=str, help='Classification label column')
        parser.add_argument('--output_fname', default='', help='Optional output filename stem for saved result CSVs')
        parser.add_argument('--output_subfolder', default='', help='Optional output subfolder under ml_prediction/Output and Predictions')
        parser.add_argument('--save_model', default='', help='Optional model save path. Use None/False/empty for AutoGluon default path, or provide a directory path.')
        parser.add_argument('--load_model', default='', help='Optional saved AutoGluon model directory to load instead of training')
        parser.add_argument('--split_type', default='random', choices=['random', 'custom'], help='Evaluation split type')
        parser.add_argument('--filter_stage', default='train_only', choices=['dataset', 'train_only'], help='Apply the score filter before CV or only on each training fold')
        parser.add_argument('--use_precomputed_folds', action='store_true', help='Reuse fold_random_<n_splits> if present')
        parser.add_argument('--stratify', action='store_true', help='Use StratifiedKFold when generating fresh random splits')
        parser.add_argument('--random_state', default=42, type=int, help='Random seed for fold generation and bootstrap')
        parser.add_argument('--n_boot', default=1000, type=int, help='Bootstrap resamples for pooled confidence intervals')
        parser.add_argument('--ci', default=0.95, type=float, help='Confidence interval level')
        parser.add_argument('--data_fracs', default='1.0', type=str, help='Comma-separated training fractions, e.g. "0.05,0.1,0.2,0.4,0.6,0.8,1.0"')
        parser.add_argument('--save_y_predictions', action='store_true', help='Save out-of-fold test predictions')
        parser.add_argument('--show_only_test_results', action='store_true', help='Only keep test rows in the summary output')
        return parser.parse_args()
    except Exception:
        print("An exception occurred with argument parsing. Check your provided options.")
        traceback.print_exc()
        raise


def _parse_data_fracs(data_fracs_arg):
    if isinstance(data_fracs_arg, str):
        return [float(v) for v in data_fracs_arg.split(',') if str(v).strip() != '']
    return list(data_fracs_arg)


def _normalize_input_stem(input_name):
    if not input_name:
        return ''
    input_path = Path(str(input_name))
    return input_path.stem if input_path.suffix == '.csv' else str(input_name)


def _load_input_dataset(
    args,
    dataset_fbase,
    featureset,
    ylabel,
    required_extra_cols,
    n_splits,
    *,
    labels_actual_fname='',
    fname_prefix='',
    input_name='',
    require_label=True,
    include_label_if_present=False,
):
    keep_label_column = require_label or include_label_if_present

    if args.input_mode == 'precompiled':
        return load_precompiled_dataset(
            input_dir=as_path(args.data_folder) / 'Input',
            dataset_fbase=dataset_fbase,
            featureset=featureset,
            ylabel=ylabel,
            extra_cols_to_get=required_extra_cols,
            n_splits=n_splits,
            input_csv_path=input_name or None,
            require_label=require_label,
        )

    labels_actual_fname = labels_actual_fname or args.labels_actual_fname or dataset_fbase
    fname_prefix = fname_prefix or f'{labels_actual_fname}_'
    component_featureset_list = resolve_component_featuresets(featureset)
    return assemble_featureset_dataset(
        data_folder=as_path(args.project_data_folder),
        data_subfolder=args.input_subfolder,
        labels_dir=as_path(args.project_data_folder) / subfolders['expdata'],
        data_suffix_list=[''],
        fname_prefix=fname_prefix,
        labels_actual_fname=labels_actual_fname,
        ylabel_list=[ylabel] if keep_label_column else [],
        component_featureset_list=component_featureset_list,
        extra_cols_to_get=required_extra_cols,
        filter_out_data={'mutations': ['WT', 'NC', 'X']},
        filter_in_data={},
        merge_on='mutations',
        deduplicate_data=False,
        get_classification_label=True,
    )


def _format_prediction_proba(y_pred_proba):
    if y_pred_proba is None:
        return None
    if isinstance(y_pred_proba, pd.DataFrame):
        proba_df = y_pred_proba.copy()
    else:
        proba_df = pd.DataFrame(y_pred_proba)
    proba_df = proba_df.reset_index(drop=True)
    proba_df.columns = [f'prob_{col}' for col in proba_df.columns]
    return proba_df


def _build_prediction_records(df_subset, y_true, y_pred, y_pred_proba, fold_id, dataset_label, featureset, filt_by):
    pred_cols = [c for c in ['protein_name', 'name', 'mutations', 'Position'] if c in df_subset.columns]
    preds = df_subset[pred_cols].copy()
    if y_true is not None:
        preds['y_true'] = y_true
    preds['y_pred'] = y_pred
    proba_df = _format_prediction_proba(y_pred_proba)
    if proba_df is not None:
        preds = pd.concat([preds.reset_index(drop=True), proba_df], axis=1)
    preds['fold_id'] = fold_id
    preds['dataset'] = dataset_label
    preds['Featureset'] = featureset
    preds['filter'] = filt_by
    return preds


def _compute_fold_metrics(y_true, y_pred, split_label, train_or_test, metrics_list, train_n, test_n):
    metrics = get_classification_metrics(y_pred, y_true, average_method='macro')
    metrics = {metric: metrics[metric] for metric in metrics_list}
    metrics.update({'split_label': split_label, 'train_or_test': train_or_test, 'train_n': train_n, 'test_n': test_n})
    return metrics


def _get_train_predictions(predictor, train_data, best_model_name):
    try:
        y_pred_train = predictor.predict_oof(train_data=train_data.drop(columns=[predictor.label]), model=best_model_name)
    except Exception:
        y_pred_train = predictor.predict(train_data.drop(columns=[predictor.label]), model=best_model_name)
    return np.asarray(y_pred_train)


def _apply_filter(df, filt_by, filt_thres, dataset_label):
    if not filt_by:
        return df
    return filter_by_score(df, filt_by, filt_thres, dataset_label)[0]


def _resolve_save_model_path(save_model_value, default_root, run_subdir):
    if save_model_value in (None, False, '', 'False', 'false', 'None', 'none'):
        return None
    if save_model_value is True:
        return str(default_root / run_subdir)
    save_model_path = Path(str(save_model_value)).expanduser()
    if not save_model_path.is_absolute():
        save_model_path = as_path('.') / save_model_path
    return str(save_model_path / run_subdir)


def _get_output_paths(output_dir, output_fname):
    save_res = output_dir / output_fname
    return {
        'ensemble_metrics_path': f'{save_res}_ensemble_metrics.csv',
        'split_metrics_path': f'{save_res}_split_metrics.csv',
        'split_metrics_summary_path': f'{save_res}_split_metrics_summary.csv',
        'predictions_path': f'{save_res}_predictions_test.csv',
    }


def _add_result_metadata(df, *, split_type, filter_stage, featureset, dataset, ylabel, filt_by, num_features, data_frac, model='autogluon'):
    df = df.copy()
    df['split_type'] = split_type
    df['filter_stage'] = filter_stage
    df['Featureset'] = featureset
    df['dataset'] = dataset
    df['ylabel'] = ylabel
    df['Model'] = model
    df['filter'] = filt_by
    df['p'] = num_features
    df['data_frac'] = data_frac
    return df


def _add_pooled_metric_columns(split_metrics_summary, pooled_metrics=None):
    split_metrics_summary = split_metrics_summary.copy()
    for metric in REFERENCE_METRICS:
        split_metrics_summary[f'{metric}_pooled'] = np.nan
        split_metrics_summary[f'{metric}_ci_low'] = np.nan
        split_metrics_summary[f'{metric}_ci_high'] = np.nan
        if pooled_metrics is not None:
            split_metrics_summary.loc[split_metrics_summary['train_or_test'] == 'test', f'{metric}_pooled'] = pooled_metrics[metric]
            split_metrics_summary.loc[split_metrics_summary['train_or_test'] == 'test', f'{metric}_ci_low'] = pooled_metrics[f'{metric}_ci_low']
            split_metrics_summary.loc[split_metrics_summary['train_or_test'] == 'test', f'{metric}_ci_high'] = pooled_metrics[f'{metric}_ci_high']
    return split_metrics_summary


def _load_custom_datasets(args, featureset, ylabel, required_extra_cols):
    train_input_name = _normalize_input_stem(args.train_input_fname)
    test_input_name = _normalize_input_stem(args.test_input_fname)
    if not train_input_name or not test_input_name:
        raise ValueError('train_input_fname and test_input_fname are required when split_type="custom".')

    train_dataset_fbase = train_input_name
    test_dataset_fbase = test_input_name
    XY_train, x_features = _load_input_dataset(
        args=args,
        dataset_fbase=train_dataset_fbase,
        featureset=featureset,
        ylabel=ylabel,
        required_extra_cols=required_extra_cols,
        n_splits=1,
        labels_actual_fname=train_input_name if args.input_mode == 'assembled' else '',
        fname_prefix=f'{train_input_name}_' if args.input_mode == 'assembled' else '',
        input_name=train_input_name,
        require_label=True,
    )
    XY_test, x_features_test = _load_input_dataset(
        args=args,
        dataset_fbase=test_dataset_fbase,
        featureset=featureset,
        ylabel=ylabel,
        required_extra_cols=[c for c in required_extra_cols if not c.startswith('fold_random_')],
        n_splits=1,
        labels_actual_fname=test_input_name if args.input_mode == 'assembled' else '',
        fname_prefix=f'{test_input_name}_' if args.input_mode == 'assembled' else '',
        input_name=test_input_name,
        require_label=False,
        include_label_if_present=True,
    )
    if list(x_features) != list(x_features_test):
        raise ValueError('Train and test feature columns do not match for custom mode.')
    return train_input_name, test_input_name, XY_train, XY_test, x_features


def _run_custom_train_test(args, dataset_label, train_dataset_label, test_dataset_label, XY_train_full, XY_test_full, x_features, output_paths):
    res_cols, res_cols_summary = get_result_columns(REFERENCE_METRICS)
    data_frac_list = _parse_data_fracs(args.data_fracs)
    y_feature = args.ylabel
    filt_by = args.filt_by or ''
    filt_thres = get_filter_thresholds(filt_by, args.filt_sift, args.filt_shanms, args.filt_dist) if filt_by else None
    test_has_labels = y_feature in XY_test_full.columns

    if filt_by and args.filter_stage == 'dataset':
        XY_train_full = _apply_filter(XY_train_full, filt_by, filt_thres, train_dataset_label)
        XY_test_full = _apply_filter(XY_test_full, filt_by, filt_thres, test_dataset_label)

    ensemble_metrics_all = []
    split_metrics_all = []
    split_metrics_summary_all = []
    predictions_test_all_fracs = []

    for data_frac in data_frac_list:
        print(f'\n>>> Featureset={args.featureset} | data_frac={data_frac} | split=custom')
        XY_train = XY_train_full.copy()
        XY_test = XY_test_full.copy()

        if filt_by and args.filter_stage == 'train_only':
            XY_train = _apply_filter(XY_train, filt_by, filt_thres, train_dataset_label)

        XY_train = subset_train_by_frac(XY_train, data_frac)
        train_n = len(XY_train)
        test_n = len(XY_test)

        train_data = TabularDataset(XY_train[x_features + [y_feature]])
        test_cols = list(x_features) + ([y_feature] if test_has_labels else [])
        test_data = TabularDataset(XY_test[test_cols])

        default_model_root = as_path(args.project_data_folder) / 'ml_prediction' / 'trained_models'
        output_subfolder = args.output_subfolder or ''
        if output_subfolder:
            default_model_root = default_model_root / output_subfolder
        run_subdir = Path(dataset_label) / f'datafrac_{data_frac}' / 'custom'
        save_model = _resolve_save_model_path(args.save_model, default_model_root, run_subdir)

        predictor, y_pred_test, y_pred_test_proba, res = autogluon_classifier(
            train_data,
            test_data,
            label=y_feature,
            metrics=AUTOGLUON_LEADERBOARD_METRICS,
            save_model=save_model,
            load_model=args.load_model or None,
            num_bag_folds=args.num_bag_folds,
            num_stack_levels=args.num_stack_levels,
            test_has_label=test_has_labels,
            class_labels=CLASS_LABELS,
        )

        for train_or_test in ['train', 'test']:
            leaderboard = res.get(train_or_test)
            if leaderboard is not None:
                leaderboard = leaderboard.copy()
                leaderboard['split_label'] = 0
                leaderboard['train_or_test'] = train_or_test
                leaderboard['data_frac'] = data_frac
                leaderboard['Featureset'] = args.featureset
                leaderboard['dataset'] = dataset_label
                ensemble_metrics_all.append(leaderboard)

        best_model_name = res['train'].iloc[0]['model']
        y_pred_train = _get_train_predictions(predictor, train_data, best_model_name)
        y_pred_test = np.asarray(y_pred_test)

        split_metric_rows = []
        train_metrics = _compute_fold_metrics(XY_train[y_feature].to_numpy(), y_pred_train, 0, 'train', REFERENCE_METRICS, train_n, test_n)
        train_metrics['data_frac'] = data_frac
        split_metric_rows.append(train_metrics)

        preds_test = _build_prediction_records(
            XY_test,
            XY_test[y_feature].to_numpy() if test_has_labels else None,
            y_pred_test,
            y_pred_test_proba,
            0,
            dataset_label,
            args.featureset,
            filt_by,
        )
        preds_test['data_frac'] = data_frac
        predictions_test_all_fracs.append(preds_test)

        if test_has_labels:
            test_metrics = _compute_fold_metrics(XY_test[y_feature].to_numpy(), y_pred_test, 0, 'test', REFERENCE_METRICS, train_n, test_n)
            test_metrics['data_frac'] = data_frac
            split_metric_rows.append(test_metrics)

        split_metrics = pd.DataFrame(split_metric_rows)
        split_metrics_summary = summarize_split_metrics(split_metrics, REFERENCE_METRICS)

        pooled_metrics = None
        if test_has_labels:
            pooled_metrics = get_classification_bootstrap_ci(
                XY_test[y_feature].to_numpy(),
                y_pred_test,
                metrics_list=REFERENCE_METRICS,
                class_labels=CLASS_LABELS,
                average_method='macro',
                n_boot=args.n_boot,
                ci=args.ci,
                random_state=args.random_state,
            )

        split_metrics_summary = _add_pooled_metric_columns(split_metrics_summary, pooled_metrics)
        split_metrics_summary = _add_result_metadata(
            split_metrics_summary,
            split_type='custom',
            filter_stage=args.filter_stage,
            featureset=args.featureset,
            dataset=dataset_label,
            ylabel=y_feature,
            filt_by=filt_by,
            num_features=len(x_features),
            data_frac=data_frac,
        )
        split_metrics = _add_result_metadata(
            split_metrics,
            split_type='custom',
            filter_stage=args.filter_stage,
            featureset=args.featureset,
            dataset=dataset_label,
            ylabel=y_feature,
            filt_by=filt_by,
            num_features=len(x_features),
            data_frac=data_frac,
        )

        if args.show_only_test_results and test_has_labels:
            split_metrics_summary = split_metrics_summary[split_metrics_summary['train_or_test'] == 'test'].reset_index(drop=True)
            split_metrics = split_metrics[split_metrics['train_or_test'] == 'test'].reset_index(drop=True)

        split_metrics = split_metrics[[c for c in res_cols if c in split_metrics.columns]]
        split_metrics_summary = split_metrics_summary[[c for c in res_cols_summary if c in split_metrics_summary.columns]]

        split_metrics_all.append(split_metrics)
        split_metrics_summary_all.append(split_metrics_summary)

        ensemble_metrics_current = pd.concat(ensemble_metrics_all, axis=0, ignore_index=True) if ensemble_metrics_all else pd.DataFrame()
        split_metrics_current = pd.concat(split_metrics_all, axis=0, ignore_index=True)
        split_metrics_summary_current = pd.concat(split_metrics_summary_all, axis=0, ignore_index=True)
        predictions_current = pd.concat(predictions_test_all_fracs, axis=0, ignore_index=True)

        should_save_predictions = args.save_y_predictions or not test_has_labels

        flush_classification_results(
            split_metrics_current,
            split_metrics_summary_current,
            output_paths['split_metrics_path'],
            output_paths['split_metrics_summary_path'],
            ensemble_metrics_df=ensemble_metrics_current,
            ensemble_metrics_path=output_paths['ensemble_metrics_path'],
            predictions_df=predictions_current if should_save_predictions else None,
            predictions_path=output_paths['predictions_path'] if should_save_predictions else None,
            digits=4,
        )

    ensemble_metrics_all = pd.concat(ensemble_metrics_all, axis=0, ignore_index=True) if ensemble_metrics_all else pd.DataFrame()
    split_metrics_all = pd.concat(split_metrics_all, axis=0, ignore_index=True)
    split_metrics_summary_all = pd.concat(split_metrics_summary_all, axis=0, ignore_index=True)
    return ensemble_metrics_all, split_metrics_all, split_metrics_summary_all


def ml_autogluon_train_test_random(args=None):
    if args is None:
        args = parse_args()

    project_data_dir = as_path(args.project_data_folder)
    train_input_stem = _normalize_input_stem(args.train_input_fname)
    if not train_input_stem:
        raise ValueError('train_input_fname is required.')
    dataset_fbase = train_input_stem
    n_splits = args.n_splits
    num_bag_folds = args.num_bag_folds
    num_stack_levels = args.num_stack_levels
    filt_by = args.filt_by
    featureset = args.featureset
    data_frac_list = _parse_data_fracs(args.data_fracs)
    y_feature = args.ylabel
    res_cols, res_cols_summary = get_result_columns(REFERENCE_METRICS)
    required_extra_cols = ['protein_name', 'mutations', 'name', 'fold_random_5', f'fold_random_{n_splits}']
    if filt_by and 'Position' not in required_extra_cols:
        required_extra_cols.append('Position')

    output_subfolder = args.output_subfolder or ''
    output_dir = project_data_dir / 'ml_prediction' / 'Output'
    if output_subfolder:
        output_dir = output_dir / output_subfolder
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.split_type == 'custom':
        train_dataset_label, test_dataset_label, XY_train, XY_test, x_features = _load_custom_datasets(
            args,
            featureset,
            y_feature,
            required_extra_cols,
        )
        dataset_label = args.output_fname or train_dataset_label
        print(f'Featureset: {featureset}')
        print(f'Input mode: {args.input_mode}')
        print(f'Label column: {y_feature}')
        print(f'Custom train dataset: {train_dataset_label}')
        print(f'Custom test dataset: {test_dataset_label}')
        print(f'Assembled feature count: {len(x_features)}')

        output_paths = _get_output_paths(output_dir, dataset_label)
        return _run_custom_train_test(
            args,
            dataset_label,
            train_dataset_label,
            test_dataset_label,
            XY_train,
            XY_test,
            x_features,
            output_paths,
        )

    XY_data, x_features = _load_input_dataset(
        args=args,
        dataset_fbase=dataset_fbase,
        featureset=featureset,
        ylabel=y_feature,
        required_extra_cols=required_extra_cols,
        n_splits=n_splits,
        labels_actual_fname=args.labels_actual_fname,
        fname_prefix=f'{train_input_stem}_' if args.input_mode == 'assembled' else '',
        input_name=train_input_stem,
    )
    print(f'Featureset: {featureset}')
    print(f'Input mode: {args.input_mode}')
    print(f'Label column: {y_feature}')
    print(f'Assembled feature count: {len(x_features)}')

    if filt_by:
        filt_thres = get_filter_thresholds(filt_by, args.filt_sift, args.filt_shanms, args.filt_dist)
        if args.filter_stage == 'dataset':
            XY_data, _ = filter_by_score(XY_data, filt_by, filt_thres, dataset_fbase)
        dataset_label = f'{dataset_fbase}_{filt_by}-filt'
    else:
        filt_thres = None
        dataset_label = dataset_fbase
    output_fname = args.output_fname or dataset_label

    split_idxs_list, _ = get_random_split_idxs(
        XY_data,
        n_splits,
        y_col=y_feature,
        use_precomputed_folds=args.use_precomputed_folds,
        stratify=args.stratify,
        random_state=args.random_state,
    )
    print(f'Total samples: {len(XY_data)}')
    print(f'Number of CV splits: {len(split_idxs_list)}')

    output_paths = _get_output_paths(output_dir, output_fname)

    ensemble_metrics_all = []
    split_metrics_all = []
    split_metrics_summary_all = []
    predictions_test_all_fracs = []

    for data_frac in data_frac_list:
        print(f'\n>>> Featureset={featureset} | data_frac={data_frac}')
        ensemble_metrics = []
        split_metrics = []
        predictions_test_all = []

        for split_idx, (train_index, test_index) in enumerate(split_idxs_list):
            print(f'--> Featureset={featureset} | data_frac={data_frac} | split={split_idx + 1}/{n_splits}')
            XY_train = XY_data.iloc[train_index].reset_index(drop=True)
            XY_test = XY_data.iloc[test_index].reset_index(drop=True)

            if filt_by and args.filter_stage == 'train_only':
                XY_train, _ = filter_by_score(XY_train, filt_by, filt_thres, dataset_fbase)

            XY_train = subset_train_by_frac(XY_train, data_frac)
            train_n = len(XY_train)
            test_n = len(XY_test)

            train_data = TabularDataset(XY_train[x_features + [y_feature]])
            test_data = TabularDataset(XY_test[x_features + [y_feature]])
            default_model_root = project_data_dir / 'ml_prediction' / 'trained_models'
            if output_subfolder:
                default_model_root = default_model_root / output_subfolder
            run_subdir = Path(dataset_label) / f'datafrac_{data_frac}' / f'fold_{split_idx}'
            save_model = _resolve_save_model_path(args.save_model, default_model_root, run_subdir)

            predictor, y_pred_test, y_pred_test_proba, res = autogluon_classifier(
                train_data,
                test_data,
                label=y_feature,
                metrics=AUTOGLUON_LEADERBOARD_METRICS,
                save_model=save_model,
                load_model=args.load_model or None,
                num_bag_folds=num_bag_folds,
                num_stack_levels=num_stack_levels,
                class_labels=CLASS_LABELS,
            )

            for train_or_test in ['train', 'test']:
                leaderboard = res.get(train_or_test)
                if leaderboard is not None:
                    leaderboard = leaderboard.copy()
                    leaderboard['split_label'] = split_idx
                    leaderboard['train_or_test'] = train_or_test
                    leaderboard['data_frac'] = data_frac
                    leaderboard['Featureset'] = featureset
                    leaderboard['dataset'] = dataset_label
                    ensemble_metrics += leaderboard.to_dict('records')

            best_model_name = res['train'].iloc[0]['model']
            y_pred_train = _get_train_predictions(predictor, train_data, best_model_name)
            y_pred_test = np.asarray(y_pred_test)

            train_metrics = _compute_fold_metrics(XY_train[y_feature].to_numpy(), y_pred_train, split_idx, 'train', REFERENCE_METRICS, train_n, test_n)
            test_metrics = _compute_fold_metrics(XY_test[y_feature].to_numpy(), y_pred_test, split_idx, 'test', REFERENCE_METRICS, train_n, test_n)
            train_metrics['data_frac'] = data_frac
            test_metrics['data_frac'] = data_frac
            split_metrics.append(train_metrics)
            split_metrics.append(test_metrics)

            preds_test = _build_prediction_records(
                XY_test,
                XY_test[y_feature].to_numpy(),
                y_pred_test,
                y_pred_test_proba,
                split_idx,
                dataset_label,
                featureset,
                filt_by,
            )
            preds_test['data_frac'] = data_frac
            predictions_test_all.append(preds_test)

        split_metrics = pd.DataFrame(split_metrics)
        ensemble_metrics = pd.DataFrame(ensemble_metrics) if ensemble_metrics else pd.DataFrame()
        predictions_test_all = pd.concat(predictions_test_all, axis=0, ignore_index=True)

        pooled_metrics = get_classification_bootstrap_ci(
            predictions_test_all['y_true'].to_numpy(),
            predictions_test_all['y_pred'].to_numpy(),
            metrics_list=REFERENCE_METRICS,
            class_labels=CLASS_LABELS,
            average_method='macro',
            n_boot=args.n_boot,
            ci=args.ci,
            random_state=args.random_state,
        )

        split_metrics_summary = summarize_split_metrics(split_metrics, REFERENCE_METRICS)
        split_metrics_summary = _add_pooled_metric_columns(split_metrics_summary, pooled_metrics)
        split_metrics_summary = _add_result_metadata(
            split_metrics_summary,
            split_type=args.split_type,
            filter_stage=args.filter_stage,
            featureset=featureset,
            dataset=dataset_label,
            ylabel=y_feature,
            filt_by=filt_by,
            num_features=len(x_features),
            data_frac=data_frac,
        )
        split_metrics = _add_result_metadata(
            split_metrics,
            split_type=args.split_type,
            filter_stage=args.filter_stage,
            featureset=featureset,
            dataset=dataset_label,
            ylabel=y_feature,
            filt_by=filt_by,
            num_features=len(x_features),
            data_frac=data_frac,
        )

        if args.show_only_test_results:
            split_metrics_summary = split_metrics_summary[split_metrics_summary['train_or_test'] == 'test'].reset_index(drop=True)
            split_metrics = split_metrics[split_metrics['train_or_test'] == 'test'].reset_index(drop=True)

        split_metrics = split_metrics[[c for c in res_cols if c in split_metrics.columns]]
        split_metrics_summary = split_metrics_summary[[c for c in res_cols_summary if c in split_metrics_summary.columns]]

        if not ensemble_metrics.empty:
            ensemble_metrics_all.append(ensemble_metrics)
        split_metrics_all.append(split_metrics)
        split_metrics_summary_all.append(split_metrics_summary)
        predictions_test_all_fracs.append(predictions_test_all)

        ensemble_metrics_current = pd.concat(ensemble_metrics_all, axis=0, ignore_index=True) if ensemble_metrics_all else pd.DataFrame()
        split_metrics_current = pd.concat(split_metrics_all, axis=0, ignore_index=True)
        split_metrics_summary_current = pd.concat(split_metrics_summary_all, axis=0, ignore_index=True)
        predictions_current = pd.concat(predictions_test_all_fracs, axis=0, ignore_index=True) if predictions_test_all_fracs else pd.DataFrame()

        flush_classification_results(
            split_metrics_current,
            split_metrics_summary_current,
            output_paths['split_metrics_path'],
            output_paths['split_metrics_summary_path'],
            ensemble_metrics_df=ensemble_metrics_current,
            ensemble_metrics_path=output_paths['ensemble_metrics_path'],
            predictions_df=predictions_current if args.save_y_predictions else None,
            predictions_path=output_paths['predictions_path'] if args.save_y_predictions else None,
            digits=4,
        )

    ensemble_metrics_all = pd.concat(ensemble_metrics_all, axis=0, ignore_index=True) if ensemble_metrics_all else pd.DataFrame()
    split_metrics_all = pd.concat(split_metrics_all, axis=0, ignore_index=True)
    split_metrics_summary_all = pd.concat(split_metrics_summary_all, axis=0, ignore_index=True)
    predictions_test_all_fracs = pd.concat(predictions_test_all_fracs, axis=0, ignore_index=True)

    return ensemble_metrics_all, split_metrics_all, split_metrics_summary_all


if __name__ == "__main__":
    # Edit these defaults for direct IDE "run file" execution.
    ide_args = SimpleNamespace(
        data_folder=_default_ml_prediction_dir(),
        project_data_folder=_default_project_data_dir(),
        input_mode='assembled',
        train_input_fname='GOh1052',
        test_input_fname=None,
        labels_actual_fname='',
        input_subfolder='',
        output_subfolder='GOh1052',
        n_splits=4,
        num_bag_folds=8,
        num_stack_levels=1,
        filt_by='',
        filt_shanms=[0.5, 1.5],
        filt_sift=[0.1, 0.45],
        filt_dist=[20, 50],
        featureset='Aggregation',
        ylabel='CategoryV3',
        output_fname='GOh1052_Aggregation',
        save_model='../data/ml_prediction/AutogluonModels',
        load_model=None,
        split_type='random',
        filter_stage='train_only',
        use_precomputed_folds=False,
        stratify=True,
        random_state=42,
        n_boot=1000,
        ci=0.95,
        data_fracs=[1.0],
        save_y_predictions=True,
        show_only_test_results=False,
    )

    # If no CLI args were provided, assume a direct IDE run and use the
    # editable defaults above. If CLI args are present, defer to argparse.
    if len(sys.argv) == 1:
        ml_autogluon_train_test_random(ide_args)
    else:
        ml_autogluon_train_test_random()
