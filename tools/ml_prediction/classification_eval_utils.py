from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE_METRICS = [
    'MCC', 'Accuracy', 'F1', 'Precision', 'Recall',
    'Precision_-1', 'Precision_0', 'Precision_1',
    'Recall_-1', 'Recall_0', 'Recall_1',
]


def coerce_float_list(values):
    if isinstance(values, str):
        return [float(v) for v in values.split(',') if str(v).strip() != '']
    return list(values)


def get_filter_thresholds(filt_by, filt_sift, filt_shanms, filt_dist):
    if filt_by == 'sift':
        return coerce_float_list(filt_sift)
    if filt_by == 'shanMS':
        return coerce_float_list(filt_shanms)
    if filt_by == 'distance':
        return coerce_float_list(filt_dist)
    raise ValueError(f'Unsupported filt_by value: {filt_by}')


def subset_train_by_frac(XY_train, data_frac):
    if data_frac >= 1:
        return XY_train.reset_index(drop=True)
    n_keep = max(1, int(len(XY_train) * data_frac))
    return XY_train.iloc[:n_keep].reset_index(drop=True)


def summarize_split_metrics(split_metrics_df, metrics_list):
    summary_rows = []
    for train_or_test in ['train', 'test']:
        subset = split_metrics_df.loc[split_metrics_df['train_or_test'] == train_or_test]
        if subset.empty:
            continue
        row = {'train_or_test': train_or_test}
        for metric in metrics_list:
            row[metric] = subset[metric].mean()
            row[f'{metric}_std'] = subset[metric].std()
        row['train_n'] = subset['train_n'].mean()
        row['test_n'] = subset['test_n'].mean()
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def get_result_columns(metrics_list):
    split_cols = [
        'dataset', 'ylabel', 'Featureset', 'Model', 'split_type', 'filter', 'filter_stage',
        'data_frac', 'p', 'train_n', 'test_n', 'train_or_test', 'split_label',
    ] + list(metrics_list)
    summary_cols = [
        'dataset', 'ylabel', 'Featureset', 'Model', 'split_type', 'filter', 'filter_stage',
        'data_frac', 'p', 'train_n', 'test_n', 'train_or_test',
    ] + list(metrics_list)
    summary_cols += [f'{metric}_pooled' for metric in metrics_list]
    summary_cols += [ci_col for metric in metrics_list for ci_col in (f'{metric}_ci_low', f'{metric}_ci_high')]
    summary_cols += [f'{metric}_std' for metric in metrics_list]
    return split_cols, summary_cols


def round_numeric_columns(df, digits=4):
    if df is None or df.empty:
        return df
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(digits)
    return df


def _write_csv_atomic(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def flush_classification_results(
    split_metrics_df,
    split_metrics_summary_df,
    split_metrics_path,
    split_metrics_summary_path,
    *,
    ensemble_metrics_df=None,
    ensemble_metrics_path=None,
    predictions_df=None,
    predictions_path=None,
    digits=4,
):
    split_metrics_df = round_numeric_columns(split_metrics_df, digits=digits)
    split_metrics_summary_df = round_numeric_columns(split_metrics_summary_df, digits=digits)
    _write_csv_atomic(split_metrics_df, split_metrics_path)
    _write_csv_atomic(split_metrics_summary_df, split_metrics_summary_path)

    if ensemble_metrics_df is not None and ensemble_metrics_path is not None and not ensemble_metrics_df.empty:
        ensemble_metrics_df = round_numeric_columns(ensemble_metrics_df, digits=digits)
        _write_csv_atomic(ensemble_metrics_df, ensemble_metrics_path)

    if predictions_df is not None and predictions_path is not None and not predictions_df.empty:
        predictions_df = round_numeric_columns(predictions_df, digits=digits)
        _write_csv_atomic(predictions_df, predictions_path)
