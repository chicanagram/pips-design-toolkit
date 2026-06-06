#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a single non-AutoGluon model with cross-validation and save one
dataset-level SHAP summary CSV plus one summary plot.

Edit the inputs in the __main__ block and run from your IDE.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

DEFAULT_NUM_THREADS = '1'
THREAD_ENV_VARS = [
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS',
]

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/matplotlib')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', DEFAULT_NUM_THREADS)
for env_var in THREAD_ENV_VARS:
    os.environ.setdefault(env_var, DEFAULT_NUM_THREADS)

if __package__ in (None, ''):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from tools.ml_prediction.model_features import COMPOSITE_FEATURESETS, feature_names
from tools.ml_prediction.metrics import get_classification_metrics
from tools.ml_prediction.splits import get_random_split_idxs

VALID_MODEL_TYPES = ('xgb', 'randomforest', 'ridge')
VALID_PLOT_TYPES = ('bar', 'beeswarm')


def resolve_repo_path(path_str):
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parent / path).resolve()


def import_plotting():
    import matplotlib

    matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    import shap

    return plt, shap


def get_feature_columns(featureset):
    if featureset in COMPOSITE_FEATURESETS:
        return list(COMPOSITE_FEATURESETS[featureset])
    if featureset in feature_names and feature_names[featureset] is not None:
        return list(feature_names[featureset])
    raise KeyError(f'Unknown or unsupported featureset: {featureset}')


def load_xy_data(data_path, feature_columns, label_col):
    data = pd.read_csv(data_path)
    required_cols = feature_columns + [label_col]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f'Input data is missing required columns: {missing_cols}')

    valid_mask = np.isfinite(data[feature_columns].to_numpy()).all(axis=1) & data[label_col].notna().to_numpy()
    n_removed = int((~valid_mask).sum())
    if n_removed > 0:
        print(f'Removed {n_removed} rows with NaN/Inf feature values or missing labels.')
    data = data.loc[valid_mask].reset_index(drop=True)
    return data


def build_model(model_type, random_state, num_classes):
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(f'model_type must be one of: {VALID_MODEL_TYPES}.')

    if model_type == 'xgb':
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective='multi:softprob' if num_classes > 2 else 'binary:logistic',
            num_class=num_classes if num_classes > 2 else None,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            eval_metric='mlogloss' if num_classes > 2 else 'logloss',
            n_jobs=1,
        )
    if model_type == 'randomforest':
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=400,
            random_state=random_state,
            n_jobs=1,
        )
    if model_type == 'ridge':
        from sklearn.linear_model import RidgeClassifier

        return RidgeClassifier(
            alpha=0.01,
            random_state=random_state,
        )
    raise AssertionError(f'Unhandled model_type: {model_type}')


def build_explainer(model_type, model, x_train):
    _, shap = import_plotting()

    if model_type in ('xgb', 'randomforest'):
        return shap.TreeExplainer(model)
    if model_type == 'ridge':
        return shap.LinearExplainer(model, x_train)
    raise AssertionError(f'Unhandled model_type: {model_type}')


def get_class_mappings(y):
    class_labels = sorted(y.unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    y_idx = y.map(label_to_idx).to_numpy()
    return class_labels, label_to_idx, idx_to_label, y_idx


def normalize_multiclass_shap_values(raw_values, n_rows, n_features, n_classes):
    if isinstance(raw_values, list):
        return np.stack(raw_values, axis=2)

    raw_values = np.asarray(raw_values)
    if raw_values.ndim == 2:
        return raw_values
    if raw_values.ndim != 3:
        raise ValueError(f'Unsupported SHAP values shape: {raw_values.shape}')

    if raw_values.shape == (n_rows, n_features, n_classes):
        return raw_values
    if raw_values.shape == (n_rows, n_classes, n_features):
        return np.transpose(raw_values, (0, 2, 1))
    if raw_values.shape == (n_classes, n_rows, n_features):
        return np.transpose(raw_values, (1, 2, 0))
    raise ValueError(f'Unsupported SHAP values shape: {raw_values.shape}')


def select_shap_matrix(shap_values, predicted_class_idx, class_index, n_rows):
    if np.asarray(shap_values).ndim == 2:
        return shap_values
    if class_index is not None:
        return shap_values[:, :, class_index]
    return shap_values[np.arange(n_rows), :, predicted_class_idx]


def fit_cv_and_collect_shap(
    data,
    feature_columns,
    label_col,
    model_type='xgb',
    n_splits=4,
    stratify=True,
    random_state=42,
    explain_class_label=None,
):
    class_labels, label_to_idx, idx_to_label, y_idx = get_class_mappings(data[label_col])

    split_idxs_list, _ = get_random_split_idxs(
        data,
        n_splits=n_splits,
        y_col=label_col,
        use_precomputed_folds=False,
        stratify=stratify,
        random_state=random_state,
    )

    shap_frames = []
    feature_frames = []
    prediction_rows = []
    fold_metrics_rows = []

    for fold_id, (train_index, test_index) in enumerate(split_idxs_list):
        x_train = data.loc[train_index, feature_columns].to_numpy()
        x_test = data.loc[test_index, feature_columns].to_numpy()
        y_train = y_idx[train_index]
        y_test = y_idx[test_index]

        model = build_model(model_type, random_state + fold_id, num_classes=len(class_labels))
        model.fit(x_train, y_train)

        y_pred_idx = model.predict(x_test)
        y_test_labels = np.asarray([idx_to_label[idx] for idx in y_test])
        y_pred_labels = np.asarray([idx_to_label[idx] for idx in y_pred_idx])
        fold_metrics = get_classification_metrics(
            y_pred=y_pred_labels,
            y_true=y_test_labels,
            model_name=model_type,
            class_labels=class_labels,
            average_method='macro',
        )
        fold_metrics_rows.append({
            'fold_id': fold_id,
            'test_n': len(x_test),
            'MCC': fold_metrics['MCC'],
            'Accuracy': fold_metrics['Accuracy'],
            'F1': fold_metrics['F1'],
            'Precision': fold_metrics['Precision'],
            'Recall': fold_metrics['Recall'],
        })
        explainer = build_explainer(model_type, model, x_train)
        raw_shap_values = explainer.shap_values(x_test)

        class_index = None
        if explain_class_label is not None:
            if explain_class_label not in label_to_idx:
                raise ValueError(f'explain_class_label={explain_class_label} not found in labels: {class_labels}')
            class_index = label_to_idx[explain_class_label]

        shap_values = normalize_multiclass_shap_values(
            raw_values=raw_shap_values,
            n_rows=len(x_test),
            n_features=len(feature_columns),
            n_classes=len(class_labels),
        )
        shap_matrix = select_shap_matrix(
            shap_values=shap_values,
            predicted_class_idx=y_pred_idx,
            class_index=class_index,
            n_rows=len(x_test),
        )

        shap_frames.append(pd.DataFrame(shap_matrix, columns=feature_columns, index=test_index))
        feature_frames.append(pd.DataFrame(x_test, columns=feature_columns, index=test_index))
        prediction_rows.append(
            pd.DataFrame({
                'row_index': test_index,
                'fold_id': fold_id,
                'y_true': y_test_labels,
                'y_pred': y_pred_labels,
            }).set_index('row_index')
        )
        print(
            f"Fold {fold_id + 1}/{len(split_idxs_list)} "
            f"| MCC={fold_metrics['MCC']:.3f} "
            f"| Accuracy={fold_metrics['Accuracy']:.3f} "
            f"| F1={fold_metrics['F1']:.3f} "
            f"| Precision={fold_metrics['Precision']:.3f} "
            f"| Recall={fold_metrics['Recall']:.3f}"
        )

    shap_df = pd.concat(shap_frames, axis=0).sort_index()
    feature_df = pd.concat(feature_frames, axis=0).sort_index()
    predictions_df = pd.concat(prediction_rows, axis=0).sort_index()
    fold_metrics_df = pd.DataFrame(fold_metrics_rows)
    return shap_df, feature_df, predictions_df, fold_metrics_df


def save_summary_outputs(
    shap_df,
    feature_df,
    output_dir,
    model_type,
    plot_type='beeswarm',
    max_display=20,
):
    plt, shap = import_plotting()
    if plot_type not in VALID_PLOT_TYPES:
        raise ValueError(f'plot_type must be one of: {VALID_PLOT_TYPES}.')

    summary_df = pd.DataFrame({
        'feature': shap_df.columns,
        'mean_abs_shap_value': np.abs(shap_df.to_numpy()).mean(axis=0),
    }).sort_values(by='mean_abs_shap_value', ascending=False)
    summary_df['model_type'] = model_type
    summary_df['n_rows_explained'] = len(shap_df)

    summary_csv = output_dir / f'shap_{model_type}_summary_mean_abs.csv'
    summary_df.to_csv(summary_csv, index=False)

    explanation = shap.Explanation(
        values=shap_df.to_numpy(),
        data=feature_df.loc[shap_df.index, shap_df.columns].to_numpy(),
        feature_names=list(shap_df.columns),
    )

    plt.figure(figsize=(10, 8 if plot_type == 'beeswarm' else 6))
    if plot_type == 'bar':
        shap.plots.bar(explanation, max_display=min(max_display, len(shap_df.columns)), show=False)
        plot_path = output_dir / f'shap_{model_type}_summary_bar.png'
    else:
        shap.plots.beeswarm(explanation, max_display=min(max_display, len(shap_df.columns)), show=False)
        plot_path = output_dir / f'shap_{model_type}_summary_beeswarm.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Saved summary SHAP CSV: {summary_csv}')
    print(f'Saved summary SHAP plot: {plot_path}')
    return summary_df


if __name__ == '__main__':
    DATA_PATH = '../data/ml_prediction/Input/GOh1052_AggStabBind-mut.csv'
    OUTPUT_DIR = '../data/ml_prediction/Output/shap/'

    FEATURESET = 'AggStabBind-mut'
    LABEL_COL = 'CategoryV3'
    MODEL_TYPE = 'randomforest' # 'ridge' # 'xgb'  #
    N_SPLITS = 4
    STRATIFY = True
    RANDOM_STATE = 42

    EXPLAIN_CLASS_LABEL = None
    SUMMARY_PLOT_TYPE = 'beeswarm'
    MAX_DISPLAY = 20

    data_path = resolve_repo_path(DATA_PATH)
    output_dir = resolve_repo_path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = get_feature_columns(FEATURESET)
    data = load_xy_data(data_path, feature_columns, LABEL_COL)

    print(f'Loaded data: {data_path}')
    print(f'Rows used: {len(data)}')
    print(f'Featureset: {FEATURESET} ({len(feature_columns)} features)')
    print(f'Model type: {MODEL_TYPE}')
    print(f'Splits: {N_SPLITS}, stratify={STRATIFY}')

    shap_df, feature_df, predictions_df, fold_metrics_df = fit_cv_and_collect_shap(
        data=data,
        feature_columns=feature_columns,
        label_col=LABEL_COL,
        model_type=MODEL_TYPE,
        n_splits=N_SPLITS,
        stratify=STRATIFY,
        random_state=RANDOM_STATE,
        explain_class_label=EXPLAIN_CLASS_LABEL,
    )

    predictions_path = output_dir / f'shap_{MODEL_TYPE}_oof_predictions.csv'
    predictions_df.to_csv(predictions_path)
    print(f'Saved out-of-fold predictions: {predictions_path}')

    metrics_path = output_dir / f'shap_{MODEL_TYPE}_fold_metrics.csv'
    fold_metrics_df.to_csv(metrics_path, index=False)
    print(f'Saved fold metrics: {metrics_path}')

    save_summary_outputs(
        shap_df=shap_df,
        feature_df=feature_df,
        output_dir=output_dir,
        model_type=MODEL_TYPE,
        plot_type=SUMMARY_PLOT_TYPE,
        max_display=MAX_DISPLAY,
    )
