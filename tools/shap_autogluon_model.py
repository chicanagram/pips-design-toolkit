#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run SHAP analysis for a saved AutoGluon model and save outputs to:
../data/ml_prediction/Output/shap/

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

import pandas as pd
from autogluon.tabular import TabularPredictor


def resolve_repo_path(path_str):
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parent / path).resolve()


def choose_fast_model_name(predictor, model_name=''):
    if model_name:
        return model_name

    leaderboard = predictor.leaderboard(silent=True)
    non_ensemble = leaderboard[~leaderboard['model'].str.contains('WeightedEnsemble', na=False)]
    if not non_ensemble.empty:
        return non_ensemble.iloc[0]['model']
    return predictor.model_best


def import_shap_plotting():
    import matplotlib

    matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    import shap

    return plt, shap


def resolve_row_index(row_index, n_rows):
    resolved_index = row_index if row_index >= 0 else n_rows + row_index
    if resolved_index < 0 or resolved_index >= n_rows:
        raise IndexError(f'row_index={row_index} is out of bounds for dataset with {n_rows} rows.')
    return resolved_index


def select_model_data(data, predictor, include_label=True):
    model_columns = list(predictor.original_features)
    missing_columns = [col for col in model_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f'Input data is missing trained feature columns: {missing_columns}')

    selected_columns = model_columns.copy()
    if include_label and predictor.label in data.columns:
        selected_columns.append(predictor.label)
    return data.loc[:, selected_columns].copy()


def resolve_class_index(predictor, row_features, model_name, class_label=''):
    proba = predictor.predict_proba(row_features, model=model_name)
    class_names = list(proba.columns)
    class_names_str = [str(label) for label in class_names]

    if class_label:
        class_label = str(class_label)
        if class_label not in class_names_str:
            raise ValueError(f'class_label={class_label} not found in predictor classes: {class_names}')
        class_index = class_names_str.index(class_label)
    else:
        predicted_label = str(proba.iloc[0].idxmax())
        class_index = class_names_str.index(predicted_label)

    return class_index, class_names[class_index]


def build_prediction_function(predictor, feature_columns, model_name, class_index):
    def predict_fn(x):
        frame = pd.DataFrame(x, columns=feature_columns)
        if predictor.problem_type in ['binary', 'multiclass']:
            proba = predictor.predict_proba(frame, model=model_name)
            return proba.iloc[:, class_index].to_numpy()
        return predictor.predict(frame, model=model_name).to_numpy()

    return predict_fn


def build_shap_explainer(
    predictor,
    features,
    model_name,
    class_label='',
    background_size=10,
    class_row_features=None,
):
    model_name = choose_fast_model_name(predictor, model_name)
    _, shap = import_shap_plotting()

    background = shap.sample(features, min(background_size, len(features)), random_state=42)
    if predictor.problem_type in ['binary', 'multiclass']:
        class_index, explained_class = resolve_class_index(
            predictor,
            class_row_features if class_row_features is not None else features.iloc[[0]],
            model_name,
            class_label,
        )
    else:
        class_index, explained_class = None, None

    predict_fn = build_prediction_function(predictor, list(features.columns), model_name, class_index)
    explainer = shap.KernelExplainer(predict_fn, background, link='identity')
    return explainer, model_name, explained_class


def save_local_shap(
    predictor,
    features,
    output_dir,
    model_name='',
    row_index=0,
    class_label='',
    background_size=10,
    nsamples=64,
):
    plt, shap = import_shap_plotting()
    row_index = resolve_row_index(row_index, len(features))
    row_features = features.iloc[[row_index]].copy()
    explainer, model_name, explained_class = build_shap_explainer(
        predictor,
        features,
        model_name=model_name,
        class_label=class_label,
        background_size=background_size,
        class_row_features=row_features,
    )
    shap_result = explainer(row_features, nsamples=nsamples)

    shap_values = shap_result.values[0]
    base_value = shap_result.base_values[0]
    local_df = pd.DataFrame({
        'feature': row_features.columns,
        'feature_value': row_features.iloc[0].values,
        'shap_value': shap_values,
        'abs_shap_value': abs(shap_values),
    }).sort_values(by='abs_shap_value', ascending=False)
    local_df['row_index'] = row_index
    local_df['model_name'] = model_name
    local_df['base_value'] = base_value
    if explained_class is not None:
        local_df['explained_class'] = explained_class

    local_csv = output_dir / f'shap_row_{row_index}.csv'
    local_df.to_csv(local_csv, index=False)

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_result[0], max_display=min(20, len(features.columns)), show=False)
    waterfall_path = output_dir / f'shap_row_{row_index}_waterfall.png'
    plt.tight_layout()
    plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Saved local SHAP CSV: {local_csv}')
    print(f'Saved waterfall plot: {waterfall_path}')
    return local_df


def save_summary_plots(
    predictor,
    features,
    output_dir,
    model_name='',
    class_label='',
    background_size=10,
    nsamples=64,
    summary_sample_size=None,
    summary_plot_type='beeswarm',
):
    plt, shap = import_shap_plotting()
    if summary_sample_size is None:
        summary_features = features.copy()
    else:
        summary_features = shap.sample(features, min(summary_sample_size, len(features)), random_state=42)
    explainer, model_name, explained_class = build_shap_explainer(
        predictor,
        summary_features,
        model_name=model_name,
        class_label=class_label,
        background_size=background_size,
        class_row_features=summary_features.iloc[[0]],
    )
    shap_result = explainer(summary_features, nsamples=nsamples)

    summary_df = pd.DataFrame({
        'feature': list(features.columns),
        'mean_abs_shap_value': abs(shap_result.values).mean(axis=0),
    }).sort_values(by='mean_abs_shap_value', ascending=False)
    summary_df['model_name'] = model_name
    summary_df['n_rows_explained'] = len(summary_features)
    if explained_class is not None:
        summary_df['explained_class'] = explained_class

    summary_csv = output_dir / 'shap_summary_mean_abs.csv'
    summary_df.to_csv(summary_csv, index=False)

    if summary_plot_type not in ['bar', 'beeswarm']:
        raise ValueError("summary_plot_type must be 'bar' or 'beeswarm'.")

    plt.figure(figsize=(10, 8 if summary_plot_type == 'beeswarm' else 6))
    if summary_plot_type == 'bar':
        shap.plots.bar(shap_result, max_display=min(20, len(features.columns)), show=False)
        plot_path = output_dir / 'shap_summary_bar.png'
    else:
        shap.plots.beeswarm(shap_result, max_display=min(20, len(features.columns)), show=False)
        plot_path = output_dir / 'shap_summary_beeswarm.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Saved summary SHAP CSV: {summary_csv}')
    print(f'Saved summary plot: {plot_path}')
    return summary_df


if __name__ == '__main__':
    MODEL_PATH = '../data/ml_prediction/AutogluonModels/GOh1052_AggStabBind-mut/datafrac_1/fold_0'
    DATA_PATH = '../data/ml_prediction/Input/GOh1052_AggStabBind-mut_fold0_test.csv'
    OUTPUT_DIR = '../data/ml_prediction/Output/shap/'

    MODEL_NAME = ''
    ROW_INDEX = 0
    CLASS_LABEL = ''
    BACKGROUND_SIZE = 10
    NSAMPLES = 64

    SAVE_LOCAL_SHAP = False
    SAVE_SUMMARY_PLOTS = True
    SUMMARY_SAMPLE_SIZE = None
    SUMMARY_PLOT_TYPE = 'beeswarm'

    model_path = resolve_repo_path(MODEL_PATH)
    data_path = resolve_repo_path(DATA_PATH)
    output_dir = resolve_repo_path(OUTPUT_DIR)

    predictor = TabularPredictor.load(
        str(model_path),
        require_version_match=False,
        require_py_version_match=False,
    )
    data = pd.read_csv(data_path)
    model_data = select_model_data(data, predictor, include_label=(predictor.label in data.columns))
    features = model_data.drop(columns=[predictor.label]) if predictor.label in model_data.columns else model_data

    output_dir.mkdir(parents=True, exist_ok=True)
    shap_model_name = choose_fast_model_name(predictor, MODEL_NAME)

    print(f'Loaded predictor: {model_path}')
    print(f'Loaded data: {data_path}')
    print(f'Output dir: {output_dir}')
    print(f'Using model for SHAP: {shap_model_name}')

    if SAVE_LOCAL_SHAP:
        save_local_shap(
            predictor=predictor,
            features=features,
            output_dir=output_dir,
            model_name=shap_model_name,
            row_index=ROW_INDEX,
            class_label=CLASS_LABEL,
            background_size=BACKGROUND_SIZE,
            nsamples=NSAMPLES,
        )

    if SAVE_SUMMARY_PLOTS:
        save_summary_plots(
            predictor=predictor,
            features=features,
            output_dir=output_dir,
            model_name=shap_model_name,
            class_label=CLASS_LABEL,
            background_size=BACKGROUND_SIZE,
            nsamples=NSAMPLES,
            summary_sample_size=SUMMARY_SAMPLE_SIZE,
            summary_plot_type=SUMMARY_PLOT_TYPE,
        )
