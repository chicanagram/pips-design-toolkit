from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def perform_mean_std_scaling(xtrain, xtest=None):
    mean = np.mean(xtrain, axis=0)
    std = np.std(xtrain, axis=0)
    xtrain_scaled = (xtrain - mean) / std
    if xtest is not None:
        xtest_scaled = (xtest - mean) / std
    else:
        xtest_scaled = None
    return mean, std, xtrain_scaled, xtest_scaled


def get_regressor_scoring(y_pred, y_true, model_name=None):
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_squared_error, r2_score

    r2 = r2_score(y_true, y_pred)
    corr_s = spearmanr(y_true, y_pred)[0]
    corr_p = pearsonr(y_true, y_pred)[0]
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)

    return {
        "Model": model_name,
        "r2": r2,
        "SpearmanR": corr_s,
        "PearsonR": corr_p,
        "mae": mae,
        "rmse": rmse,
    }


def get_score(y, ypred, scoring):
    if scoring == 'mae':
        return mean_absolute_error(y, ypred)
    if scoring == 'rmse':
        return root_mean_squared_error(y, ypred)
    raise ValueError(f'Unsupported scoring metric: {scoring}')


def get_classification_metrics(y_pred, y_true, model_name=None, class_labels=None, average_method='macro'):
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score

    if class_labels is None:
        class_labels = [-1, 0, 1]

    metric_kwargs = {
        'labels': class_labels,
        'average': average_method,
        'zero_division': 0,
    }
    precision_per_class = precision_score(y_true, y_pred, labels=class_labels, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, labels=class_labels, average=None, zero_division=0)
    metrics = {
        'Model': model_name,
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred, **metric_kwargs),
        'Precision': precision_score(y_true, y_pred, **metric_kwargs),
        'Recall': recall_score(y_true, y_pred, **metric_kwargs),
    }
    for label, precision_val, recall_val in zip(class_labels, precision_per_class, recall_per_class):
        metrics[f'Precision_{label}'] = precision_val
        metrics[f'Recall_{label}'] = recall_val
    return metrics


def get_classification_bootstrap_ci(
    y_true,
    y_pred,
    metrics_list=('MCC', 'Accuracy', 'F1', 'Precision', 'Recall'),
    class_labels=(-1, 0, 1),
    average_method='macro',
    n_boot=1000,
    ci=0.95,
    random_state=42,
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid_mask = ~pd.isna(y_true) & ~pd.isna(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        metrics_ci = {}
        for metric in metrics_list:
            metrics_ci[metric] = np.nan
            metrics_ci[f'{metric}_ci_low'] = np.nan
            metrics_ci[f'{metric}_ci_high'] = np.nan
        return metrics_ci

    rng = np.random.default_rng(random_state)
    point_estimates = get_classification_metrics(
        y_pred,
        y_true,
        model_name=None,
        class_labels=list(class_labels),
        average_method=average_method,
    )

    alpha = (1 - ci) / 2
    boot_metrics = {metric: [] for metric in metrics_list}
    sample_size = len(y_true)

    for _ in range(n_boot):
        idx = rng.integers(0, sample_size, size=sample_size)
        y_true_boot = y_true[idx]
        y_pred_boot = y_pred[idx]
        metrics_boot = get_classification_metrics(
            y_pred_boot,
            y_true_boot,
            model_name=None,
            class_labels=list(class_labels),
            average_method=average_method,
        )
        for metric in metrics_list:
            boot_metrics[metric].append(metrics_boot[metric])

    metrics_ci = {}
    for metric in metrics_list:
        values = np.asarray(boot_metrics[metric], dtype=float)
        metrics_ci[metric] = point_estimates[metric]
        metrics_ci[f'{metric}_ci_low'] = np.quantile(values, alpha)
        metrics_ci[f'{metric}_ci_high'] = np.quantile(values, 1 - alpha)
    return metrics_ci
