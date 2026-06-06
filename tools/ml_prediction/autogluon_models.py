# -*- coding: utf-8 -*-
"""
AutoGluon helper utilities for classification evaluation.
"""

from __future__ import annotations

import numpy as np
from autogluon.core.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef


ag_mcc_scorer = make_scorer(
    name='mcc',
    score_func=matthews_corrcoef,
    optimum=1,
    greater_is_better=True,
)

eval_metric_dict = {
    'accuracy': 'accuracy',
    'balanced_accuracy': 'balanced_accuracy',
    'mcc': ag_mcc_scorer,
    'mae': 'mae',
}


def calculate_sklearn_metrics(y, ypred, eval_metric_list, class_labels=None):
    metric_vals = []
    if class_labels is None:
        class_labels = sorted(set(np.asarray(y)).union(set(np.asarray(ypred))))

    for eval_metric in eval_metric_list:
        if eval_metric == 'mcc':
            from sklearn.metrics import matthews_corrcoef

            metric_vals.append(matthews_corrcoef(y, ypred))
        elif eval_metric == 'accuracy':
            from sklearn.metrics import accuracy_score

            metric_vals.append(accuracy_score(y, ypred))
        elif eval_metric in [f'precision_{i}' for i in [0, 1, 2, 3, 4, 5]]:
            from sklearn.metrics import precision_score

            class_idx = int(eval_metric[-1])
            precision_per_class = precision_score(
                y,
                ypred,
                labels=class_labels,
                average=None,
                zero_division=0,
            )
            metric_vals.append(precision_per_class[class_idx] if class_idx < len(precision_per_class) else 0.0)
        elif eval_metric == 'precision':
            from sklearn.metrics import precision_score

            metric_vals.append(precision_score(y, ypred, average='micro'))
        elif eval_metric in [f'recall_{i}' for i in [0, 1, 2, 3, 4, 5]]:
            from sklearn.metrics import recall_score

            class_idx = int(eval_metric[-1])
            recall_per_class = recall_score(
                y,
                ypred,
                labels=class_labels,
                average=None,
                zero_division=0,
            )
            metric_vals.append(recall_per_class[class_idx] if class_idx < len(recall_per_class) else 0.0)
        elif eval_metric == 'recall':
            from sklearn.metrics import recall_score

            metric_vals.append(recall_score(y, ypred, average='micro'))
    return metric_vals


def get_leaderboard_metrics(data, label, predictor, leaderboard, metrics, train_or_test, has_label=True, class_labels=None):
    y_pred_dict = {}
    for metric in metrics:
        leaderboard[metric] = np.nan

    if has_label:
        y = data[label].to_numpy()

    for model in leaderboard.model.tolist():
        if train_or_test == 'test':
            if has_label and label in data.columns:
                feature_data = data.drop(columns=[label])
            else:
                feature_data = data
            y_pred_dict[model] = predictor.predict(feature_data, model=model)
        elif train_or_test == 'train':
            y_pred_dict[model] = predictor.predict_oof(
                train_data=data.drop(columns=[label]),
                model=model,
            )

        if has_label:
            metric_vals = calculate_sklearn_metrics(y, y_pred_dict[model], metrics, class_labels=class_labels)
            leaderboard.loc[(leaderboard.model == model), metrics] = metric_vals

    return leaderboard, y_pred_dict


def autogluon_classifier(
    train_data,
    test_data,
    label,
    metrics,
    save_model=None,
    load_model=None,
    model_settings=None,
    num_bag_folds=None,
    num_stack_levels=None,
    test_has_label=True,
    class_labels=None,
):
    from autogluon.tabular import TabularPredictor

    if model_settings is None:
        model_settings = {}

    res_cols = ['model'] + metrics
    excluded_model_types = model_settings.get('excluded_model_types', [])
    print('Excluded_model_types:', excluded_model_types)

    if load_model is not None:
        predictor = TabularPredictor.load(
            load_model,
            require_version_match=False,
            require_py_version_match=False,
        )
        print(f'Loaded previously trained model from {load_model}.')
        print('VALIDATION PERFORMANCE')
        train_res = predictor.fit_summary()
        train_leaderboard_filt = train_res['leaderboard']
    else:
        predictor = TabularPredictor(
            label=label,
            eval_metric=eval_metric_dict[metrics[0]],
            path=save_model,
        ).fit(
            train_data,
            num_bag_folds=num_bag_folds,
            num_stack_levels=num_stack_levels,
            excluded_model_types=excluded_model_types,
        )
        print('VALIDATION PERFORMANCE')
        train_res = predictor.fit_summary()
        train_leaderboard = train_res['leaderboard']
        train_leaderboard, _ = get_leaderboard_metrics(
            train_data,
            label,
            predictor,
            train_leaderboard[['model', 'score_val']].copy(),
            metrics,
            'train',
            class_labels=class_labels,
        )
        train_leaderboard_filt = train_leaderboard.copy()
        for model_to_exclude in excluded_model_types:
            train_leaderboard_filt = train_leaderboard_filt[
                ~train_leaderboard_filt.model.str.contains(model_to_exclude)
            ]

    best_model_name = train_leaderboard_filt.iloc[0]['model']
    print('Best model after filtering (val score):', best_model_name)

    y_pred = None
    y_pred_proba = None
    test_leaderboard_filt = None
    if test_data is not None:
        if test_has_label and label in test_data.columns:
            test_leaderboard, y_pred_test_dict = get_leaderboard_metrics(
                test_data,
                label,
                predictor,
                train_leaderboard_filt[['model', 'score_val']].copy(),
                metrics,
                'test',
                has_label=True,
                class_labels=class_labels,
            )
            feature_data = test_data.drop(columns=[label])
            y_pred = y_pred_test_dict[best_model_name].copy()
            y_pred_proba = predictor.predict_proba(feature_data, model=best_model_name)

            print('TEST PERFORMANCE')
            test_leaderboard_filt = test_leaderboard.copy()
            for model_to_exclude in excluded_model_types:
                test_leaderboard_filt = test_leaderboard_filt[
                    ~test_leaderboard_filt.model.str.contains(model_to_exclude)
                ]
            test_leaderboard_filt = test_leaderboard_filt.sort_values(by='score_val', ascending=False)
            print(test_leaderboard_filt[[c for c in res_cols if c in test_leaderboard_filt] + ['score_val']])
        else:
            print('INFERENCE ONLY')
            y_pred = predictor.predict(test_data, model=best_model_name)
            y_pred_proba = predictor.predict_proba(test_data, model=best_model_name)

    res = {
        'train': train_leaderboard_filt[[c for c in res_cols if c in train_leaderboard_filt]],
        'test': (
            test_leaderboard_filt[[c for c in res_cols if c in test_leaderboard_filt]]
            if test_data is not None and test_leaderboard_filt is not None
            else None
        ),
    }
    return predictor, y_pred, y_pred_proba, res
