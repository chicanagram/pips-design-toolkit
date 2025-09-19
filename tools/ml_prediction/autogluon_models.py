# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 09:32:02 2022

@author: Jhoann
"""

import numpy as np
from sklearn.metrics import matthews_corrcoef
from autogluon.core.metrics import make_scorer

ag_mcc_scorer = make_scorer(name='mcc', score_func=matthews_corrcoef, optimum=1, greater_is_better=True)
eval_metric_dict = {'accuracy':'accuracy', 'balanced_accuracy':'balanced_accuracy', 'mcc':ag_mcc_scorer, 'mae':'mae'}
def calculate_sklearn_metrics(y, ypred, eval_metric_list):
    metric_vals = []
    for eval_metric in eval_metric_list:
        if eval_metric=='mcc':
            from sklearn.metrics import matthews_corrcoef
            metric_vals.append(matthews_corrcoef(y, ypred))
        elif eval_metric=='accuracy':
            from sklearn.metrics import accuracy_score
            metric_vals.append(accuracy_score(y, ypred))
        elif eval_metric in [f'precision_{i}' for i in [0,1,2,3,4,5]]:
            from sklearn.metrics import precision_score
            class_idx = int(eval_metric[-1])
            precision_per_class = precision_score(y, ypred, average=None)
            metric_vals.append(precision_per_class[class_idx])
        elif eval_metric == 'precision':
            from sklearn.metrics import precision_score
            metric_vals.append(precision_score(y, ypred, average='micro'))
        elif eval_metric in [f'recall_{i}' for i in [0, 1, 2, 3, 4, 5]]:
            from sklearn.metrics import recall_score
            class_idx = int(eval_metric[-1])
            recall_per_class = recall_score(y, ypred, average=None)
            metric_vals.append(recall_per_class[class_idx])
        elif eval_metric == 'recall':
            from sklearn.metrics import recall_score
            metric_vals.append(recall_score(y, ypred, average='micro'))
    return metric_vals
def get_leaderboard_metrics(data, label, predictor, leaderboard, metrics, train_or_test):
    y_pred_dict = {}
    y = data[label].to_numpy()
    for metric in metrics:
        leaderboard[metric] = np.nan
    # evaluate test performance for each model and each metric
    for model in leaderboard.model.tolist():
        if train_or_test=='test':
            y_pred_dict[model] = predictor.predict(data.drop(columns=[label]), model=model)
        elif train_or_test=='train':
            y_pred_dict[model] = predictor.get_oof_pred(train_data=data.drop(columns=[label]), model=model)
        # calculate metrics
        metric_vals = calculate_sklearn_metrics(y, y_pred_dict[model], metrics)
        leaderboard.loc[(leaderboard.model == model), metrics] = metric_vals
    return leaderboard, y_pred_dict

def autogluon_classifier(train_data, test_data, label, metrics, save_model=None, load_model=None, model_settings={}, num_bag_folds=None, num_stack_levels=None):
    from autogluon.tabular import TabularPredictor

    res_cols = ['model'] + metrics

    if 'excluded_model_types' in model_settings:
        excluded_model_types = model_settings['excluded_model_types']
    else:
        excluded_model_types = []
    print('Excluded_model_types:', excluded_model_types)

    # train #
    if load_model is not None:
        predictor = TabularPredictor.load(load_model, require_version_match=False, require_py_version_match=False)
        print(f'Loaded previously trained model from {load_model}.')

        print('VALIDATION PERFORMANCE')
        train_res = predictor.fit_summary()
        train_leaderboard_filt = train_res['leaderboard']
    else:
        predictor = TabularPredictor(label=label, eval_metric=eval_metric_dict[metrics[0]], path=save_model).fit(
            train_data,
            num_bag_folds=num_bag_folds,
            num_stack_levels=num_stack_levels,
            excluded_model_types=excluded_model_types
        )
        print('VALIDATION PERFORMANCE')
        train_res = predictor.fit_summary()
        train_leaderboard = train_res['leaderboard']

        # get train leaderboard with all metrics
        train_leaderboard, _ = get_leaderboard_metrics(train_data, label, predictor, train_leaderboard[['model', 'score_val']].copy(), metrics, 'train')
        train_leaderboard_filt = train_leaderboard.copy()
        for model_to_exclude in excluded_model_types:
            train_leaderboard_filt = train_leaderboard_filt[~train_leaderboard_filt.model.str.contains(model_to_exclude)]

    best_model_name = train_leaderboard_filt.iloc[0]['model']
    print('Best model after filtering (val score):', best_model_name)

    # test #
    y_pred = None
    if test_data is not None:
        # get test leaderboard with all metrics
        test_leaderboard, y_pred_test_dict = get_leaderboard_metrics(test_data, label, predictor, train_leaderboard_filt[['model', 'score_val']].copy(), metrics, 'test')

        # get test predictions for best model
        y_pred = y_pred_test_dict[best_model_name].copy()
        y_pred.head()

        # evaluate using autogluon
        print('TEST PERFORMANCE')
        # filter out models to exclude
        test_leaderboard_filt = test_leaderboard.copy()
        for model_to_exclude in excluded_model_types:
            test_leaderboard_filt = test_leaderboard_filt[~test_leaderboard_filt.model.str.contains(model_to_exclude)]
        # order test leaderboard by val score
        test_leaderboard_filt = test_leaderboard_filt.sort_values(by='score_val', ascending=False)
        print(test_leaderboard_filt[[c for c in res_cols if c in test_leaderboard_filt]+['score_val']])

    res = {
        'train': train_leaderboard_filt[[c for c in res_cols if c in train_leaderboard_filt]],
        'test': test_leaderboard_filt[[c for c in res_cols if c in test_leaderboard_filt]] if test_data is not None else None
        }
    return predictor, y_pred, res


def load_autogluon_tutorial_datasets(tutorial_name, y_feature):

    from autogluon.tabular import TabularDataset

    if tutorial_name=='quickstart':
        data_url = 'https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/'
        train_data = TabularDataset(f'{data_url}train.csv')
        train_data.head()
        train_data[y_feature].describe()
        test_data = TabularDataset(f'{data_url}test.csv')
        return train_data, test_data

    elif tutorial_name=='in-depth':
        train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
        subsample_size = 500  # subsample subset of data for faster demo, try setting this to much larger values
        train_data = train_data.sample(n=subsample_size, random_state=0)
        print(train_data.head())
        print(f"Summary of {y_feature} column: \n", train_data[y_feature].describe())
        new_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
        test_data = new_data[5000:].copy()  # this should be separate data in your applications
        test_data_label = test_data[y_feature]
        test_data_nolabel = test_data.drop(columns=[y_feature])  # delete label column
        val_data = new_data[:5000].copy()

        return train_data, test_data_label, test_data_nolabel, val_data



