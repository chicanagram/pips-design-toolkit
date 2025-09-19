#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 07:58:44 2024
"""
import warnings
import pandas as pd
warnings.simplefilter(action='ignore', category=FutureWarning)

def parseArgs():
    from ml_prediction.model_variables import featuresets
    """Parse command line arguments"""
    import argparse
    import traceback
    try:
        parser = argparse.ArgumentParser()
        # dataset_fbase
        parser.add_argument('-f', default='GOh1052', help='Base dataset name found in csv filename, also to be used as the folder name for the trained model, e.g. "GOh1052"')
        # data_folder
        parser.add_argument('--data_folder', default='./data/ml_prediction/', help='Relative path of directory in which dataset Input subdirectory is found, e.g. "./examples/"')
        # n_splits
        parser.add_argument('--n_splits', default=4, type=int, help='Number of splits for k-fold testing')
        # test_frac
        parser.add_argument('--test_frac', default=0.25, type=float, help='Fraction of data in test set')
        # num_bag_folds
        parser.add_argument('--num_bag_folds', default=8, type=int, help='Number of folds used for model bagging')
        # num_stack_levels
        parser.add_argument('--num_stack_levels', default=1, type=int, help='Number of levels of model stacking')
        # filt_by
        parser.add_argument('--filt_by', default='shanMS', type=str, help='Filter by SIFT, ShanMS, or distance score')
        # filt_shanms
        parser.add_argument('--filt_shanms', default='0.5,1.5', type=str, help='Filter by ShanMS score')
        # filt_sift
        parser.add_argument('--filt_sift', default='0.1,0.45', type=str, help='Filter by SIFTnorm score')
        # filt_dist
        parser.add_argument('--filt_dist', default='20,50', type=str, help='Filter by distance to substrate')
        # save_model
        parser.add_argument('-save_model', default=False, type=bool, help='Folder name for saving the trained model')
        return parser.parse_args()
    except:
        print("An exception occurred with argument parsing. Check your provided options.")
        traceback.print_exc()

def main():

    import pandas as pd
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    from sklearn.model_selection import train_test_split
    from autogluon.tabular import TabularDataset
    from ml_prediction.model_variables import featuresets
    from ml_prediction.model_utils import get_split_metrics_summary, plot_model_metrics_splits, plot_model_metrics_summary
    from ml_prediction.autogluon_models import autogluon_classifier
    from data_reduction.filter_data import filter_by_score

    # get inputs from commandline
    args = parseArgs()
    data_folder = args.data_folder
    dataset_fbase = args.f
    n_splits = args.n_splits
    test_frac = args.test_frac
    num_bag_folds = args.num_bag_folds
    num_stack_levels = args.num_stack_levels
    filt_by = args.filt_by

    # other settings
    load_model = None # set to model directory if loading a pre-trained model
    y_feature = 'CategoryV3'
    x_features = featuresets['mut']
    eval_metric_list = ['mcc', 'accuracy', 'recall_0', 'recall_1', 'recall_2', 'precision_0', 'precision_1', 'precision_2']
    res_cols = ['model', 'train_or_test', 'split_label'] + eval_metric_list
    ensemble_metrics = []
    split_metrics = []

    # get data
    dataset_fpath_all = f'{data_folder}Input/{dataset_fbase}.csv'
    if filt_by=='':
        XY_data = pd.read_csv(dataset_fpath_all)[x_features + [y_feature]]
    else:
        XY_data = pd.read_csv(dataset_fpath_all)[x_features + [y_feature, 'Position']]

    # update fbase and save names with filter condition
    if filt_by!='':
        dataset_fbase_0 = dataset_fbase
        dataset_fbase = dataset_fbase + '_' + filt_by + '-filt'
    save_res = f'{data_folder}Output/{dataset_fbase}/{dataset_fbase}'
    if args.save_model:
        save_model = f'{data_folder}trained_models/{dataset_fbase}/'
    else:
        save_model = None

    # iterate through splits and load each dataset
    for i in range(n_splits):
        print(f'Split {i}')
        # get random split
        if test_frac > 0:
            XY_train, XY_test = train_test_split(XY_data, test_size=test_frac, random_state=i)
        else:
            XY_train = XY_data
            XY_test = None

        # filter train data
        if filt_by!='':
            if filt_by == 'sift':
                filt_thres = [float(v) for v in args.filt_sift.split(',')]
            elif filt_by == 'shanMS':
                filt_thres = [float(v) for v in args.filt_shanms.split(',')]
            elif filt_by == 'distance':
                filt_thres = [float(v) for v in args.filt_dist.split(',')]
            XY_train, _ = filter_by_score(XY_train, filt_by, filt_thres, dataset_fbase_0)
            # XY_test = pd.concat([XY_test,XY_test_append], axis=0)

        # convert to autogluon tabular dataset
        print('Train size:', len(XY_train))
        print('Test size:', len(XY_test))
        train_data = TabularDataset(XY_train)
        test_data = TabularDataset(XY_test) if XY_test is not None else None

       # train and validate model #
        predictor, y_pred, res = autogluon_classifier(
            train_data, test_data, label=y_feature, metrics=eval_metric_list,
            save_model=save_model, load_model=load_model,
            num_bag_folds=num_bag_folds, num_stack_levels=num_stack_levels
        )
        # update overall results
        for train_or_test in ['train', 'test']:
            if train_or_test in res:
                leaderboard = res[train_or_test]
                leaderboard['split_label'] = i
                leaderboard['train_or_test'] = train_or_test
                split_metrics_selected = leaderboard.iloc[0].to_dict()
                split_metrics.append(split_metrics_selected)
                ensemble_metrics += leaderboard.to_dict('records')

    # save data
    ensemble_metrics = pd.DataFrame(ensemble_metrics)[res_cols]
    split_metrics = pd.DataFrame(split_metrics)[res_cols]
    split_metrics_summary = get_split_metrics_summary(split_metrics, eval_metric_list=eval_metric_list, save_results=None)
    ensemble_metrics.to_csv(f'{save_res}_ensemble_metrics.csv')
    split_metrics.to_csv(f'{save_res}_split_metrics.csv')
    split_metrics_summary.to_csv(f'{save_res}_split_metrics_summary.csv')

    return ensemble_metrics, split_metrics, split_metrics_summary

if __name__ == "__main__":
    ensemble_metrics, split_metrics, split_metrics_summary = main()