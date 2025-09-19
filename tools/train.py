#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 07:58:44 2024
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def parseArgs():
    """Parse command line arguments"""
    import argparse
    import traceback
    try:
        parser = argparse.ArgumentParser()
        # dataset_fbase
        parser.add_argument('-f', default='GOh1052_sift-filt', help='Base dataset name found in csv filename, also to be used as the folder name for the trained model, e.g. "GOh1052"')
        # data_folder
        parser.add_argument('--data_folder', default='./data/ml_prediction/', help='Relative path of directory in which dataset Input subdirectory is found, e.g. "./examples/"')
        # num_bag_folds
        parser.add_argument('--num_bag_folds', default=8, type=int, help='Number of folds used for model bagging')
        # num_stack_levels
        parser.add_argument('--num_stack_levels', default=1, type=int, help='Number of levels of model stacking')
        # filt_by
        parser.add_argument('--filt_by', default='', type=str, help='Filter by SIFT, ShanMS, or distance score')
        # filt_shanms
        parser.add_argument('--filt_shanms', default='0.5,1.5', type=str, help='Filter by ShanMS score')
        # filt_sift
        parser.add_argument('--filt_sift', default='0.1,0.45', type=str, help='Filter by SIFTnorm score')
        # filt_dist
        parser.add_argument('--filt_dist', default='20,50', type=str, help='Filter by distance to substrate')
        # save_model
        parser.add_argument('-save_model', default=True, type=bool, help='Folder name for saving the trained model')
        return parser.parse_args()
    except:
        print("An exception occurred with argument parsing. Check your provided options.")
        traceback.print_exc()

def main():

    import pandas as pd
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    from autogluon.tabular import TabularDataset
    from ml_prediction.model_variables import featuresets
    from ml_prediction.autogluon_models import autogluon_classifier
    from data_reduction.filter_data import filter_by_score

    # get inputs from commandline
    args = parseArgs()
    data_folder = args.data_folder
    dataset_fbase = args.f
    num_bag_folds = args.num_bag_folds
    num_stack_levels = args.num_stack_levels
    filt_by = args.filt_by

    # other settings
    load_model = None # set to model directory if loading a pre-trained model
    y_feature = 'CategoryV3'
    x_features = featuresets['mut']
    eval_metric_list = ['mcc', 'accuracy', 'recall_0', 'recall_1', 'recall_2', 'precision_0', 'precision_1', 'precision_2']

    # get data
    dataset_fpath_all = f'{data_folder}Input/{dataset_fbase}_train.csv'
    if filt_by=='':
        XY_train = pd.read_csv(dataset_fpath_all)[x_features + [y_feature]]
    else:
        XY_train = pd.read_csv(dataset_fpath_all)[x_features + [y_feature, 'Position']]

    # update fbase and save names with filter condition
    if filt_by!='':
        dataset_fbase_0 = dataset_fbase
        dataset_fbase = dataset_fbase + '_' + filt_by + '-filt'
    save_res = f'{data_folder}Output/{dataset_fbase}/{dataset_fbase}'
    if args.save_model:
        save_model = f'{data_folder}trained_models/{dataset_fbase}/'
    else:
        save_model = None

    # filter train data
    if filt_by!='':
        if filt_by == 'sift':
            filt_thres = [float(v) for v in args.filt_sift.split(',')]
        elif filt_by == 'shanMS':
            filt_thres = [float(v) for v in args.filt_shanms.split(',')]
        elif filt_by == 'distance':
            filt_thres = [float(v) for v in args.filt_dist.split(',')]
        XY_train, _ = filter_by_score(XY_train, filt_by, filt_thres, dataset_fbase_0)

    # convert to autogluon tabular dataset
    train_data = TabularDataset(XY_train)

   # train and validate model #
    predictor, y_pred, res = autogluon_classifier(
        train_data, None, label=y_feature, metrics=eval_metric_list,
        save_model=save_model, load_model=load_model,
        num_bag_folds=num_bag_folds, num_stack_levels=num_stack_levels
    )


if __name__ == "__main__":
    main()