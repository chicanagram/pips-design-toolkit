import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .metrics import perform_mean_std_scaling, get_regressor_scoring, get_classification_metrics
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def sklearn_classifier(X_train, y_train, X_test, y_test, model_dict, class_labels=[-1,0,1], print_res=False, scale_data=True, multiclass_average_method='macro'):

    # get scaled data
    if scale_data:
        mean, std, X_train_, X_test_ = perform_mean_std_scaling(X_train, X_test)
    else:
        X_train_ = X_train
        X_test_ = X_test

    # retrain model if not provided
    model_type = model_dict['model_type']
    yoffset = 0
    if model_type == 'pls':
        from sklearn.linear_model import PLSRegression
        param_name = 'n_components'
        param_val = 100 # model_dict[param_name]
        model = PLSRegression(n_components=param_val)
    elif model_type == 'ridge':
        from sklearn.linear_model import RidgeClassifier
        param_name = 'alpha'
        param_val = 1 # model_dict[param_name]
        model = RidgeClassifier(alpha=param_val)
    elif model_type == 'lasso':
        from sklearn.linear_model import LogisticRegression
        param_name = 'penalty'
        param_val = 'l1' # model_dict[param_name]
        model = LogisticRegression(max_iter=50000, penalty='l1', solver='liblinear')
    elif model_type == 'randomforest':
        from sklearn.ensemble import RandomForestClassifier
        param_name = 'n_estimators'
        param_val = 100 # model_dict[param_name]
        model = RandomForestClassifier(n_estimators=param_val, random_state=0)
    elif model_type == 'xgb':
        from xgboost import XGBClassifier
        param_name = 'n_estimators'
        param_val = 100 # model_dict[param_name]
        model = XGBClassifier(objective="'multi:softprob'", n_estimators=param_val, random_state=0)
        yoffset = 1

    # get train results
    model.fit(X_train_, y_train+yoffset)
    ypred_train = model.predict(X_train_) - yoffset
    metrics_train = get_classification_metrics(ypred_train, y_train, model_name=model_type, class_labels=class_labels, average_method=multiclass_average_method)
    metrics_train.update({'train_or_test':'train'})

    # perform evaluation on test data
    ypred_test = model.predict(X_test_) - yoffset
    metrics_test = get_classification_metrics(ypred_test, y_test, model_name=model_type, class_labels=class_labels, average_method=multiclass_average_method)
    metrics_test.update({'train_or_test': 'test'})
    metrics = [metrics_train, metrics_test]

    if print_res:
        print(pd.DataFrame(metrics))

    return metrics, model, ypred_test


def sklearn_regressor(X_train, y_train, X_test, y_test, model_dict, print_res=False, scale_data=True):
    # get scaled data
    if scale_data:
        mean, std, X_train_, X_test_ = perform_mean_std_scaling(X_train, X_test)
    else:
        X_train_ = X_train
        X_test_ = X_test

    # retrain model if not provided
    model_type = model_dict['model_type']

    if model_type == 'plsr':
        from sklearn.cross_decomposition import PLSRegression
        n_components = min(20, X_train.shape[1])
        model = PLSRegression(n_components=n_components)
    elif model_type == 'lasso':
        from sklearn.linear_model import Lasso
        max_iter = 10000
        alpha = 100
        model = Lasso(max_iter=max_iter, alpha=alpha)
    elif model_type == 'randomforest':
        from sklearn.ensemble import RandomForestRegressor
        n_estimators = 100
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
    elif model_type == 'xgb':
        from xgboost import XGBRegressor
        n_estimators = 100
        model = XGBRegressor(n_estimators=n_estimators, random_state=0)

    # get train results
    model.fit(X_train_, y_train)
    ypred_train = model.predict(X_train_)
    metrics_train = get_regressor_scoring(ypred_train, y_train, model_name=model_type)
    metrics_train.update({'train_or_test':'train'})

    # perform evaluation on test data
    ypred_test = model.predict(X_test_)
    metrics_test = get_regressor_scoring(ypred_test, y_test, model_name=model_type)
    metrics_test.update({'train_or_test': 'test'})
    metrics = [metrics_train, metrics_test]

    if print_res:
        print(pd.DataFrame(metrics))
    return metrics, model, ypred_test

def remove_nan_inf_data(arr, print_res=None):
    size_pre = arr.shape[0]
    valid_mask = ~np.any(np.isinf(arr) | np.isnan(arr), axis=1)
    valid_row_indices = np.where(valid_mask)[0]
    size_post = len(valid_row_indices)
    arr = arr[valid_row_indices,:]
    if print_res is not None:
        print(f'{print_res} size before/after NaN/Inf removal: {size_pre} >> {size_post}')
    return arr, valid_row_indices

def fit_sklearn_classifier_kfold(X, y, model_dict, class_labels, n_splits=None, scale_data=True, split_idxs_list=None, data_frac=1, multiclass_average_method='macro', return_fold_ids=False):
    # get split indices based on random kfold splits, if not provided
    if split_idxs_list is None:
        kFold = KFold(n_splits=n_splits, shuffle=False)
        split_idxs_list = kFold.split(X)
        print(f'Obtained {n_splits} random splits.')

    # initialize results storage
    n = len(y)
    ypred = np.zeros((n,))
    ypred[:] = np.nan
    fold_ids = np.zeros((n,))
    fold_ids[:] = np.nan
    metrics_kfold = []

    # iterate through all splits
    for i, (train_index, test_index) in enumerate(split_idxs_list):
        if data_frac < 1:
            train_index = train_index[:max(1, int(len(train_index) * data_frac))]
        # get train and test datasets
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        # remove inf data
        X_train, idx_keep_train = remove_nan_inf_data(X_train, print_res='train')
        X_test, idx_keep_test = remove_nan_inf_data(X_test, print_res='test')
        y_train = y_train[idx_keep_train]
        y_test = y_test[idx_keep_test]
        print(f'Split {i+1}/{n_splits}')
        metrics, _, ypred_test = sklearn_classifier(X_train, y_train, X_test, y_test, model_dict, class_labels, print_res=True, scale_data=scale_data, multiclass_average_method=multiclass_average_method)
        metrics_kfold += metrics
        ypred[[test_index[i] for i in idx_keep_test]] = ypred_test
        fold_ids[[test_index[i] for i in idx_keep_test]] = i
    if return_fold_ids:
        return metrics_kfold, ypred, fold_ids
    return metrics_kfold, ypred


def fit_sklearn_regressor_kfold(X, y, model_dict, n_splits=None, scale_data=True, split_idxs_list=None, plot_scatter=False):
    # get split indices based on random kfold splits, if not provided
    if split_idxs_list is None:
        kFold = KFold(n_splits=n_splits, shuffle=False)
        split_idxs_list = kFold.split(X)
        print(f'Obtained {n_splits} random splits.')

    # initialize results storage
    n = len(y)
    ypred = np.zeros((n,))
    ypred[:] = np.nan
    metrics_kfold = []

    # iterate through all splits
    for i, (train_index, test_index) in enumerate(split_idxs_list):
        # get train and test datasets
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        # remove inf data
        X_train, idx_keep_train = remove_nan_inf_data(X_train, print_res='train')
        X_test, idx_keep_test = remove_nan_inf_data(X_test, print_res='test')
        y_train = y_train[idx_keep_train]
        y_test = y_test[idx_keep_test]
        print(f'Split {i+1}/{n_splits}')
        metrics, _, ypred_test = sklearn_regressor(X_train, y_train, X_test, y_test, model_dict, print_res=True, scale_data=scale_data)
        metrics_kfold += metrics
        ypred[[test_index[i] for i in idx_keep_test]] = ypred_test

    if plot_scatter:
        # Fit best-fit line: y_pred as a function of y
        bestfit = LinearRegression().fit(y.reshape(-1, 1), ypred)
        yp_fit = bestfit.predict(y.reshape(-1, 1))
        overall_r2 = round(r2_score(y,ypred),3)
        print('Overall R2:', overall_r2, '; n =', len(y))
        print(', '.join([str(i) for i in y]))
        print(', '.join([str(i) for i in ypred]))
        plt.scatter(y, ypred)
        plt.plot(y, yp_fit, color='red', linewidth=0.5)
        plt.xlabel('Y actual')
        plt.ylabel('Y predicted')
        plt.legend([f'n={len(y)} data points', 'Trendline: R2='+str(overall_r2)+'\nPearson rho='+str(round(np.sqrt(overall_r2),2))], frameon=False)
        plt.title('Predicted vs Actual Output')
        plt.show()
    return metrics_kfold, ypred
