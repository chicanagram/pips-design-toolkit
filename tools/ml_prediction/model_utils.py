# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 09:32:02 2022
"""

import os, shutil
import random
import pandas as pd
import numpy as np
def get_split_metrics_summary(split_metrics, eval_metric_list, save_results=None):
    if isinstance(eval_metric_list, str):
        eval_metric_list = [eval_metric_list]
    split_metrics_summary = []
    for train_or_test in ['train', 'test']:
        metrics_dict = {'train_or_test': train_or_test}
        metrics_filt = split_metrics[(split_metrics.train_or_test == train_or_test)]
        for eval_metric in eval_metric_list:
            metrics_filt_avg = metrics_filt[[eval_metric]].rename(columns={eval_metric: f'{eval_metric}_avg'}).mean(
                axis=0).round(3).to_dict()
            metrics_filt_std = metrics_filt[[eval_metric]].rename(columns={eval_metric: f'{eval_metric}_std'}).std(
                axis=0).round(3).to_dict()
            metrics_dict.update(metrics_filt_avg)
            metrics_dict.update(metrics_filt_std)
        split_metrics_summary.append(metrics_dict)
    split_metrics_summary = pd.DataFrame(split_metrics_summary)
    if save_results is not None:
        split_metrics_summary.to_csv(f'{save_results}split_metrics_summary.csv')
    return split_metrics_summary

def plot_model_metrics_splits(split_metrics, split_by_col='split_label', eval_metric='mcc', eval_metric_suffix='', barwidth=0.4, figtitle=None, savefig=None, annotate=False):
    import matplotlib.pyplot as plt
    split_label_list = list(set(split_metrics[split_by_col].tolist()))
    split_label_list.sort()
    fig, ax = plt.subplots(figsize=(len(split_label_list)*3, 6))
    for i, split_label in enumerate(split_label_list):
        for j, train_or_test in enumerate(['train', 'test']):
            c = 'b' if train_or_test=='train' else 'orange'
            metrics_filt = split_metrics[(split_metrics[split_by_col]==split_label) & (split_metrics.train_or_test==train_or_test)].iloc[0].to_dict()
            metric_avg = metrics_filt[eval_metric+eval_metric_suffix]
            x = np.array([i+(-1)**(j+1)*barwidth/2])
            y = np.array([metric_avg])
            ax.bar(x, y, width=barwidth, color=c)
            if annotate:
                ax.annotate(round(metric_avg,2), (x,y+0.02), fontsize=10)

            if eval_metric+'_std' in metrics_filt:
                metric_std = metrics_filt[eval_metric+'_std']
                ax.errorbar(x, y, yerr=np.array([metric_std]), color='k', fmt='o',
                            markersize=8, capsize=6, label=None)
    # set labels and axes
    ax.set_ylim([0, 1])
    ax.set_ylabel(eval_metric, fontsize=20)
    ax.set_xticks(np.arange(len(split_label_list)))
    ax.set_xticklabels(split_label_list, rotation=45, fontsize=12)
    if figtitle is not None:
        plt.suptitle(figtitle, fontsize=18)
    if savefig is not None:
        fig.savefig(savefig, bbox_inches='tight')
    plt.show()

def plot_model_metrics_summary(split_metrics_summary, eval_metric='mcc', figsize=(8,6), barwidth=0.8, figtitle=None, savefig=None, plot_errors=True, annotate=False):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    for i, train_or_test in enumerate(['train', 'test']):
        c = 'b' if train_or_test == 'train' else 'orange'
        metrics_filt = split_metrics_summary[(split_metrics_summary.train_or_test==train_or_test)].iloc[0].to_dict()
        metric_avg = metrics_filt[f'{eval_metric}_avg']
        metric_std = metrics_filt[f'{eval_metric}_std']
        x = np.array([i])
        y = np.array([metric_avg])
        ax.bar(x, y, width=barwidth, color=c)
        if annotate:
            ax.annotate(round(metric_avg, 2), (x,y+0.02), fontsize=10)
        if plot_errors:
            ax.errorbar(x, y, yerr=np.array([metric_std]), color='k', fmt='o', markersize=8, capsize=10, label=None)
    # set labels and axes
    ax.set_ylim([0, 1])
    ax.set_ylabel(eval_metric, fontsize=20)
    ax.set_xticks(np.array([0,1]))
    ax.set_xticklabels(['train', 'test'], fontsize=12)
    if figtitle is not None:
        plt.suptitle(figtitle, fontsize=18)
    if savefig is not None:
        fig.savefig(savefig, bbox_inches='tight')
    plt.show()