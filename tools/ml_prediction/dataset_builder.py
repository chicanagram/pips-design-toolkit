#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared dataset assembly helpers for sklearn and AutoGluon pipelines.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools.utils.variables import subfolders
from tools.ml_prediction.model_features import (
    feature_names,
    feature_names_multimodal,
    get_featureset_dir,
)
from tools.ml_prediction.datacleaning_utils import (
    deduplicate_data_average_labels,
    load_data_allsuffixes,
)


ARRAY_FEATURE_FORMATS = {
    'ohe': '.npz',
    'oheMT': '.npz',
    'georgiev': '.npy',
    'georgievMT': '.npy',
}


def as_path(path_like) -> Path:
    return Path(path_like).expanduser().resolve()


def infer_dataset_fbase(fname_prefix, labels_actual_fname):
    if fname_prefix:
        return fname_prefix[:-1] if fname_prefix.endswith('_') else fname_prefix
    return labels_actual_fname.split('_')[0]


def ensure_name_column(df):
    if 'name' in df.columns:
        return df
    df = df.copy()
    if 'protein_name' in df.columns and 'mutations' in df.columns:
        df['name'] = [f'{prot}_{mut}' for prot, mut in zip(df['protein_name'].tolist(), df['mutations'].tolist())]
        return df
    if 'mutations' in df.columns:
        df['name'] = df['mutations'].astype(str)
        return df
    raise ValueError("Unable to construct a 'name' column from the available label columns.")


def validate_feature_merge(df, feature_list, component_featureset, expected_size):
    if len(df) == 0 or not feature_list:
        return
    feature_cols_present = [c for c in feature_list if c in df.columns]
    if not feature_cols_present:
        raise ValueError(f'No feature columns found after merging features for {component_featureset}.')
    valid_rows = int(df[feature_cols_present].notna().all(axis=1).sum())
    if expected_size > 0 and valid_rows < 0.9 * expected_size:
        raise ValueError(
            f'Feature merge for {component_featureset} retained only {valid_rows}/{expected_size} rows. '
            f'This usually means the merge key is mismatched between the labels file and feature file.'
        )


def _candidate_feature_bases(component_featureset, fname_prefix, data_subfolder=''):
    fname_base = component_featureset
    if 'LLRsum' in fname_base and 'entropy' not in fname_base:
        fname_base = fname_base.replace('_LLRsum', '_LLRsum_entropy')

    base_name = f'{fname_prefix}{fname_base}'
    candidates = []
    if data_subfolder:
        candidates.append(str(Path(data_subfolder) / base_name))
    candidates.append(base_name)
    return fname_base, candidates


def resolve_feature_stem(data_folder, component_featureset, fname_prefix, data_subfolder=''):
    feature_dir = subfolders[get_featureset_dir(component_featureset)]
    fmt = '.csv' if feature_names[component_featureset] is not None else ARRAY_FEATURE_FORMATS.get(component_featureset, '.csv')

    for candidate in _candidate_feature_bases(component_featureset, fname_prefix, data_subfolder)[1]:
        stem_path = as_path(data_folder) / feature_dir / candidate
        if stem_path.with_suffix(fmt).exists():
            return feature_dir, candidate

    _, candidates = _candidate_feature_bases(component_featureset, fname_prefix, data_subfolder)
    return feature_dir, candidates[0]


def load_label_dataframe(
    labels_dir,
    labels_actual_fname,
    data_suffix_list=None,
    filter_out_data=None,
    filter_in_data=None,
):
    if data_suffix_list is None:
        data_suffix_list = ['']
    if filter_out_data is None:
        filter_out_data = {'mutations': ['WT', 'NC', 'X']}
    if filter_in_data is None:
        filter_in_data = {}

    df0 = load_data_allsuffixes(labels_actual_fname, str(as_path(labels_dir)) + '/', data_suffix_list, fmt='.csv')
    for col, val in filter_out_data.items():
        df0 = df0[~df0[col].isin(val)]
    for col, val in filter_in_data.items():
        df0 = df0[df0[col].isin(val)]
    return ensure_name_column(df0)


def assemble_featureset_dataset(
    data_folder,
    data_subfolder,
    labels_dir,
    data_suffix_list,
    fname_prefix,
    labels_actual_fname,
    ylabel_list,
    component_featureset_list,
    extra_cols_to_get,
    filter_out_data=None,
    filter_in_data=None,
    merge_on='mutations',
    deduplicate_data=False,
    get_classification_label=False,
):
    df0 = load_label_dataframe(
        labels_dir=labels_dir,
        labels_actual_fname=labels_actual_fname,
        data_suffix_list=data_suffix_list,
        filter_out_data=filter_out_data,
        filter_in_data=filter_in_data,
    )
    print('labels dataframe shape:', df0.shape)

    df = df0.copy()
    feature_list_all = []

    for component_featureset in component_featureset_list:
        feature_subdir, features_fname = resolve_feature_stem(data_folder, component_featureset, fname_prefix, data_subfolder)
        feature_list = feature_names[component_featureset]
        feature_stem = str(as_path(data_folder) / feature_subdir / features_fname)

        print(component_featureset, feature_list)
        print(f'Parsing features for {component_featureset}...')
        if feature_list is not None:
            df_features = load_data_allsuffixes('', feature_stem, data_suffix_list, fmt='.csv')
        else:
            fmt = ARRAY_FEATURE_FORMATS.get(component_featureset)
            if fmt is None:
                raise ValueError(f'Unsupported dense feature array format for {feature_stem}')
            arr = load_data_allsuffixes('', feature_stem, data_suffix_list, fmt=fmt)
            feature_list = list(range(arr.shape[1]))
            df_features = pd.DataFrame(arr, columns=feature_list)
            df_features.insert(0, 'name', df['name'].tolist())

        if 'name' not in df_features:
            if 'protein_name' in df_features and 'mutations' in df_features:
                df_features['name'] = [f'{prot}_{mut}' for prot, mut in zip(df_features['protein_name'].tolist(), df_features['mutations'].tolist())]
            elif 'Unnamed: 0' in df_features:
                df_features = df_features.rename(columns={'Unnamed: 0': 'name'})

        if 'embeddings' in component_featureset and len([fs for fs in component_featureset_list if 'embeddings' in fs]) > 1:
            embedding_prefix = component_featureset[:component_featureset.find('_') + 1]
            replacement = component_featureset[:component_featureset.find('embeddings')]
            feature_list = [f.replace(embedding_prefix, replacement) for f in feature_list]
            df_features.columns = [f.replace(embedding_prefix, replacement) for f in df_features.columns.tolist()]

        feature_list_all += feature_list
        df_features = df_features.drop_duplicates()
        print('df_features.shape:', df_features.shape)

        if len(df) == len(df_features) and component_featureset in ['ohe', 'oheMT', 'georgiev', 'georgievMT']:
            df = pd.concat([df.reset_index(drop=True), df_features[feature_list].reset_index(drop=True)], axis=1)
            print('Concatenated dense features with dataframe:', df.shape)
        else:
            merge_cols = [merge_on] + feature_list if merge_on in df_features.columns else ['name'] + feature_list
            merge_key = merge_on if merge_on in df_features.columns else 'name'
            df = df.merge(df_features[merge_cols], on=merge_key, how='left')
            print('Merged features with dataframe:', df.shape)
        validate_feature_merge(df, feature_list, component_featureset, len(df0))

    required_cols = [c for c in list(extra_cols_to_get) + feature_list_all + list(ylabel_list) if c in df.columns]
    if deduplicate_data:
        df = deduplicate_data_average_labels(
            df,
            merge_on,
            feature_list_all + list(ylabel_list),
            [c for c in extra_cols_to_get if c in df.columns],
            get_classification_label=get_classification_label,
        )
        print('After deduplicating:', df.shape)
        required_cols = [c for c in list(extra_cols_to_get) + feature_list_all + list(ylabel_list) if c in df.columns]

    df = df[required_cols]
    df = df.dropna(axis=0, ignore_index=True)
    print('FINAL XY DATAFRAME SIZE:', df.shape)
    return df, feature_list_all


def load_precompiled_dataset(input_dir, dataset_fbase, featureset, ylabel, extra_cols_to_get=None, n_splits=5):
    input_dir = as_path(input_dir)
    x_features = feature_names_multimodal.get(featureset, feature_names[featureset])
    dataset_fpath = input_dir / f'{dataset_fbase}_{featureset}.csv'
    df_all = pd.read_csv(dataset_fpath)

    base_cols = list(x_features) + [ylabel]
    optional_candidates = list(extra_cols_to_get or [])
    optional_candidates += ['mutations', 'Position', 'fold_random_5', f'fold_random_{n_splits}', 'protein_name', 'name']
    optional_cols = [c for c in optional_candidates if c in df_all.columns and c not in base_cols]
    df = df_all[base_cols + optional_cols].dropna(axis=0, ignore_index=True)
    return df, list(x_features)
