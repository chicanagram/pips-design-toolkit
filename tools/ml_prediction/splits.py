from __future__ import annotations

import numpy as np
import pandas as pd


def sort_list(lst):
    lst.sort()
    return lst


def get_random_split_idxs(df, n_splits=5, y_col=None, use_precomputed_folds=False, stratify=False, random_state=42):
    split_col = f'fold_random_{n_splits}'
    if use_precomputed_folds and split_col in df.columns:
        return get_split_idxs_from_col(df, split_col, n_splits=n_splits), split_col

    from sklearn.model_selection import KFold, StratifiedKFold

    if stratify:
        if y_col is None:
            raise ValueError('y_col is required when stratify=True.')
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(df, df[y_col])
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(df)
    return list(split_iter), None


def get_mutres_modulo_splits(df, n_splits=5, proteinbase_mutres_col='proteinbase_mutres'):
    print('Getting mutres_modulo splits...')
    proteinbase_mutres_list = sort_list(list(set(df[proteinbase_mutres_col].tolist())))
    print(f'{len(proteinbase_mutres_list)} unique protein(base)/mutated positions in dataset.')

    modulo_splits = {k: [] for k in range(n_splits)}
    modulo_splits_traintest = []
    for i, proteinbase_mutres in enumerate(proteinbase_mutres_list):
        modulo_splits[i % n_splits].append(proteinbase_mutres)

    for k, modulo_list_k in modulo_splits.items():
        df_subset = df[df[proteinbase_mutres_col].isin(modulo_list_k)]
        idxs_test = list(df_subset.index)
        idxs_train = [idx for idx in range(len(df)) if idx not in idxs_test]
        modulo_splits_traintest.append((idxs_train, idxs_test))
        print(k, f'{len(modulo_list_k)} unique proteinbase_mutres', modulo_list_k)
        print(f'{len(idxs_test)} samples', idxs_test)
    return modulo_splits_traintest, modulo_splits


def get_protein_splits(df, n_splits=5, protein_base_col='protein_base'):
    print('Getting protein splits...')
    protein_base_list = sort_list(list(set(df[protein_base_col].tolist())))
    protein_base_to_nsamples = []
    print(f'{len(protein_base_list)} unique proteins (base) in dataset.')
    for protein_base in protein_base_list:
        df_proteinbase = df[df[protein_base_col] == protein_base]
        protein_base_to_nsamples.append({protein_base_col: protein_base, 'num_samples': len(df_proteinbase)})
    protein_base_to_nsamples = pd.DataFrame(protein_base_to_nsamples)
    protein_base_to_nsamples = protein_base_to_nsamples.sort_values(by='num_samples', ascending=False).reset_index(drop=False)
    print(protein_base_to_nsamples)

    test_size_thres = int(np.ceil(len(df) / n_splits))
    print('Test size thres:', test_size_thres)

    protein_base_to_nsamples_filt = protein_base_to_nsamples[protein_base_to_nsamples['num_samples'] <= test_size_thres].reset_index(drop=True)
    protein_base_list_filt = protein_base_to_nsamples_filt[protein_base_col].tolist()
    num_proteins_for_test_pool = len(protein_base_to_nsamples_filt)
    num_proteins_for_test_pool_halved = int(np.floor(num_proteins_for_test_pool / 2))
    print('# of proteins with <= test_size_thres samples:', num_proteins_for_test_pool)

    protein_base_idx_list_ordered = [0]
    n = len(protein_base_list_filt)
    for protein_base_idx in range(1, num_proteins_for_test_pool_halved):
        protein_base_idx_list_ordered += [n - protein_base_idx, protein_base_idx]
    print('protein_base_idx_list_ordered:', protein_base_idx_list_ordered)

    protein_splits_all = {k: [] for k in range(n_splits)}
    for i in range(0, num_proteins_for_test_pool_halved):
        protein_splits_all[i % n_splits] += protein_base_idx_list_ordered[2 * i:2 * i + 2]

    protein_splits = {k: [] for k in range(n_splits)}
    protein_splits_traintest = []
    for k in range(n_splits):
        protein_base_list_k = protein_splits_all[k]
        print(k, len(protein_base_list_k), protein_base_list_k)
        protein_base_to_nsamples_k = protein_base_to_nsamples_filt.loc[protein_base_list_k, :]
        num_samples_k = protein_base_to_nsamples_k['num_samples'].to_numpy()
        cum_num_samples_k = np.cumsum(num_samples_k)
        protein_base_to_nsamples_k['cum_num_samples'] = cum_num_samples_k
        print(protein_base_to_nsamples_k)
        protein_base_test_k = protein_base_to_nsamples_k[protein_base_to_nsamples_k['cum_num_samples'] <= test_size_thres]
        protein_list_k = protein_base_test_k[protein_base_col].tolist()
        protein_splits[k] = protein_list_k

        df_subset = df[df[protein_base_col].isin(protein_list_k)]
        idxs_test = list(df_subset.index)
        idxs_train = [idx for idx in range(len(df)) if idx not in idxs_test]
        protein_splits_traintest.append((idxs_train, idxs_test))
        print(k, f'{len(protein_list_k)} unique protein_base', protein_list_k)
        print(f'{len(idxs_test)} samples', idxs_test)
    print()

    return protein_splits_traintest, protein_splits


def get_custom_split(df, test_csv_fpath, cols_to_overlay=None):
    df_test = pd.read_csv(test_csv_fpath)
    if cols_to_overlay is None:
        if {'gi', 'variation'}.issubset(df_test.columns) and {'gi', 'mutations'}.issubset(df.columns):
            cols_to_overlay = [('gi', 'gi'), ('variation', 'mutations')]
        elif 'mutations' in df_test.columns and 'mutations' in df.columns:
            cols_to_overlay = [('mutations', 'mutations')]
        elif 'name' in df_test.columns and 'name' in df.columns:
            cols_to_overlay = [('name', 'name')]
        else:
            raise ValueError(
                'Unable to infer custom split overlay columns. '
                'Expected either gi/variation, mutations, or name columns.'
            )

    test_idxs = []
    for i in range(len(df_test)):
        test_df_filt = df.copy()
        for test_col, df_col in cols_to_overlay:
            test_df_filt = test_df_filt[test_df_filt[df_col] == df_test.iloc[i][test_col]]
        test_idx = int(test_df_filt.index[0])
        test_idxs.append(test_idx)
    train_idxs = [i for i in range(len(df)) if i not in test_idxs]
    return [(train_idxs, test_idxs)]


def get_split_idxs_from_col(df, col, n_splits=5):
    df = df.reset_index(drop=True)
    all_idxs = list(range(len(df)))
    split_idxs = []
    for k in range(n_splits):
        df_split = df[df[col] == k]
        test_idxs = list(df_split.index)
        train_idxs = [i for i in all_idxs if i not in test_idxs]
        split_idxs.append((train_idxs, test_idxs))
    return split_idxs
