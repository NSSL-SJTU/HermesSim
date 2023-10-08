#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################################
#                                                                            #
#  Code for the USENIX Security '24 paper:                                   #
#  Code is not Natural Language: Unlock the Power of Semantics-Oriented      #
#             Graph Representation for Binary Code Similarity Detection      #
#                                                                            #
#  MIT License                                                               #
#                                                                            #
#  Copyright (c) 2023 SJTU NSSL Lab                                     #
#                                                                            #
##############################################################################

import sys
import numpy as np
import os
import pickle
import torch
import pandas as pd

from tqdm import tqdm
from collections import defaultdict
from os.path import basename, dirname

import scipy.stats as stats


def merge_data(df_pairs, df_similarity, is_pos=None):
    df_pairs = df_pairs.merge(
        df_similarity,
        how='inner',
        left_on=['idb_path_1', 'fva_1', 'idb_path_2', 'fva_2', 'db_type'],
        right_on=['idb_path_1', 'fva_1', 'idb_path_2', 'fva_2', 'db_type'])

    if is_pos:
        # If positive pairs, the perfect similarity is 1
        df_pairs['gt'] = [1] * df_pairs.shape[0]
    elif is_pos is not None:
        # if negative pairs, the perfect similarity is 0
        df_pairs['gt'] = [-1] * df_pairs.shape[0]

    return df_pairs


# ============================================================================
# Recall & Mrr10 =============
# ============================================================================

POOL_SIZES = sorted([2**i for i in range(1, 14)] + [100, 10000])


def compute_ranking(df_pos, df_neg, test_name, r_dict, rank_method):
    for task in sorted(set(df_pos['db_type'])):
        df_pos_task = df_pos[df_pos['db_type'] == task]
        df_neg_task = df_neg[df_neg['db_type'] == task]

        # Compute the ranking for all the positive test cases
        rank_list = defaultdict(list)
        worest_case = (0, None)
        for idx, group in df_neg_task.groupby(['idb_path_1', 'fva_1']):
            c1 = (df_pos_task['idb_path_1'] == idx[0])
            c2 = (df_pos_task['fva_1'] == idx[1])
            pos_pred = df_pos_task[c1 & c2]['sim'].values[0]
            neg_pred = list(group['sim'].values)
            for psize in POOL_SIZES:
                ranks = stats.rankdata(
                    [pos_pred] + neg_pred[:psize-1], method=rank_method)
                r = psize + 1 - ranks[0]
                rank_list[psize].append(r)
            if r > worest_case[0]:
                worest_case = (r, df_pos_task[c1 & c2])

        # Save data in a temporary dictionary
        if task not in r_dict:
            r_dict[task] = defaultdict(list)

        r_dict[task]['model_name'].append(test_name)
        for psize, ranks in rank_list.items():
            recall_1 = len([x for x in ranks if x <= 1]) / len(ranks)
            mrr = sum([1 / x for x in ranks]) / len(ranks)
            r_dict[task][f"Recall_1@P{psize}"].append(recall_1)
            r_dict[task][f"MRR@P{psize}"].append(mrr)


def model_name_from_result_path(result_path: str):
    test_name = basename(result_path)
    if test_name.isdigit() or test_name in ['last', 'best', 'full', 'size', 'x64'] or test_name.startswith('ckpt_'):
        test_name = model_name_from_result_path(dirname(result_path))
    return test_name


def compute_mrr_and_recall(df_pos, df_neg, pos_name, results_dir, rank_method):
    print("[D] Using rank_method: {}".format(rank_method))
    results_dict = dict()

    pos_csv_file = pos_name + "_sim.csv"
    neg_csv_file = pos_csv_file.replace("pos", "neg")
    print("[D] Processing\n\t{}\n\t{}".format(pos_csv_file, neg_csv_file))

    pos_csv_fp = os.path.join(results_dir, pos_csv_file)
    neg_csv_fp = os.path.join(results_dir, neg_csv_file)
    if not os.path.exists(pos_csv_fp) or not os.path.exists(neg_csv_fp):
        print(f"[E] {pos_csv_fp} / {neg_csv_fp} does not exist. ")
        return

    df_pos_sim = pd.read_csv(pos_csv_fp)
    df_neg_sim = pd.read_csv(neg_csv_fp)

    assert (df_pos_sim.isna().sum()['sim'] == 0)
    assert (df_neg_sim.isna().sum()['sim'] == 0)

    df_pos_m = merge_data(df_pos, df_pos_sim)
    df_neg_m = merge_data(df_neg, df_neg_sim)

    test_name = model_name_from_result_path(results_dir)
    compute_ranking(df_pos_m, df_neg_m, test_name,
                    results_dict, rank_method=rank_method)

    return results_dict


def compute_pair_sims(df: pd.DataFrame, embeddings):
    pair_1, pair_2 = np.array(df['idx_1']), np.array(df['idx_2'])
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.from_numpy(embeddings)
    BATCH_SIZE = 100000
    all_sims = []
    for i in tqdm(range(0, len(pair_1), BATCH_SIZE)):
        e_1 = embeddings[pair_1[i:i+BATCH_SIZE]]
        e_2 = embeddings[pair_2[i:i+BATCH_SIZE]]
        sims = torch.cosine_similarity(e_1, e_2).cpu().numpy()
        all_sims.append(sims)
    return np.concatenate(all_sims, 0)


def form_result_csv_name(pos_testing_fn, rank_method):
    return "{}_MRR_Recall_{}.csv".format(pos_testing_fn[4:-4], rank_method)


def from_dict_to_df(r_dict, output_dir, rank_method, pos_testing_fn):
    for task in r_dict.keys():
        print("[D] Task: {}".format(task))
        df_rank = pd.DataFrame.from_dict(r_dict[task])
        df_rank.set_index('model_name', inplace=True)
        csv_name = form_result_csv_name(pos_testing_fn, rank_method)
        df_rank.to_csv(os.path.join(output_dir, csv_name))


def compute_ranking_with_pkl(df_pos, df_neg, test_name, r_dict, rank_method):

    for task in sorted(set(df_pos['db_type'])):
        df_pos_task = df_pos[df_pos['db_type'] == task]
        df_neg_task = df_neg[df_neg['db_type'] == task]

        # Compute the ranking for all the positive test cases
        rank_list = [list() for _ in POOL_SIZES]
        for idx, group in df_neg_task.groupby(['idb_path_1', 'fva_1']):
            c1 = (df_pos_task['idb_path_1'] == idx[0])
            c2 = (df_pos_task['fva_1'] == idx[1])
            pos_pred = df_pos_task[c1 & c2]['sim'].values[0]
            neg_pred = list(group['sim'].values)
            for i, psize in enumerate(POOL_SIZES):
                ranks = stats.rankdata(
                    [pos_pred] + neg_pred[:psize-1], method=rank_method)
                r = psize + 1 - ranks[0]
                rank_list[i].append(r)

        # Save data in a temporary dictionary
        if task not in r_dict:
            r_dict[task] = defaultdict(list)
        r_dict[task]['model_name'].append(test_name)

        for ranks, psize in zip(rank_list, POOL_SIZES):
            recall_1 = len([x for x in ranks if x <= 1]) / len(ranks)
            mrr = sum([1/x for x in ranks]) / len(ranks)
            r_dict[task][f"Recall_1@P{psize}"].append(recall_1)
            r_dict[task][f"MRR@P{psize}"].append(mrr)


def compute_mrr_and_recall_with_pkl(df_pos, df_neg, pkl_fp, results_dir, rank_method):
    print("[D] Using rank_method: {}".format(rank_method))
    with open(pkl_fp, "rb") as f:
        pkl_data = pickle.load(f)
    print(f"Loaded: {pkl_fp}")
    sims_pos = compute_pair_sims(df_pos, pkl_data)
    print("POS sims shape: ", sims_pos.shape)
    sims_neg = compute_pair_sims(df_neg, pkl_data)
    print("NEG sims shape: ", sims_neg.shape)
    df_pos['sim'] = sims_pos
    df_neg['sim'] = sims_neg
    results_dict = dict()
    test_name = model_name_from_result_path(results_dir)
    compute_ranking_with_pkl(df_pos, df_neg, test_name,
                             results_dict, rank_method=rank_method)
    return results_dict


# Alternatives of rank_method: min or max
def compute_and_save_mrr_and_recall(RESULTS_DIR, pos_pair_csv, pkl_fp, rank_method='max'):
    base_path = os.path.dirname(pos_pair_csv)
    pos_testing_fn = os.path.basename(pos_pair_csv)
    assert pos_testing_fn.startswith(
        "pos-") and pos_testing_fn.endswith(".csv")
    neg_testing_fn = pos_testing_fn.replace("pos-", "neg-")

    if pkl_fp is not None and not os.path.exists(pkl_fp):
        print(f"Failed to find pkl data: {pkl_fp}")
        exit(0)

    df_pos_testing = pd.read_csv(
        os.path.join(base_path, pos_testing_fn),
        index_col=0)

    df_neg_testing = pd.read_csv(
        os.path.join(base_path, neg_testing_fn),
        index_col=0)

    print(f"POS TESTING: {pos_testing_fn}")

    if pkl_fp is not None:
        results_dict = compute_mrr_and_recall_with_pkl(df_pos_testing, df_neg_testing,
                                                       pkl_fp, RESULTS_DIR, rank_method)
    else:
        pos_name = pos_testing_fn.split('.')[0]
        results_dict = compute_mrr_and_recall(df_pos_testing, df_neg_testing,
                                              pos_name, RESULTS_DIR, rank_method)
    from_dict_to_df(results_dict, RESULTS_DIR, rank_method, pos_testing_fn)


if __name__ == "__main__":
    RESULTS_DIR = sys.argv[1]
    POS_PAIR_PATH = sys.argv[2]
    PKL_PATH = sys.argv[3] if len(sys.argv) > 3 else None
    compute_and_save_mrr_and_recall(RESULTS_DIR, POS_PAIR_PATH, PKL_PATH)
