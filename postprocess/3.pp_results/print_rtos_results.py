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

import os
import pickle
import torch
import pandas as pd
from os.path import join, basename
from collections import defaultdict

PROJ_ROOT = "."


def load_pickle(fp):
    with open(fp, "rb") as f:
        return pickle.load(f)


def calculate_err(fp):
    f_info_fp = join(
        PROJ_ROOT, "dbs/Dataset-RTOS-525/testing_Dataset-RTOS-525.csv")
    matched_fp = join(PROJ_ROOT, "dbs/Dataset-RTOS-525/matched.csv")

    f_infos = pd.read_csv(f_info_fp)
    matched = pd.read_csv(matched_fp)

    pairs = {}
    for idx, r in matched.iterrows():
        query_id, m_id = r['matched_id'], r['id']
        if not query_id in pairs:
            pairs[query_id] = ([], [])
        pairs[query_id][r['category']].append(m_id)

    q_sims = {}
    if fp.endswith(".pkl"):
        embeddings = load_pickle(fp)
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.from_numpy(embeddings)
        for q_id in pairs:
            q_sims[q_id] = torch.cosine_similarity(
                embeddings[[q_id,]], embeddings)
    elif fp.endswith(".csv"):
        df = pd.read_csv(fp)
        for q_id in pairs:
            qs = (df['idb_path_1'] == f_infos.iloc[q_id]['idb_path']) & \
                (df['fva_1'] == f_infos.iloc[q_id]['fva'])
            assert df[qs]['idx_2'].is_monotonic_increasing and \
                len(df[qs]['idx_2']) == len(f_infos)
            q_sims[q_id] = df[qs]['sim'].values

    def isxa(idx): return any([s in f_infos.iloc[idx]['idb_path'] for s in [
        'fac1200RQ_F400', 'M6G_F400', 'ap1207gi_FB00', 'sg2206r_1174C', 'wa933re_F114', 'wdr5650_1F400', 'wr886n_A200']])

    errs = {}
    xa, xa_f = [0, 0], [0, 0]
    for q_id, (c0_items, c1_items) in pairs.items():
        sims = q_sims[q_id]
        sims_with_idx = list(zip(list(range(len(sims))), sims))
        sorted_sims = sorted(sims_with_idx, key=lambda x: x[1], reverse=True)
        err_c, err_s = [0, 0], [0, 0]
        recall, mrr = [[], []], [[], []]
        acc_c0, acc_c1, acc_c2 = 0, 0, 0
        for rank, (idx, score) in enumerate(sorted_sims):
            if idx in c0_items:
                if acc_c1+acc_c2 != 0:
                    xa_f[0] += 1 if isxa(idx) else 0
                    err_c[0] += 1
                    err_s[0] += acc_c1+acc_c2
                    recall[0].append(0)
                    mrr[0].append(1 / (acc_c1+acc_c2+1))
                else:
                    xa[0] += 1 if isxa(idx) else 0
                    recall[0].append(1)
                    mrr[0].append(1)
                acc_c0 += 1
            elif idx in c1_items:
                if acc_c2 != 0:
                    xa_f[1] += 1 if isxa(idx) else 0
                    err_c[1] += 1
                    err_s[1] += acc_c2
                    recall[1].append(0)
                    mrr[1].append(1 / (acc_c2+1))
                else:
                    xa[1] += 1 if isxa(idx) else 0
                    recall[1].append(1)
                    mrr[1].append(1)
                acc_c1 += 1
            else:
                acc_c2 += 1
            if acc_c0 == len(c0_items) and acc_c1 == len(c1_items):
                break
        errs[q_id] = ((len(c0_items), len(c1_items)),
                      err_c, err_s, recall, mrr)
    return errs


def merge_rtos_results(results):
    from copy import deepcopy
    merged_r = deepcopy(results[0])

    for qs in results[1:]:
        for q_id, r in qs.items():
            merged_r[q_id][2].extend(r[2])
            for i in range(2):
                merged_r[q_id][1][i] += r[1][i]
                merged_r[q_id][2][i] += r[2][i]
                merged_r[q_id][3][i].extend(r[3][i])
                merged_r[q_id][4][i].extend(r[4][i])

    for q_id in qs.keys():
        for i in range(2):
            merged_r[q_id][1][i] = (merged_r[q_id][1][i]/len(results))
            merged_r[q_id][2][i] = (merged_r[q_id][2][i]/len(results))

    return merged_r


def cal_mean(lst):
    return sum(lst)/len(lst) if len(lst) != 0 else 1.0


if __name__ == "__main__":
    results = {}

    # Get Stats
    rtos_pkl_fn = "testing_Dataset-RTOS-525.pkl"
    for model in ['SAFE', 'Trex']:
        print(f"Collect results of {model}")
        pkl_fp = join("outputs/experiments/baselines",
                      model.lower(), rtos_pkl_fn)
        results[model] = calculate_err(pkl_fp)

    print("Collect results of GMN")
    gmn_csv_fp = "outputs/experiments/baselines/gmn/pairs-xm_sim.csv"
    results['GMN'] = calculate_err(gmn_csv_fp)

    print("Collect results of HermesSim")
    hermes_sim_results_dir = "outputs/experiments/hermes_sim"
    hermes_sim_results = []
    ## Results of Hermes is the mean of ten independent runs.
    for fn in os.listdir(hermes_sim_results_dir):
        pkl_fp = join(hermes_sim_results_dir, fn, "last",
                      "testing_Dataset-RTOS-525.pkl")
        hermes_sim_results.append(calculate_err(pkl_fp))
    results['HermesSim'] = merge_rtos_results(hermes_sim_results)

    # Postprocess
    names = []
    err_cnts = defaultdict(lambda: [[], []])
    recalls, mrrs = defaultdict(
        lambda: [[], []]), defaultdict(lambda: [[], []])
    order_map = {
        "SAFE": 1,
        "Trex": 2,
        "GMN": 3,
        "HermesSim": 4,
    }
    r_list = sorted([(name, qs) for name, qs in results.items()],
                    key=lambda x: order_map[x[0]])
    for name, qs in r_list:
        names.append(name)
        for q_id, r in qs.items():
            for i in range(2):
                recalls[name][i].extend(r[3][i])
                mrrs[name][i].extend(r[4][i])
            err_cnts[q_id][0].append((r[1][0], r[1][1]))

    # Print results in the form of latex table
    print(" & ".join(names))
    total = [0 for i in range(2+2*len(names))]
    def ff(s): return f"{s*100:.0f}"
    recall_s, mrr_s = "recall & ", "mrr & "
    for oid, q_id in enumerate(qs):
        for i in range(2):
            total[i] += results[names[0]][q_id][0][i]
        for i in range(len(names)):
            total[i*2+2] += err_cnts[q_id][0][i][0]
            total[i*2+3] += err_cnts[q_id][0][i][1]
        print(oid, '&', ' & '.join([str(i) for i in results[names[0]][q_id][0]]), '&', ' & '.join(
            [' & '.join([str(tt) for tt in t]) for t in err_cnts[q_id][0]]), "\\\\")
    print('total', '&', ' & '.join([str(tt) for tt in total]), "\\\\")
    for name, _ in r_list:
        recall_s += f" & {ff(cal_mean(recalls[name][0]))} & {ff(cal_mean(recalls[name][1]))}"
        mrr_s += f" & {ff(cal_mean(mrrs[name][0]))} & {ff(cal_mean(mrrs[name][1]))}"
    print(recall_s, "\\\\")
    print(mrr_s, "\\\\")
