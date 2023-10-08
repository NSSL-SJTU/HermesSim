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

# This script generate testing pairs under various restriction for evaluation.

import json
import pandas as pd
import sys
import os
import networkx as nx
import pickle
import random
import numpy as np
from os.path import join, basename
from tqdm import tqdm
from collections import defaultdict


DB_TYPE = 'XM'
SEED = 0
N_POS = 200
N_NEG_PER_POS = 100

SIZE_RANGE_FOR_QUERY_ONLY = False
SIZE_RANGE = None

RESTRICTION = {
    ## E.g.
    # "arch": ["x"],
    # "bit": [64],
    # "proj": ['nmap_ncat', 'z3_z3', 'nmap_nmap', 'nmap_nping'],
}
SAVE_SYMBOL = True

CONFIG_ITEMS = [
    'DB_TYPE', 'SEED', 'N_POS', 'N_NEG_PER_POS',
    'SIZE_RANGE_FOR_QUERY_ONLY', 'SIZE_RANGE',
    'RESTRICTION', 'SAVE_SYMBOL'
]


# Filename of the output pair csv
def get_desc():
    s = f"{DB_TYPE.lower()}-{N_POS}-{N_NEG_PER_POS}"
    if SEED != 0:
        s += f"-{SEED}"
    if SIZE_RANGE is not None:
        q = "q_" if SIZE_RANGE_FOR_QUERY_ONLY else ""
        s += f"-{q}{SIZE_RANGE[0]}_{SIZE_RANGE[1]}"
    for k, v in RESTRICTION.items():
        s += f"-{k}_{'_'.join([str(t) for t in v])}"
    return s


def get_bin_name(idb_path):
    bin_name = basename(idb_path)
    assert bin_name.endswith(".i64")
    return bin_name[:-4]


def get_acfg_features_json(acfg_jsons_folder, bin_name):
    json_fp = join(acfg_jsons_folder, bin_name + "_acfg_disasm.json")
    with open(json_fp, "r") as f:
        obj = json.load(f)
    return obj


def read_config(fp):
    with open(fp, 'r') as f:
        c = json.load(f)
    ds_info_key = 'DATASET_INFO_CSV'
    if not ds_info_key in c:
        print("Error: Configure file should at least contain a path to a csv that describe the dataset. ")
        exit(0)
    thismodule = sys.modules[__name__]
    for item in CONFIG_ITEMS:
        if item in c:
            setattr(thismodule, item, c[item])
    return c[ds_info_key]


if __name__ == "__main__":
    out_dir = sys.argv[1]
    config_fp = sys.argv[2]
    ds_info_csv = read_config(config_fp)
    random.seed(SEED)
    summary = defaultdict(lambda: defaultdict(dict))
    df = pd.read_csv(ds_info_csv)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        idb_path = row['idb_path']
        fva = row['fva']
        arch, bit, opt = row['arch'], row['bit'], row['optimizations']
        comp, cver = row['compiler'], row['version']
        gsize = row['sizes']
        proj = f"{row['project']}_{row['library']}"
        fit = True
        for key, allowed in RESTRICTION.items():
            if (key == 'arch' and not arch in allowed) or \
                (key == 'bit' and not bit in allowed) or \
                    (key == 'proj' and not proj in allowed):
                fit = False
                break
        if not fit:
            continue
        f_id = (idb_path, fva)
        if not SIZE_RANGE_FOR_QUERY_ONLY and SIZE_RANGE is not None and (
                np.isnan(gsize) or gsize < SIZE_RANGE[0] or gsize >= SIZE_RANGE[1]):
            continue
        func_name = row['func_name']
        start_ea = row['start_ea']
        ## Output Type
        # ,idb_path_1,fva_1,func_name_1,idb_path_2,fva_2,func_name_2,db_type
        if DB_TYPE == 'XO':
            summary[(proj, func_name)][(arch, bit, comp, cver)][opt] = (
                idb_path, start_ea, func_name, idx, gsize)
        elif DB_TYPE == 'XC':
            summary[(proj, func_name)][(arch, bit)][(comp, cver, opt)] = (
                idb_path, start_ea, func_name, idx, gsize)
        elif DB_TYPE == 'XB':
            summary[(proj, func_name)][(arch, comp, cver, opt)][(bit,)] = (
                idb_path, start_ea, func_name, idx, gsize)
        elif DB_TYPE == 'XA':
            summary[(proj, func_name)][(comp, cver, opt)][(arch, bit)] = (
                idb_path, start_ea, func_name, idx, gsize)
        elif DB_TYPE == 'XM':
            summary[(proj, func_name)][''][(arch, bit, comp, cver, opt)] = (
                idb_path, start_ea, func_name, idx, gsize)

    n_syms = len(summary)
    all_syms = list(summary.keys())
    print(f"Number Symbols: {n_syms}")
    count, skipped = 0, 0
    out_triplets = [[[] for _ in range(4)] for _ in range(N_NEG_PER_POS+2)]
    summary_list = list(summary.items())
    random.shuffle(summary_list)
    process = iter(tqdm(range(N_POS)))

    def filter_group(group: dict[tuple, tuple]):
        global SIZE_RANGE_FOR_QUERY_ONLY
        if SIZE_RANGE_FOR_QUERY_ONLY and SIZE_RANGE is not None:
            v_t = {}
            for k, v in group.items():
                gsize = v[4]
                if np.isnan(gsize) or gsize < SIZE_RANGE[0] or gsize >= SIZE_RANGE[1]:
                    continue
                v_t[k] = v
            group = v_t
        return group
    for sym, sdata in summary_list:
        one_sym = [(k, filter_group(v)) for k, v in sdata.items()]
        one_sym = [(k, v) for k, v in one_sym if len(v) >= 2]
        if len(one_sym) == 0:
            continue
        # 1. Sample a Query Function
        ## chosen_k is the settings kept by all compared functions
        chosen_k, chosen_v = random.choice(one_sym)
        pairs = random.sample(list(chosen_v.values()), 2)

        # 2. Sample a Pool for the Query
        neg_cands = [s for s in all_syms if s !=
                     sym and chosen_k in summary[s]]
        if len(neg_cands) < N_NEG_PER_POS:
            skipped += 1
            continue
        neg_funcs = random.sample(neg_cands, N_NEG_PER_POS)
        for f in neg_funcs:
            sam = random.choice(list(summary[f][chosen_k].values()))
            pairs.append(sam)

        for i in range(N_NEG_PER_POS+2):
            for j in range(4):
                out_triplets[i][j].append(pairs[i][j])

        count += 1
        next(process)
        if count == N_POS:
            break
    next(process, None)

    print(f"Number Pos Pairs: {len(out_triplets[0][1])}")
    print(f"Number Neg Pairs: {len(out_triplets[0][1])*N_NEG_PER_POS}")

    os.makedirs(out_dir, exist_ok=True)

    pos_data = {
        'idb_path_1': out_triplets[0][0],
        'fva_1': out_triplets[0][1],
        'idx_1': out_triplets[0][3],
        'idb_path_2': out_triplets[1][0],
        'fva_2': out_triplets[1][1],
        'idx_2': out_triplets[1][3],
        'db_type': [DB_TYPE for _ in out_triplets[0][1]],
    }
    if SAVE_SYMBOL:
        pos_data['func_name_1'] = out_triplets[0][2]
        pos_data['func_name_2'] = out_triplets[1][2]
    posdf = pd.DataFrame(pos_data)
    pos_df_path = join(out_dir, f"pos-{get_desc()}_Ds1.csv")
    posdf.to_csv(pos_df_path)
    print(f"Saved {pos_df_path}")

    neg_pairs = [[[] for _ in range(4)] for _ in range(2)]
    for i in range(2, N_NEG_PER_POS+2):
        for k in range(4):
            neg_pairs[0][k].extend(out_triplets[0][k])
            neg_pairs[1][k].extend(out_triplets[i][k])

    neg_data = {
        'idb_path_1': neg_pairs[0][0],
        'fva_1': neg_pairs[0][1],
        'idx_1': neg_pairs[0][3],
        'idb_path_2': neg_pairs[1][0],
        'fva_2': neg_pairs[1][1],
        'idx_2': neg_pairs[1][3],
        'db_type': [DB_TYPE for _ in neg_pairs[0][1]],
    }
    if SAVE_SYMBOL:
        neg_data['func_name_1'] = neg_pairs[0][2]
        neg_data['func_name_2'] = neg_pairs[1][2]
    negdf = pd.DataFrame(neg_data)
    neg_df_path = join(out_dir, f"neg-{get_desc()}_Ds1.csv")
    negdf.to_csv(neg_df_path)
    print(f"Saved {neg_df_path}")
