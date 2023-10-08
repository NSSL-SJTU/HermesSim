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

import pickle
import argparse
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from os.path import basename, join
from collections import defaultdict


def get_bin_name(idb_path):
    bin_name = basename(idb_path)
    assert bin_name.endswith(".i64")
    return bin_name[:-4]


def get_embed(ebeds_map, idb_path, fname):
    bin_name = get_bin_name(idb_path)
    arch, comp, cver, opt_proj = bin_name.split('-')
    opt, proj = opt_proj.split('_')
    proj_triplet = '-'.join([comp, cver, proj])
    return ebeds_map[proj_triplet][fname][opt]


def compute_sim_for_jTrans(pkl_path, pair_path, output_dir):
    with open(pkl_path, 'rb') as ff:
        ebds = pickle.load(ff)

    ebeds_map = defaultdict(lambda: defaultdict(dict))
    for func in ebds:
        proj = func['proj']
        fname = func['funcname']
        ebeds_map[proj][fname] = func

    pair_df = pd.read_csv(pair_path)
    sims = []
    dropped = []
    ## Output format
    # ,idb_path_1,fva_1,func_name_1,idb_path_2,fva_2,func_name_2,db_type,sim
    for _, r in tqdm(pair_df.iterrows(), total=pair_df.shape[0]):
        embed1 = get_embed(ebeds_map, r['idb_path_1'], r['func_name_1'])
        embed2 = get_embed(ebeds_map, r['idb_path_2'], r['func_name_2'])
        sim = F.cosine_similarity(embed1, embed2).item()
        sims.append(sim)

    rest_df = pair_df.copy()
    rest_df = rest_df.drop(dropped)
    rest_df = rest_df.drop(columns=['Unnamed: 0'])
    rest_df['sim'] = sims
    rest_df.to_csv(join(output_dir, basename(pair_path)[:-4]+"_sim.csv"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="jTrans-FastEval")
    parser.add_argument("--experiment_path", type=str,
                        default='./experiments/Dataset-1/x64_testing.pkl', help="experiment to be evaluated")
    parser.add_argument("--pair_path", type=str, required=True,
                        help="experiment to be evaluated")
    parser.add_argument("--output_path", type=str,
                        required=True, help="experiment to be evaluated")

    args = parser.parse_args()
    compute_sim_for_jTrans(args.experiment_path,
                           args.pair_path, args.output_path)
