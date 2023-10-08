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
import math
import pandas as pd
from os.path import join, basename

PROJ_ROOT = "../.."
EXPERIMENT_ROOT = join(PROJ_ROOT, "outputs/experiments")

## Utils

id_map = {
    "CFG-OPC200": 1, 
    "CFG-PalmTree": 2, 
    "CFG-HBMP": 3, 
    "P-DFG": 4, 
    "P-CDFG": 5, 
    "P-ISCG": 6, 
    "P-TSCG": 7, 
    "P-SOG": 8,
    
    "Set2Set": 9, 
    "Softmax": 10, 
    "Gated": 11, 

    "SAFE": 10, 
    "Asm2Vec": 11, 
    "Trex": 12, 
    "GMN": 13, 
    "jTrans": 14, 
    "HermesSim": 15, 
}

def map_name(name):
    if name.endswith("_last"):
        name = name[:-5]
    name_map = {
        "acfg_opc200": "CFG-OPC200", 
        "acfg_hbmp": "CFG-HBMP", 
        "palmtree": "CFG-PalmTree", 
        "dfg": "P-DFG", 
        "cdfg": "P-CDFG", 
        "iscg": "P-ISCG", 
        "tscg": "P-TSCG", 
        # "sog": "P-SOG", 
        "hermes_sim": "HermesSim", 
        "sog": "HermesSim", 
        "safe": "SAFE", 
        "trex": "Trex", 
        "asm2vec": "Asm2Vec", 
        "set2set": "Set2Set", 
        "jTrans": "jTrans", 
        "gmn": "GMN", 
        "gated": "Gated", 
        "softmax": "Softmax", 
    }
    return name_map.get(name, name)

def get_size_range(s):
    lst = s.split("_")
    if not len(lst) in [2, 3]:
        return None
    if not lst[-1].isdigit() or not lst[-2].isdigit():
        return None
    if len(lst) == 2:
        return [int(a) for a in lst]
    if lst[0] != 'q':
        return None
    return [int(a) for a in lst[1:]] + ['q']


def taskname_from_summary_fn(fn):
    postfix = "_Ds1_MRR_Recall_max.csv"
    fn = fn[:-len(postfix)]
    size_ranges = [s for s in map(get_size_range, fn.split("-")) if not s is None]
    assert len(size_ranges) <= 1
    size_ranges_desc = ""
    if len(size_ranges) == 1:
        size_ranges_desc = "-small" if size_ranges[0][0] == 0 else "-large"
        if len(size_ranges[0]) == 3:
            size_ranges_desc += '-query-only'
    return ("x64-" if '-arch_x-bit_64' in fn else "") + fn[8:10].upper() + size_ranges_desc

def get_groupped_dataframe(summary_fn, root=EXPERIMENT_ROOT):
    df = pd.read_csv(join(root, summary_fn), index_col=0)
    df = df.groupby("model_name").mean()
    df.index = [map_name(name) for name in df.index]
    df['sort_id'] = [id_map.get(name, 0) for name in df.index]
    df = df.sort_values(by="sort_id").drop(columns=['sort_id'])
    name = taskname_from_summary_fn(summary_fn)
    return name, df
