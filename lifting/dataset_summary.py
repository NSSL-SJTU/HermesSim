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
#============================================================================#
# Summarize the dataset infos for ease of lifting                            #
##############################################################################


import argparse
import json
import pandas
from os.path import join, basename
from tqdm import tqdm


def get_bin_name(idb_path):
    bin_name = basename(idb_path)
    assert bin_name.endswith(".i64")
    return bin_name[:-4]


def get_acfg_features_json(acfg_jsons_folder, bin_name):
    json_fp = join(acfg_jsons_folder, bin_name + "_acfg_features.json")
    with open(json_fp, "r") as f:
        obj = json.load(f)
    return obj


def get_func_info(row):
    assert row['fva'] == row['start_ea']
    return {
        "start_ea": row['start_ea'],
        "func_name": row['func_name'],
        "nodes": [],
        "edges": None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Dataset summary files generator',
        description='A helper tool to generate summary files for the pcode-lifer. ',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--cfg_summary", default="dbs/Dataset-1/cfg_summary/testing",
                        help="Path to the output summary directory. ")

    parser.add_argument("--dataset_info_csv", default="dbs/Dataset-1/testing_Dataset-1.csv",
                        help="Path to the {training, validation, testing}_Dataset-*.csv files that describe all functions contained in the dataset. ")

    parser.add_argument("--cfgs_folder", default="dbs/Dataset-1/features/testing/acfg_features_Dataset-1_testing",
                        help="Path to the input folder that contains description/features of all functions in the dataset. ")

    args = parser.parse_args()

    summary_folder, ds_info_csv, cfgs_folder = (getattr(
        args, arg) for arg in ['cfg_summary', 'dataset_info_csv', 'cfgs_folder'])

    summary = {}
    df = pandas.read_csv(ds_info_csv)
    for idx, row in tqdm(df.iterrows(), total=df.size):
        idb_path = row['idb_path']
        finfo = get_func_info(row)
        if summary.get(idb_path, None) is None:
            summary[idb_path] = [finfo]
        else:
            summary[idb_path].append(finfo)

    for idb_path, finfos in tqdm(summary.items()):
        bin_name = get_bin_name(idb_path)
        acfg_feats = get_acfg_features_json(cfgs_folder, bin_name)[idb_path]
        for finfo in finfos:
            fva = finfo["start_ea"]
            finfo["edges"] = acfg_feats[fva]["edges"]
            for bb_va, bb_feats in acfg_feats[fva]["basic_blocks"].items():
                bb_va_hex = hex(int(bb_va))
                finfo["nodes"].append([bb_va_hex, bb_feats["bb_len"]])
        summary_fn = join(summary_folder, bin_name + "_cfg_summary.json")
        with open(summary_fn, "w") as f:
            json.dump({idb_path: finfos}, f)
        summary[idb_path] = None
