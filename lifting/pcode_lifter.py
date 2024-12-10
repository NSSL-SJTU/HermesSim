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
# This script acts as a helper to lift binary functions in batch             #
#                                                   with the gsat tool.      #
##############################################################################


import argparse
import sys
import os
import subprocess
import json
import pandas as pd
from os.path import basename, join
from multiprocessing import Pool

from tqdm import tqdm

ACFG_POSTFIX = "_acfg_disasm.json"
CFG_SUMMARY_POSTFIX = "_cfg_summary.json"
GSAT_BIN_PATH = "bin/gsat-1.0.jar"
ROOT_PATH = "."
NPROC = 1

AR_OBJ_MAP = {
    ## Dataset-2
    "libmicrohttpd.a": "libmicrohttpd_la-connection.o",
    "libtomcrypt.a": "aes.o",
}


def idb_path_to_binary_path(idb_path):
    assert idb_path[-4:] == ".i64" or idb_path[-4:] == ".idb"
    assert idb_path[:5] == "IDBs/"
    return "binaries" + idb_path[4:-4]


def extract_time(out: str):
    sig_s = "Time for extraction: "
    sig_e = "secs."
    out = out[out.find(sig_s) + len(sig_s) :]
    out = out[: out.find(sig_e)]
    return float(out)


def do_one_extractor(
    cfg_summary_fp, graph_type, verbose, output_dir, firmware_info=None
):
    with open(cfg_summary_fp, "r") as f:
        idb_fp = list(json.load(f).keys())[0]
    bin_fp = idb_path_to_binary_path(idb_fp)
    bin_base = os.path.basename(bin_fp)
    output_name = bin_base + ACFG_POSTFIX
    output_fp = os.path.join(output_dir, output_name)

    if firmware_info is not None:
        language_id, load_addr = firmware_info
        bin_selector = f"-m binary -l {language_id} -b {load_addr}"
    elif bin_fp.endswith(".a"):
        obj_name = None
        for name, obj in AR_OBJ_MAP.items():
            if bin_fp.endswith(name):
                obj_name = obj
                break
        if obj_name is None:
            return -2, bin_fp
        bin_selector = f"-m ar-obj -af {obj_name}"
    else:
        bin_selector = "-m elf"
    enable_assert = "-ea"
    # enable_assert = ""
    prefer_raw = "-opt 0"
    heap_config = "-Xmx16G -XX:+UseCompressedOops"
    cmd = f"java {enable_assert} {heap_config} -jar {GSAT_BIN_PATH} pcode-extractor-v2 {bin_selector} \
        -f {bin_fp} -c {cfg_summary_fp} -of {graph_type} -v {verbose}\
        {prefer_raw} -o {output_fp}"
    proc = subprocess.Popen(cmd, shell=True, text=True, stdout=subprocess.PIPE)
    out, _ = proc.communicate()
    code = proc.returncode
    if (code != 0) or (not os.path.exists(output_fp)):
        print("[=] " + cmd)
        code = code if code is not None else -1
        return code, bin_fp
    cost = extract_time(out)
    # print(f"[*] Saving {output_fp}")
    return None, cost


def do_one_extractor_wrap(args):
    return do_one_extractor(*args)


def get_firmware_info(info_csv):
    info = pd.read_csv(info_csv)
    info_map = {}
    for idx, r in info.iterrows():
        fn, arch_str, load_addr = r["file_name"], r["arch_str"], r["load_addr"]
        info_map[fn] = (arch_str, load_addr)
    return info_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Pcode-Lifter",
        description="A helper script to lift binary functions into various Pcode-based representations, including SOG, ISCG, TSCG, and ACFG. ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--cfg_summary",
        default="./dbs/Dataset-1/cfg_summary/testing",
        help="Path to the input summary files, which is needed by the lifter. ",
    )

    parser.add_argument(
        "--output_dir",
        default="./dbs/Dataset-1/features/testing/pcode_raw_Dataset-1_testing",
        help="Path to the output feature directory. ",
    )

    parser.add_argument(
        "--graph_type",
        default="ALL",
        choices=["ALL", "SOG", "ACFG"],
        help="The target type of graph to lift into. ALL is for SOG, ISCG, TSCG, and ACFG. ",
    )

    parser.add_argument(
        "--verbose",
        default="1",
        choices=["0", "1"],
        help="Whether the output files should contain verbose info of instructions/nodes. ",
    )

    # For safety, use only one process by default to avoid exhausting cpu / memory resources.
    parser.add_argument("--nproc", default="1", help="Number of processes to use. ")

    parser.add_argument(
        "--firmware_info", default=None, help="Number of processes to use. "
    )

    args = parser.parse_args()

    cfg_summary_dir, output_dir, graph_type, verbose = (
        getattr(args, arg)
        for arg in ["cfg_summary", "output_dir", "graph_type", "verbose"]
    )

    firmware_info = None
    if args.firmware_info is not None:
        firmware_info = get_firmware_info(args.firmware_info)

    NPROC = int(args.nproc)

    os.chdir(ROOT_PATH)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    summary_files = os.listdir(cfg_summary_dir)

    processed = []
    for fn in os.listdir(output_dir):
        if not fn.endswith(ACFG_POSTFIX):
            continue
        fn = fn[: -len(ACFG_POSTFIX)]
        if firmware_info is not None:
            # Handle filename mismatch between these two datasets. 
            fn = "-".join(fn.split("-")[1:])
        one = 0
        for idx, summary_fn in enumerate(summary_files):
            if summary_fn[: -len(CFG_SUMMARY_POSTFIX)] == fn:
                processed.append(idx)
                one += 1
                if one > 1:
                    print(f"Error: {fn} <-> {summary_fn}")
    processed = sorted(processed, reverse=True)
    print(f"{len(processed)} samples have been processed. ")
    for idx in processed:
        del summary_files[idx]

    pbar = tqdm(total=len(summary_files))
    p = Pool(NPROC)
    failed_list = []
    time_cost = []
    for code, info in p.imap_unordered(
        do_one_extractor_wrap,
        [
            (
                join(cfg_summary_dir, summary_fn),
                graph_type,
                verbose,
                output_dir,
                firmware_info[summary_fn[: -len(CFG_SUMMARY_POSTFIX)]]
                if firmware_info is not None
                else None,
            )
            for summary_fn in summary_files
        ],
    ):
        if code is not None:
            bin_fp = info
            failed_list.append(bin_fp)
            print("==========================================")
            print(f"Fail to process {bin_fp} (code: {code}). ")
            print("==========================================")
        else:
            time_cost.append(info)
        pbar.update(1)
    p.close()
    pbar.close()
    print(failed_list)
    print("Failed Count: ", len(failed_list))
    print("Tot. Extraction Time: %.2f" % (sum(time_cost)))
