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

import click
import json
import networkx as nx
import numpy as np
import os
import pickle

from collections import Counter
from collections import defaultdict
from tqdm import tqdm

GRAPH_TYPES = ['ISCG','TSCG','SOG']

def parse_nxopr(pcode_asm):
    s = pcode_asm.find('(')
    e = pcode_asm.find(')')
    return tuple(pcode_asm[s+1:e].split(', ')), pcode_asm[e+1:].strip()


def parse_pcode(pcode_asm):
    '''
    Examples: 
    (register, 0x20, 4) COPY (const, 0x0, 4)
    (unique, 0x8380, 4) INT_ADD (register, 0x4c, 4) , (const, 0xfffffff0, 4)
     ---  STORE (STORE, 0x1a1, 0) , (unique, 0x8280, 4) , (register, 0x20, 4)
     ---  BRANCH (ram, 0x22128, 4)
    '''
    NOP_OPERAND = ' --- '
    dst_opr = None
    if pcode_asm.startswith(NOP_OPERAND):
        pcode_asm = pcode_asm[len(NOP_OPERAND)+1:]
    else:
        dst_opr, pcode_asm = parse_nxopr(pcode_asm)
    opc_e = pcode_asm.find(' ')
    if opc_e != -1:
        opc, pcode_asm = pcode_asm[:opc_e], pcode_asm[opc_e:].strip()
    else:
        opc, pcode_asm = pcode_asm, ""
    oprs = [] if dst_opr is None else [dst_opr,]
    while len(pcode_asm) != 0:
        src_opr, pcode_asm = parse_nxopr(pcode_asm)
        oprs.append(src_opr)
    return (opc, oprs)


def normalize_pcode_opr(opr, arch):
    if opr[0] in ['register']:
        return f'{arch}_reg', arch + '_' + '_'.join(opr)
    elif opr[0] in ['STORE', 'const']:
        return 'val', '_'.join(opr[:-1]) # omit dummy size field
    elif opr[0] in ['unique', 'NewUnique', 'ram', 'stack', 'VARIABLE']:
        return 'val', opr[0]
    else:
        raise Exception(f"Unkown operand type {opr[0]}. FULL: {opr}. ")


def normalize_pcode(pcode, arch):
    normalized_pcode = [('opc', pcode[0])]
    for opr in pcode[1]:
        normalized_pcode.append(normalize_pcode_opr(opr, arch))
    return normalized_pcode


def normalize_sng_opc(opc, arch):
    # Raw Formats:
    # ConstLong: L(%x, %d)
    # ConstDouble: D(%f, %d)
    # MemorySpace: SPACE(%x)
    # Store: "%s(%x, %d)", {'REG','MEM','STA','OTH'}, id, size
    # Project/Subpiece: PROJ(%d)
    # 1. Retain small integers.
    # 2. Register regs with arch id.
    if opc.startswith('L('):
        v = int(opc[2:opc.find(',')], 16)
        return 'val', f'L_{v}'
    elif opc.startswith('D('):
        v = float(opc[2:opc.find(',')])
        return 'val', f'D_{v}'
    elif opc.startswith('REG('):
        v = int(opc[4:opc.find(',')], 16)
        s = int(opc[opc.find(',')+1:opc.find(')')], 16)
        return f'{arch}_reg', f'{arch}_REG_{v}_{s}'
    elif opc[:3] in ['MEM', 'STA', 'OTH']:
        return 'val', opc[:3]
    elif opc.startswith('PROJ('):
        return 'opc', 'PROJ'
    else:
        return 'opc', opc


def process_nverb(gtype, nverb: list, arch):
    assert isinstance(nverb, list)
    if len(nverb) == 0:
        return []
    if gtype == 'SOG':
        ty, opc = normalize_sng_opc(nverb[0], arch)
        return [(ty, opc),]
    elif gtype == 'ISCG':
        parsed = parse_pcode(nverb[0])
        return normalize_pcode(parsed, arch)
    elif gtype == 'ACFG':
        results = []
        for inst in nverb:
            parsed = parse_pcode(inst)
            oprs = normalize_pcode(parsed, arch)
            results.extend(oprs)
        return results
    elif gtype == 'TSCG':
        if '(' == nverb[0][0]:
            ty, opc = normalize_pcode_opr(parse_nxopr(nverb[0])[0], arch)
        else:
            ty, opc = 'opc', nverb[0]
        return [(ty, opc),]
    else:
        assert "Unkown Graph Type"


def token_mapping(input_folder, output_dir, freq_mode=True):
    print("[i] Freq_mode: ", freq_mode)
    idmaps, opc_counters, opc_occurs = {}, {}, {}
    cached = {}
    num_func = 0

    # Try loading caches
    for gtype in GRAPH_TYPES:
        sub_dir = get_sub_dir(output_dir, gtype)
        counter_path = os.path.join(sub_dir, "opc_counter.json")
        occurs_path = os.path.join(sub_dir, "opc_occurs.json")
        if os.path.exists(counter_path) and os.path.exists(occurs_path):
            with open(counter_path, "r") as f:
                opc_counters[gtype] = json.load(f)
            with open(occurs_path, "r") as f:
                opc_occurs[gtype] = json.load(f)
                num_func = opc_occurs[gtype]["num_funcs"]
            cached[gtype] = True
        else:
            opc_counters[gtype], opc_occurs[gtype] = (dict([
                ('opc', Counter()),
                ('val', Counter()),
                *[(f'{arch}_reg', Counter()) for arch in ['mips', 'arm', 'x']],
            ]) for _ in range(2))
            cached[gtype] = False

    any_cached = sum(cached[gtype] for gtype in GRAPH_TYPES) != 0
    not_cached_graph_types = list(g for g in GRAPH_TYPES if not cached[g])

    if len(not_cached_graph_types) != 0:
        # Collect opc stats info
        for f_json in tqdm(os.listdir(input_folder)):
            if not f_json.endswith(".json"):
                continue

            json_path = os.path.join(input_folder, f_json)
            with open(json_path) as f_in:
                jj = json.load(f_in)

            arch = f_json.split('-')[0][:-2]
            idb_path = list(jj.keys())[0]
            j_data = jj[idb_path]
            for key in ['arch']:
                if key in j_data:
                    del j_data[key]

            # Iterate over each function
            for fva in j_data:
                for gtype in not_cached_graph_types:
                    opc_sets = defaultdict(set)
                    fva_data = j_data[fva][gtype]
                    # Iterate over each basic-block
                    for bb in fva_data['nverbs']:
                        nverb = fva_data['nverbs'][bb]
                        for ty, opc in process_nverb(gtype, nverb, arch):
                            opc_counters[gtype][ty].update([opc])
                            opc_sets[ty].add(opc)
                    for ty, opc_set in opc_sets.items():
                        opc_occurs[gtype][ty].update(opc_set)
            if not any_cached:
                num_func += len(j_data)

        # Cache results
        for gtype in not_cached_graph_types:
            sub_dir = get_sub_dir(output_dir, gtype)
            output_path = os.path.join(sub_dir, "opc_counter.json")
            with open(output_path, "w") as f:
                json.dump(opc_counters[gtype], f)
            output_path = os.path.join(sub_dir, "opc_occurs.json")
            opc_occurs[gtype]["num_funcs"] = num_func
            with open(output_path, "w") as f:
                json.dump(opc_occurs[gtype], f)

    # Assigning each word an ID
    print(f"num funcs: {num_func}")
    for gtype in GRAPH_TYPES:
        idmaps[gtype] = {'padding': 0} if gtype in ['ISCG', 'ACFG'] else {}
    for gtype, opc_cnts in opc_counters.items():
        print(f"[D] Processing {gtype}. ")
        ths = dict([
            ('opc', 55 if gtype != 'ACFG' else 50), 
            ('val', 0.01),    
            *[(f'{arch}_reg', 0.01) for arch in ['mips', 'arm', 'x']],
        ]) # thresholds
        for ty, opc_cnt in opc_cnts.items():
            if ty.endswith("_occur"):
                continue
            if not isinstance(opc_cnt, dict):
                opc_cnt = [(k,v) for k,v in opc_cnt.most_common()]
            else:
                opc_cnt = sorted(list(opc_cnt.items()), key=lambda k:k[1], reverse=True)
            mapped_cnt = 0
            tot_cnt = sum([v for _, v in opc_cnt])
            idmaps[gtype][ty] = len(idmaps[gtype])
            start_id = len(idmaps[gtype])
            for i, (k, v) in enumerate(opc_cnt):
                idmaps[gtype][k] = i + start_id
                mapped_cnt += v
                if isinstance(ths[ty], float):
                    if not freq_mode and mapped_cnt / tot_cnt > ths[ty]:
                        break
                    elif freq_mode and v / num_func < ths[ty]:
                        break
                elif isinstance(ths[ty], int) and i + 1 >= ths[ty]:
                    break
            print("[D] Found: {} mnemonics.".format(len(opc_cnt)))
            print("[D] Num of mnemonics mapped: {}".format(len(idmaps[gtype])-start_id))
        print("[D] Tot Num of mnemonics mapped: {}".format(len(idmaps[gtype])))
    return idmaps


def create_graph(fva_data):
    NUM_EDGE_TYPE = 4
    NUM_POS_ENC = 8

    nodes, edges = fva_data['nodes'], fva_data['edges']

    G = nx.MultiDiGraph()
    for node in nodes:
        G.add_node(node)
    last_edge, pos_id = (-1, -1), -1
    for edge in edges:
        if (edge[0], edge[2]) == last_edge:
            if pos_id + 1 < NUM_POS_ENC:
                pos_id += 1
        else:
            pos_id = 0
            last_edge = (edge[0], edge[2])
        n = NUM_EDGE_TYPE * pos_id + edge[2]
        G.add_edge(edge[0], edge[1], weight=n)

    nodelist = list(G.nodes())
    adj_mat = nx.to_scipy_sparse_array(
        G, nodelist=nodelist, dtype=np.int8, format='coo')
    return adj_mat, nodelist


def coo2tuple(coo_mat):
    return (coo_mat.row, coo_mat.col, coo_mat.data, *coo_mat.shape)


def create_features_matrix(node_list, fva_data, opc_dict, gtype, arch):
    """
    Create the matrix with numerical features.

    Args:
        node_list: list of basic-blocks addresses
        fva_data: dict with features associated to a function
        opc_dict: selected opcodes.

    Return
        np.matrix: Numpy matrix with selected features.
    """
    assert gtype in ['SOG', 'TSCG', 'ISCG', 'ACFG']
    if gtype in ['SOG', 'TSCG']:
        opcs = []
        for node_fva in node_list:
            assert str(node_fva) in fva_data['nverbs']
            node_data = fva_data['nverbs'][str(node_fva)]
            for ty, nopc in process_nverb(gtype, node_data, arch):
                if nopc in opc_dict:
                    opcs.append(opc_dict[nopc])
                else:
                    opcs.append(opc_dict[ty])
        asms = opcs
    elif gtype in ['ISCG', 'ACFG']:
        asms = []
        I_SIZE = 12 if gtype == 'ISCG' else 256
        PADDING = opc_dict['padding']
        assert PADDING == 0
        for node_fva in node_list:
            opcs = np.zeros(I_SIZE, dtype=np.uint16)
            node_data = fva_data['nverbs'].get(str(node_fva), [])
            for i, (ty, nopc) in enumerate(process_nverb(gtype, node_data, arch)):
                if i >= I_SIZE:
                    break
                if nopc in opc_dict:
                    opcs[i] = opc_dict[nopc]
                else:
                    opcs[i] = opc_dict[ty]
            asms.append(opcs)
    return np.array(asms, dtype=np.uint16)


def csc_matrix_to_str(csr_mat):
    return coo_matrix_to_str(csr_mat.tocoo())


def coo_matrix_to_str(cmat):
    """
    Convert the Numpy matrix in input to a Scipy sparse matrix.

    Args:
        np_mat: a Numpy matrix

    Return
        str: serialized matrix
    """
    # Custom string serialization
    row_str = ';'.join([str(x) for x in cmat.row])
    col_str = ';'.join([str(x) for x in cmat.col])
    data_str = ';'.join([str(x) for x in cmat.data])
    n_row = str(cmat.shape[0])
    n_col = str(cmat.shape[1])
    mat_str = "::".join([row_str, col_str, data_str, n_row, n_col])
    return mat_str

def process_one_file(args):
    json_path, opc_dicts, dump_str, dump_pkl = args
    with open(json_path) as f_in:
        jj = json.load(f_in)
    f_json = os.path.basename(json_path)
    arch = f_json.split('-')[0][:-2]
    idb_path = list(jj.keys())[0]
    # print("[D] Processing: {}".format(idb_path))
    str_func_dict, pkl_func_dict = defaultdict(dict), defaultdict(dict)
    j_data = jj[idb_path]
    for key in ['arch', 'failed_functions', 'overrange_functions', 'underrange_functions']:
        if key in j_data:
            del j_data[key]
    # Iterate over each function
    for fva in j_data:
        for gtype in GRAPH_TYPES:
            fva_data = j_data[fva][gtype]
            g_coo_mat, nodes = create_graph(fva_data)
            f_list = create_features_matrix(
                nodes, fva_data, opc_dicts[gtype], gtype, arch)
            if not fva.startswith("0x"):
                fva = hex(int(fva, 10))
            if dump_str:
                str_func_dict[gtype][fva] = {
                    'graph': coo_matrix_to_str(g_coo_mat),
                    'opc': f_list
                }
            if dump_pkl:
                pkl_func_dict[gtype][fva] = {
                    'graph': coo2tuple(g_coo_mat),
                    'opc': f_list
                }
    return idb_path, str_func_dict, pkl_func_dict


def create_functions_dict(input_folder, opc_dicts, dump_str, dump_pkl):
    """
    Convert each function into a graph with BB-level features.

    Args:
        input_folder: a folder with JSON files from IDA_acfg_disasm
        opc_dict: dictionary that maps most common opcodes to their ranking.

    Return
        dict: map each function to a graph and features matrix
    """
    str_func_dict = {g:defaultdict(dict) for g in GRAPH_TYPES} if dump_str else {}
    pkl_func_dict = {g:defaultdict(dict) for g in GRAPH_TYPES} if dump_pkl else {}
    args = []
    for f_json in os.listdir(input_folder):
        if not f_json.endswith(".json"):
            continue
        json_path = os.path.join(input_folder, f_json)
        args.append((json_path, opc_dicts, dump_str, dump_pkl))
    for idb_path, str_func_one, pkl_func_one in \
        tqdm(map(process_one_file, args), total=len(args)):
        if dump_str:
            for gtype, data in str_func_one.items():
                str_func_dict[gtype][idb_path] = data
        if dump_pkl:
            for gtype, data in pkl_func_one.items():
                pkl_func_dict[gtype][idb_path] = data
    return str_func_dict, pkl_func_dict


def get_sub_dir(output_dir, gtype, dataset=None):
    if dataset is not None:
        sub_dir = os.path.join(output_dir, f'pcode_{gtype.lower()}', dataset)
    else:
        sub_dir = os.path.join(output_dir, f'pcode_{gtype.lower()}')
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    return sub_dir


@click.command()
@click.option('-i', '--input-dir', required=True,
              help='A directory that contains JSON-formed SOG/ISCG/TSCG/ACFG. ')
@click.option('--training', required=False, is_flag=True,
              help='In training mode, this script generates a new token mapping. ')
@click.option('--freq-mode', default=False, is_flag=True,
              help='In frequency mode, the number of tokens to map is determined by the frequency of occurrence of the token, rather than by a predefined number/ratio. ')
@click.option('-d', '--opcodes-json',
              default="opcodes_dict.json",
              help='Token mapping result file name. ')
@click.option('-o', '--output-dir', required=True,
              help='Output directory. The output path is formed as output-dir/graph-type/dataset. ')
@click.option('-s', '--dataset', required=True,
              help='The name of dataset. Used as part of the output path. ')
@click.option('-f', '--out_format', default='pkl', required=False,
              help='Output format ("json", "pkl" or "both"). ')
def main(input_dir, training, freq_mode, opcodes_json, output_dir, dataset, out_format):
    # Create output directory if it doesn't exist
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if training:
        # Conduct token mapping and save results. 
        opc_dicts = token_mapping(
            input_dir, output_dir, freq_mode)
        for gtype in GRAPH_TYPES:
            sub_dir = get_sub_dir(output_dir, gtype)
            output_path = os.path.join(sub_dir, opcodes_json)
            with open(output_path, "w") as f_out:
                json.dump(opc_dicts[gtype], f_out)
    else:
        # Load previous token mapping results. 
        opc_dicts = {}
        for gtype in GRAPH_TYPES:
            sub_dir = get_sub_dir(output_dir, gtype)
            json_path = os.path.join(sub_dir, opcodes_json)
            if not os.path.isfile(json_path):
                print("[!] Error loading {}".format(json_path))
                return
            with open(json_path) as f_in:
                opc_dict = json.load(f_in)
            opc_dicts[gtype] = opc_dict

    # Two 
    dump_str = out_format == "json" or out_format == "both"
    dump_pkl = out_format == "pkl" or out_format == "both"
    str_dict, pkl_dict = create_functions_dict(
        input_dir, opc_dicts, dump_str, dump_pkl)
    for gtype, g_str_dict in str_dict.items():
        o_json = "graph_func_dict_opc_{}.json".format(freq_mode)
        sub_dir = get_sub_dir(output_dir, gtype, dataset)
        output_path = os.path.join(sub_dir, o_json)
        with open(output_path, 'w') as f_out:
            json.dump(g_str_dict, f_out)
    for gtype, g_pkl_dict in pkl_dict.items():
        o_json = "graph_func_dict_opc_{}.pkl".format(freq_mode)
        sub_dir = get_sub_dir(output_dir, gtype, dataset)
        output_path = os.path.join(sub_dir, o_json)
        with open(output_path, 'wb') as f_out:
            pickle.dump(g_pkl_dict, f_out)


if __name__ == '__main__':
    main()
