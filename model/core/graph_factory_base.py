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

import numpy as np
from tqdm import tqdm
from enum import Enum
from multiprocessing import Pool

import logging
log = logging.getLogger('gnn')


def pkl_exist(fp):
    return os.path.exists(fp + ".pkl")


def load_pkl(fp):
    with open(fp + ".pkl", "rb") as f:
        return pickle.load(f)


def dump_pkl(obj, fp):
    with open(fp + ".pkl", "wb") as f:
        pickle.dump(obj, f)


def filter_edges(args):
    idb, idb_data, used_subgraphs = args
    for fva, f_data in idb_data.items():
        g = f_data['graph']
        row, col, data, n_row, n_col = g
        mask = [d in used_subgraphs for d in (((data-1) & 3)+1)]
        row, col, data = row[mask], col[mask], data[mask]
        f_data['graph'] = (row, col, data, n_row, n_col)
    return idb, idb_data


class GraphFactoryMode(Enum):
    TRAIN = 0
    VALIDATE = 1
    TEST = 2


class GraphFactoryBase(object):
    """Base class for all the graph similarity learning datasets.

    This class defines some common interfaces a graph similarity dataset can have,
    in particular the functions that creates iterators over pairs and triplets.
    """
    SUBGRAPH_IDS = [1, 2, 3, 4]

    def __init__(self, mode, feat_path, batch_size, used_subgraphs, edge_feature_dim) -> None:
        self._mode = mode
        self._batch_size = batch_size
        log.info("Batch size ({}): {}".format(mode, self._batch_size))

        self._features_type = 'opc'

        self._used_subgraphs = used_subgraphs
        self._need_g_filter = set(
            self.SUBGRAPH_IDS) != set(self._used_subgraphs)
        used_subg_str = (
            '_' + '_'.join([str(i) for i in used_subgraphs])) if self._need_g_filter else ''

        # Load function features
        log.debug("Loading {}".format(feat_path))
        cache_path = feat_path[:-5]

        filter_g_cache_path = cache_path + used_subg_str
        if pkl_exist(filter_g_cache_path):
            self._fdict = load_pkl(filter_g_cache_path)
        else:
            if not pkl_exist(cache_path):
                raise FileNotFoundError(f"{cache_path}.pkl does not exist. ")
            self._fdict = load_pkl(cache_path)
            log.info(f"Cache data {cache_path} used. ")

            if self._need_g_filter:
                self._filter_edges()
                dump_pkl(self._fdict, filter_g_cache_path)

        # Inferring feature_dim
        self._edge_feature_dim = edge_feature_dim

    def _filter_edges(self):
        POOL_SIZE = 12
        log.info(f"Filter Edges with pool size {POOL_SIZE}")
        with Pool(POOL_SIZE) as p:
            args = [(idb, idb_data, self._used_subgraphs)
                    for idb, idb_data in self._fdict.items()]
            for idb, idb_data in tqdm(p.imap_unordered(filter_edges, args), total=len(args)):
                self._fdict[idb] = idb_data

    def _pack_batch(self, graphs, features):
        from_idx = []
        to_idx = []
        node_features = []
        graph_idx = []
        attr = []

        n_total_nodes = 0
        n_total_edges = 0
        for i, d in enumerate(zip(graphs, features)):
            g, f = d[0], d[1]

            # g: ([row_indexs], [col_indexs], [data], n_rows, n_cols)
            # f: scipy sparse matrix / np array
            row, col, data, n_row, n_col = g

            if not isinstance(f, np.ndarray):
                f = f.toarray()

            n_nodes = f.shape[0]
            n_edges = len(row)

            # shift the node indices for the edges
            from_idx.append(row + n_total_nodes)
            to_idx.append(col + n_total_nodes)
            attr.append(data)
            node_features.append(f)
            graph_idx.append(np.ones(n_nodes, dtype=np.int32)*i)

            n_total_nodes += n_nodes
            n_total_edges += n_edges

        if node_features[0].dtype.kind == 'f':
            node_features = np.concatenate(node_features, axis=0)
        else:
            node_features = np.concatenate(
                node_features, axis=0, dtype=np.int32)
        from_idx = np.concatenate(from_idx, axis=0)
        to_idx = np.concatenate(to_idx, axis=0)
        graph_idx = np.concatenate(graph_idx, axis=0)
        attr = np.concatenate(attr, axis=0) - 1 # let them start from 0

        batch_size = len(graphs)

        node_features = torch.tensor(node_features)
        edge_feat = torch.tensor(attr, dtype=torch.int)
        graph_idx = torch.tensor(graph_idx, dtype=torch.long)
        edge_index = torch.tensor([from_idx, to_idx], dtype=torch.long)
        return node_features, edge_index, edge_feat, graph_idx, batch_size

    def _triplets(self):
        """Create an iterator over triplets."""
        pass

    def _pairs(self):
        """Create an iterator over pairs."""
        pass

    def _batch_triplets(self):
        """Create an iterator over a batch of triplets."""
        pass

    def pairs(self):
        return self._pairs()

    def triplets(self):
        return self._triplets()

    def batch_triplets(self):
        return self._batch_triplets()
