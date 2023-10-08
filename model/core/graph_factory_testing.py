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

import math
import numpy as np
import pandas as pd

from .graph_factory_base import *
from tqdm import tqdm

import logging
log = logging.getLogger('gnn')


class GraphFactoryTesting(GraphFactoryBase):

    def __init__(self, func_info_path, feat_path, batch_size, used_subgraphs, edge_feature_dim):
        """
            Args:
                feat_path: JSON file with function features
                batch_size: size of the batch for each iteration
        """
        super(GraphFactoryTesting, self).__init__(
            GraphFactoryMode.VALIDATE, feat_path, batch_size, used_subgraphs, edge_feature_dim)

        # Load positive and negative function pairs
        log.debug("Reading {}".format(func_info_path))
        self._func_info = pd.read_csv(func_info_path)

        # Number of positive or negative function pairs
        self._num_funcs = self._func_info.shape[0]
        self._group_ids = np.array([
            r['group'] for _, r in self._func_info.iterrows()])
        log.info("Tot num funcs (validation): {}".format(self._num_funcs))
        self._num_batches_in_epoch = math.floor(
            self._num_funcs / self._batch_size)
        self._last_batch_size = self._num_funcs - \
            self._num_batches_in_epoch * self._batch_size
        log.info("Num batches in epoch (validation): {}".format(
            self._num_batches_in_epoch))

    def get_group_ids(self):
        return self._group_ids

    def _pairs(self):
        """Yields batches of pair data."""
        numIter = self._num_funcs
        log.info("(Re-)initializing the iterators")
        iterator_f = self._func_info.iterrows()
        batch_graphs = list()
        batch_features = list()
        # Every iter process two pairs
        for idx in tqdm(range(numIter), total=numIter):
            finfo = next(iterator_f)[1]

            f = self._fdict[finfo['idb']][finfo['fva']]

            batch_graphs.append(f['graph'])
            batch_features.append(f[self._features_type])

            if (idx+1) % (self._batch_size*2) != 0 and idx != numIter - 1:
                continue

            # Pack everything in a graph data structure
            packed_graphs = self._pack_batch(batch_graphs,
                                             batch_features)
            yield packed_graphs

            batch_graphs = list()
            batch_features = list()

    def _triplets(self):
        """ Yields batches of triplet data. For training only."""
        pass
