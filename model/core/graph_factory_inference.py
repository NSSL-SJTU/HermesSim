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
import pandas as pd

from .graph_factory_base import *
from tqdm import tqdm

import logging
log = logging.getLogger('gnn')


class GraphFactoryInference(GraphFactoryBase):

    def __init__(self, func_path, feat_path, batch_size, used_subgraphs, edge_feature_dim):
        """
            Args:
                func_path: CSV file with function pairs
                feat_path: JSON file with function features
                batch_size: size of the batch for each iteration
        """
        super(GraphFactoryInference, self).__init__(
            GraphFactoryMode.TEST, feat_path, batch_size, used_subgraphs, edge_feature_dim)

        # Load function pairs
        log.debug("Reading {}".format(func_path))
        self._func = pd.read_csv(func_path)

        # Init iter infos.
        self._num_funcs = self._func.shape[0]
        log.info("Num funcs (inference): {}".format(self._num_funcs))
        self._num_batches_in_epoch = math.floor(
            self._num_funcs // self._batch_size)
        self._last_batch_size = self._num_funcs - \
            self._num_batches_in_epoch * self._batch_size
        log.info("Num batches in epoch (inference): {}".format(
            self._num_batches_in_epoch))

    def _pairs(self):
        """Yields batches of pair data."""
        log.info("(Re-)initializing the iterators")
        iterator = self._func.iterrows()
        for ii in range(2):
            if ii == 0:
                numIter = self._num_batches_in_epoch
                batch_iter = tqdm(range(numIter), total=numIter)
                pair_iter = range(self._batch_size)
            else:
                numPair = self._last_batch_size
                batch_iter = range(1)
                pair_iter = tqdm(range(numPair), total=numPair)
            for _ in batch_iter:
                batch_graphs = list()
                batch_features = list()
                for _ in pair_iter:
                    r = next(iterator)[1]
                    g = self._fdict[r['idb_path']][r['fva']]
                    batch_graphs.append(g['graph'])
                    batch_features.append(g[self._features_type])

                if len(batch_graphs) == 0:
                    continue

                # Pack everything in a graph data structure
                packed_graphs = self._pack_batch(batch_graphs, batch_features)
                yield packed_graphs

    def _triplets(self):
        """ Yields batches of triplet data. For training only."""
        pass
