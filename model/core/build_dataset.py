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
#----------------------------------------------------------------------------#
#  This implementation contains code from:                                   #
#  https://github.com/Cisco-Talos/binary_function_similarity/tree/main       #
#                                                                            #
#  Licensed under MIT License                                                #
#                                                                            #
#  Copyright (c) 2019-2022 Cisco Talos                                       #
#                                                                            #
#  Permission is hereby granted, free of charge, to any person obtaining     #
#  a copy of this software and associated documentation files (the           #
#  "Software"), to deal in the Software without restriction, including       #
#  without limitation the rights to use, copy, modify, merge, publish,       #
#  distribute, sublicense, and/or sell copies of the Software, and to        #
#  permit persons to whom the Software is furnished to do so, subject to     #
#  the following conditions:                                                 #
#                                                                            #
#  The above copyright notice and this permission notice shall be            #
#  included in all copies or substantial portions of the Software.           #
#                                                                            #
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,           #
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF        #
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND                     #
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE    #
#  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION    #
#  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION     #
#  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.           #
#----------------------------------------------------------------------------#
#  This implementation contains code from:                                   #
#  https://github.com/deepmind/deepmind-research/blob/master/                #
#    graph_matching_networks/graph_matching_networks.ipynb                   #
#    licensed under Apache License 2.0                                       #
##############################################################################


from .graph_factory_testing import GraphFactoryTesting
from .graph_factory_inference import GraphFactoryInference
from .graph_factory_training import GraphFactoryTraining

import logging
log = logging.getLogger('gnn')

import torch
from torch.utils import data

class DatasetWrap(data.IterableDataset):
    def __init__(self, training_gen, mode):
        self.gen = training_gen
        self.mode = mode
    
    def __iter__(self):
        if self.mode == 'pair':
            bg = self.gen.pairs()
        elif self.mode == 'triplet':
            bg = self.gen.triplets()
        elif self.mode.startswith('batch'):
            bg = self.gen.batch_triplets()
        else:
            raise Exception(f'Unkown train mode {self.gen}')
        return bg

    def step(self):
        return self.gen.step()

    def reset_seed(self, seed):
        return self.gen.reset_seed(seed)

def build_train_validation_generators(config):
    """Utility function to build train and validation batch generators.

    Args
      config: global configuration
    """
    training_gen = DatasetWrap(GraphFactoryTraining(
        func_path=config['training']['df_train_path'],
        feat_path=config['training']['features_train_path'],
        batch_size=config['training']['batch_size'],
        max_num_nodes=config['training']['max_num_nodes'],
        max_num_edges=config['training']['max_num_edges'],
        n_sim_funcs=config['training']['n_sim_funcs'], 
        used_subgraphs=config['used_subgraphs'], 
        edge_feature_dim=config['edge_feature_dim'],
    ), config['training']['mode'])

    validation_gen = GraphFactoryTesting(
        func_info_path=config['validation']['func_info_csv_path'],
        feat_path=config['validation']['features_validation_path'],
        batch_size=config['batch_size'],
        used_subgraphs=config['used_subgraphs'], 
        edge_feature_dim=config['edge_feature_dim'],
    )

    return training_gen, validation_gen


def build_testing_generator(config, csv_path):
    """Build a batch_generator from the CSV in input.

    Args
      config: global configuration
      csv_path: CSV input path
    """
    testing_gen = DatasetWrap(GraphFactoryInference(
        func_path=csv_path,
        feat_path=config['testing']['features_testing_path'],
        batch_size=config['batch_size'],
        used_subgraphs=config['used_subgraphs'], 
        edge_feature_dim=config['edge_feature_dim'],
    ), 'pair')

    return testing_gen



