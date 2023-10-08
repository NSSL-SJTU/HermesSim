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

import torch

def euclidean_distance(x, y):
    """Compute the squared Euclidean distance."""
    return torch.sum((x - y)**2, dim=-1)

def pairwise_cosine_similarity(x, y, eps: float=1e-8):
    """Compute the cosine similarity between x and y.

    Args:
      x: NxD float tensor.
      y: MxD float tensor.

    Returns:
      sim_mt: NxM float tensor, the pairwise cosine similarity.
    """
    x_n, y_n = x.norm(2, dim=1).unsqueeze(1), y.norm(2, dim=1).unsqueeze(1)
    x_norm = x / torch.max(x_n, eps * torch.ones_like(x_n))
    y_norm = y / torch.max(y_n, eps * torch.ones_like(y_n))
    sim_mt = torch.mm(x_norm, y_norm.transpose(0, 1))
    return sim_mt

def compute_similarity(loss_type: str, x, y):
    """Compute the distance between x and y vectors.

    The distance will be computed based on the training loss type.

    Args:
      config: a config dict.
      x: [n_examples, feature_dim] float tensor.
      y: [n_examples, feature_dim] float tensor.
      or 
      x: [n_examples, feature_dim] float tensor.
      y: [feature_dim] float tensor.

    Returns:
      dist: [n_examples] float tensor.

    Raises:
      ValueError: if loss type is not supported.
    """
    if loss_type == 'margin':
        # similarity is negative distance
        return -euclidean_distance(x, y)
    elif loss_type == 'cosine':
        return torch.cosine_similarity(x, y)
    else:
        raise ValueError('Unknown loss type %s' % loss_type)
