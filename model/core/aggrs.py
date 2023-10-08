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


from typing import Optional

import torch

from torch_geometric.nn import inits
from torch_geometric.utils import softmax

from torch import nn
from torch import Tensor
from torch.nn import Parameter as Param, LayerNorm

from torch_geometric.nn import (
    Aggregation,
    Linear,
    SumAggregation,
    SoftmaxAggregation,
    SetTransformerAggregation
)
from torch_geometric.nn import SetTransformerAggregation

from .mlp import *


class GatedAggr(torch.nn.Module):
    def __init__(self, n_node_trans, n_hidden_channels, n_out_channels, gated) -> None:
        super().__init__()
        self.lin = nn.Sequential(
            *[layer for _ in range(n_node_trans)
                for layer in (
                Linear(-1, n_hidden_channels * (2 if gated else 1),
                       **LINEAR_INIT), nn.ReLU(inplace=True))][:-1])
        self.agg = SumAggregation()
        self.graph_tran = Linear(
            n_hidden_channels, n_out_channels, **LINEAR_INIT)
        self._gated = gated
        self._n_hidden_channels = n_hidden_channels
        self._n_out_channels = n_out_channels

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.graph_tran.reset_parameters()

    def forward(self, x, graph_idx, batch_size: int):
        x = self.lin(x)
        if self._gated:
            gates = torch.special.expit(x[:, :self._n_hidden_channels])
            x = x[:, self._n_hidden_channels:] * gates
        g_x = self.agg(x, graph_idx, dim_size=batch_size)
        g_x = g_x.reshape(batch_size, -1)
        return self.graph_tran(g_x)


class SoftmaxAggr(torch.nn.Module):
    def __init__(self, out_channels, n_graph_trans=0) -> None:
        super().__init__()
        hidden_channels = out_channels

        self.lin = Linear(-1, hidden_channels, **LINEAR_INIT)
        self.graph_tran = MLP(n_graph_trans, out_channels)
        self.agg = SoftmaxAggregation(learn=True, channels=hidden_channels)
        self._hidden_channels = hidden_channels

    @torch.jit.ignore
    def reset_parameters(self):
        self.lin.reset_parameters()
        self.graph_tran.reset_parameters()

    def forward(self, x, graph_idx, batch_size: int):
        x = self.lin(x).relu_()
        agg_x = self.agg(x, graph_idx, dim_size=batch_size)
        g_x = agg_x.reshape(batch_size, -1)
        g_x = self.graph_tran(g_x)
        return g_x


class Set2Set(torch.nn.Module):
    r"""
    This aggregator is copied from torch_geometric and then modified by SJTU NSSL Lab. 
    The Set2Set aggregation operator based on iterative content-based
    attention, as described in the `"Order Matters: Sequence to sequence for
    Sets" <https://arxiv.org/abs/1511.06391>`_ paper

    .. math::
        \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})

        \alpha_{i,t} &= \mathrm{softmax}(\mathbf{x}_i \cdot \mathbf{q}_t)

        \mathbf{r}_t &= \sum_{i=1}^N \alpha_{i,t} \mathbf{x}_i

        \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,

    where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
    the dimensionality as the input.

    Args:
        in_channels (int): Size of each input sample.
        processing_steps (int): Number of iterations :math:`T`.
        **kwargs (optional): Additional arguments of :class:`torch.nn.LSTM`.
    """

    def __init__(self, out_channels: int, processing_steps: int, **kwargs):
        super().__init__()
        hidden_channels = out_channels // 2
        self.lin = Linear(-1, hidden_channels, **LINEAR_INIT)
        self.in_channels = hidden_channels
        self.out_channels = 2 * hidden_channels
        self.processing_steps = processing_steps
        self.lstm = torch.nn.LSTM(self.out_channels, hidden_channels, **kwargs)
        self.sum_agg = SumAggregation()
        self.reset_parameters()
        self._hidden_channels = hidden_channels

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def forward(self, x: Tensor, index: Tensor, dim_size: int) -> Tensor:
        ptr = None

        x = self.lin(x)
        h = (torch.zeros(self.lstm.num_layers, dim_size, self._hidden_channels).to(x.device),
             torch.zeros(self.lstm.num_layers, dim_size, self._hidden_channels).to(x.device))
        q_star = x.new_zeros(dim_size, self.out_channels)

        for _ in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(dim_size, self.in_channels)
            e = (x * q[index]).sum(dim=-1, keepdim=True)
            a = softmax(e, index, ptr, dim_size)
            r = self.sum_agg(a * x, index, ptr, dim_size)
            q_star = torch.cat([q, r], dim=-1)

        return q_star

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')


class SetTransformerAggr(torch.nn.Module):
    def __init__(self, **args):
        super().__init__()
        self._in_channel = args['channels']
        self.lin = Linear(-1, self._in_channel, **LINEAR_INIT)
        self.set_trans = SetTransformerAggregation(**args)

    def reset_parameters(self):
        reset_subnet(self.lin)

    def forward(self, x, graph_idx, batch_size: int):
        x = self.lin(x).relu_()
        x = self.set_trans(x, graph_idx, dim_size=batch_size)
        x = x.view(batch_size, -1)
        return x


class AdapMultiSoftmaxAggrV2(torch.nn.Module):
    def __init__(self, num_querys, hidden_channels, n_node_trans=1, n_agg_trans=0, q_scale=1., out_method='lin') -> None:
        super().__init__()
        multi_hidden_channels = hidden_channels * num_querys
        self.nlin = \
            MLP(n_node_trans, multi_hidden_channels, True)
        if out_method == 'lin':
            self.agglin = MLP(n_agg_trans, multi_hidden_channels)
        elif out_method == 'mh':
            self.agglin = torch.nn.ModuleList([
                MLP(n_agg_trans, hidden_channels)
                for _ in range(num_querys)])
        self.l_norm = LayerNorm(hidden_channels, elementwise_affine=False)
        self.querys = Param(torch.Tensor(num_querys, hidden_channels))
        self.sum_agg = SumAggregation()
        self._num_querys = num_querys
        self._hidden_channels = hidden_channels
        self._out_method = out_method
        self.querys.data.fill_(q_scale)

    @torch.jit.ignore
    def reset_parameters(self):
        self.nlin.reset_parameters()
        self.querys.data.fill_(1.)
        if self._out_method == 'lin':
            self.agglin.reset_parameters()
        elif self._out_method == 'mh':
            reset_subnet(self.agglin)

    def forward(self, x, graph_idx, batch_size: int):
        x = self.nlin(x).reshape(-1, self._num_querys, self._hidden_channels)
        q = self.l_norm(x) * self.querys.unsqueeze(0)
        a = softmax(q, graph_idx, num_nodes=batch_size, dim=0)
        r = self.sum_agg(a * x, graph_idx, dim_size=batch_size, dim=0)
        ## r: batch_size x num_querys x self._hidden_channels
        if isinstance(self.agglin, torch.nn.ModuleList):
            g_x = torch.zeros_like(r)
            for i, alin in enumerate(self.agglin):
                g_x[:, i, :] = alin(r[:, i, :])
            return g_x.reshape(batch_size, -1)
        else:
            g_x = r.reshape(batch_size, -1)
            return self.agglin(g_x)
