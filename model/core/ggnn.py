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

from torch import Tensor
from torch import nn
from torch_geometric.nn import Linear
from torch.nn import Parameter as Param

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.nn.aggr.utils import (
    MultiheadAttentionBlock, 
)

from .mlp import *

class GatedGraphConv(MessagePassing):
    r"""The gated graph convolution operator from the `"Gated Graph Sequence
    Neural Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    Args:
        out_channels (int): Size of each output sample.
        num_layers (int): The sequence length :math:`L`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`

    """

    def __init__(self, out_channels: int, n_message_net_layers: int,
                 aggr: str = 'add', aggr_kwargs={}, bias: bool = True, **kwargs):
        super().__init__(aggr=aggr, aggr_kwargs=aggr_kwargs, **kwargs)

        meg_channels = out_channels * 2

        self.message_net = MLP(n_message_net_layers, meg_channels, acti_fini=False)
        self.rev_message_net = MLP(n_message_net_layers, meg_channels, acti_fini=False)
        self.rnn = torch.nn.GRUCell(meg_channels, out_channels, bias=bias)

        self.out_channels = out_channels
        self.reset_parameters()

    @torch.jit.ignore
    def reset_parameters(self):
        self.message_net.reset_parameters()
        self.rev_message_net.reset_parameters()
        self.rnn.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_feat: Tensor) -> Tensor:
        """"""
        nnodes = x.shape[0]
        # propagate_type: (x: Tensor, edge_feat: Tensor)
        m = self.propagate(edge_index, x=x, size=(nnodes, nnodes), edge_feat=edge_feat)
        x = self.rnn(m, x)

        return x

    def message(self, x_i: Tensor, x_j: Tensor, edge_feat: Tensor):
        return torch.stack(
            (self.message_net(torch.cat((x_i, x_j, edge_feat), dim=1)),
             self.rev_message_net(torch.cat((x_j, x_i, edge_feat), dim=1))))

    def aggregate(self, inputs: Tensor, edge_index_i: Tensor, edge_index_j: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        return self.aggr_module(inputs[0], edge_index_i, ptr=ptr, dim_size=dim_size,
                                dim=self.node_dim) + \
            self.aggr_module(inputs[1], edge_index_j, ptr=ptr, dim_size=dim_size,
                             dim=self.node_dim)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.out_channels})')


class MLPEncoder(torch.nn.Module):
    def __init__(self, n_node_feat_dim, n_edge_feat_dim):
        super().__init__()
        self.node_encoder = nn.Sequential(
            *[layer for _ in range(1)
              for layer in (Linear(-1, n_node_feat_dim, **LINEAR_INIT),
                            nn.ReLU(inplace=True))][:-1])
        self.edge_encoder = nn.Sequential(
            *[layer for _ in range(1)
              for layer in (Linear(-1, n_edge_feat_dim, **LINEAR_INIT),
                            nn.ReLU(inplace=True))][:-1])
    
    def reset_parameters(self):
        reset_subnet(self.node_encoder)
        reset_subnet(self.edge_encoder)
    
    def forward(self, x, e):
        x = self.node_encoder(x.to(dtype=torch.float))
        e = self.edge_encoder(e.to(dtype=torch.float).view(e.shape[0], -1))
        return x, e

class EmbeddingEncoder(torch.nn.Module):
    def __init__(self, n_node_feat_dim, n_edge_feat_dim, n_node_attr, n_edge_attr, n_pos_enc):
        super().__init__()
        self.node_encoder = nn.Embedding(n_node_attr, n_node_feat_dim)
        self.edge_encoder = nn.Embedding(n_edge_attr, n_edge_feat_dim)
        self.pos_encoder = nn.Embedding(n_pos_enc, n_edge_feat_dim)
        self._n_pos_enc = n_pos_enc
        self._n_edge_attr = n_edge_attr
    
    def reset_parameters(self):
        reset_subnet(self.node_encoder)
        reset_subnet(self.edge_encoder)
    
    def forward(self, x, e):
        x = self.node_encoder(x)
        if self._n_pos_enc > 0:
            p = self.pos_encoder(e // self._n_edge_attr)
            e = self.edge_encoder(e % self._n_edge_attr)
            e += p
        else:
            e = self.edge_encoder(e)
        return x, e


class GGNN(torch.nn.Module):
    def __init__(self, encoder, aggr, n_node_feat_dim, n_edge_feat_dim, layer_groups, n_message_net_layers, skip_mode, output_mode, num_query, n_atte_layers, layer_aggr="add", layer_aggr_kwargs={}, concat_skip=1):
        super().__init__()
        # Message Passing layers
        self.convs = torch.nn.ModuleList()
        self.skips = torch.nn.ParameterList()
        for num_layers in layer_groups:
            group_convs = torch.nn.ModuleList()
            for i in range(num_layers):
                if i == 0:
                    conv = GatedGraphConv(n_node_feat_dim,
                                          n_message_net_layers,
                                          layer_aggr, 
                                          layer_aggr_kwargs).jittable()
                else:
                    conv = group_convs[0]
                group_convs.append(conv)
            self.convs.append(group_convs)
            self.skips.append(Param(torch.Tensor(1 if skip_mode==2 else 0)))
        
        self.softmax = nn.Softmax(dim=-1)
        if output_mode == 2:
            self.pos_embeds = nn.Embedding(len(layer_groups)+1, n_node_feat_dim)
            self.mab = MultiheadAttentionBlock(
                n_node_feat_dim, 1, True, 0.)
        else:
            self.pos_embeds = None
            self.mab = None

        self.encoder = encoder
        self.aggr = aggr
        # Misc
        self._skip_mode = skip_mode
        self._output_mode = output_mode
        self._concat_skip = concat_skip
        # init
        self.reset_parameters()
        return

    @torch.jit.ignore
    def reset_parameters(self):
        for group_convs, skip_beta in zip(self.convs, self.skips):
            group_convs[0].reset_parameters()
            skip_beta.data.fill_(0.0)

    def forward(self, x, edge_index, edge_feat, graph_idx, batch_size: int):
        x, e = self.encoder(x, edge_feat)
        out_feats = [x]
        for group_convs, skip_beta in zip(self.convs, self.skips):
            skip_input = x
            for conv in group_convs:
                x = conv(x, edge_index, e)
            if self._output_mode > 0:
                out_feats.append(x)
            if self._skip_mode == 2:
                alpha = skip_beta.sigmoid()
                x = x * (1 - alpha) + skip_input * alpha
            elif self._skip_mode == 1:
                x += skip_input
        if self.pos_embeds is not None and self.mab is not None:
            x = torch.stack(out_feats[1:], dim=1)
            # x: n_node x n_layer x n_channel
            pos = torch.tensor(list(range(x.shape[1])), 
                               dtype=torch.int, device=x.device)
            x = x + self.pos_embeds(pos)
            x = self.mab(out_feats[0].unsqueeze(1), x, x_mask=None, y_mask=None).squeeze(1)
        elif self._output_mode == 1:
            x = torch.cat(out_feats[self._concat_skip:], dim=-1)
        return self.aggr(x, graph_idx, batch_size)



