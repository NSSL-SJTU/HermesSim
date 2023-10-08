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

import torch
import gc
import torch.nn as nn
from torch.autograd import Variable

from .ggnn import GGNN
from .mlp import MLP

class HBMPEncoder(nn.Module):
    """
    Prepare and encode sentence embeddings
    """

    def __init__(self, hbmp_config):
        super(HBMPEncoder, self).__init__()
        self.word_embedding = nn.Embedding(
            hbmp_config["embed_size"], hbmp_config["embed_dim"], 0)
        self.encoder = HBMP(hbmp_config)
        proj_size = hbmp_config['proj_size']
        self.proj_lin = MLP(1, proj_size) if proj_size > 0 else None
        self._seq_limit = hbmp_config['seq_limit']

    def forward(self, input_sentence, edge_feat):
        input_sentence = input_sentence[:, :self._seq_limit]
        input_sentence = input_sentence.permute(1, 0).to(dtype=torch.long)
        sentence = self.word_embedding(input_sentence)
        embedding = self.encoder(sentence)
        if self.proj_lin is not None:
            embedding = self.proj_lin(embedding)
        return embedding, edge_feat.to(dtype=torch.float).view(-1, 1)


class HBMP(nn.Module):
    """
    Hierarchical Bi-LSTM Max Pooling Encoder
    """

    def __init__(self, hbmp_config):
        super(HBMP, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.cells = hbmp_config["layers"]*2
        self.hidden_dim = hbmp_config["hidden_dim"]
        self.rnn1 = nn.LSTM(input_size=hbmp_config["embed_dim"],
                            hidden_size=hbmp_config["hidden_dim"],
                            num_layers=hbmp_config["layers"],
                            dropout=hbmp_config["dropout"],
                            bidirectional=True)
        self.rnn2 = nn.LSTM(input_size=hbmp_config["embed_dim"],
                            hidden_size=hbmp_config["hidden_dim"],
                            num_layers=hbmp_config["layers"],
                            dropout=hbmp_config["dropout"],
                            bidirectional=True)
        self.rnn3 = nn.LSTM(input_size=hbmp_config["embed_dim"],
                            hidden_size=hbmp_config["hidden_dim"],
                            num_layers=hbmp_config["layers"],
                            dropout=hbmp_config["dropout"],
                            bidirectional=True)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        h_0 = c_0 = torch.zeros(
            self.cells, batch_size, self.hidden_dim, device=inputs.device)
        out1, (ht1, ct1) = self.rnn1(inputs, (h_0, c_0))
        emb1 = self.max_pool(out1.permute(1, 2, 0)).permute(2, 0, 1)
        out2, (ht2, ct2) = self.rnn2(inputs, (ht1, ct1))
        emb2 = self.max_pool(out2.permute(1, 2, 0)).permute(2, 0, 1)
        out3, (ht3, ct3) = self.rnn3(inputs, (ht2, ct2))
        emb3 = self.max_pool(out3.permute(1, 2, 0)).permute(2, 0, 1)
        emb = torch.cat([emb1, emb2, emb3], 2)
        emb = emb.squeeze(0)
        return emb


class GruEncoder(nn.Module):
    """
    Prepare and encode sentence embeddings
    """

    def __init__(self, c):
        super().__init__()
        self.word_embedding = nn.Embedding(c["embed_size"], c["embed_dim"], 0)
        self.encoder = nn.GRU(  input_size=c["embed_dim"],
                                hidden_size=c["hidden_dim"],
                                num_layers=c["layers"],
                                bidirectional=True)
        proj_size = c['proj_size']
        self.proj_lin = MLP(1, proj_size) if proj_size > 0 else None
        self._hidden_dim = c["hidden_dim"]
        self._seq_limit = c['seq_limit']

    def forward(self, input_sentence, edge_feat):
        input_sentence = input_sentence[:, :self._seq_limit]
        input_sentence = input_sentence.permute(1, 0).to(dtype=torch.long)
        sentence = self.word_embedding(input_sentence)
        # take the last inst embedding
        embedding, hn = self.encoder(sentence)
        embedding = torch.concat([
            embedding[-1, :, :self._hidden_dim], 
            embedding[ 0, :, self._hidden_dim:], 
        ], dim = -1)
        if self.proj_lin is not None:
            embedding = self.proj_lin(embedding)
        return embedding, edge_feat.to(dtype=torch.float).view(-1, 1)
