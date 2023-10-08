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
from torch_geometric.nn import Linear

LINEAR_INIT = {
    'weight_initializer': 'kaiming_uniform',
    'bias_initializer': 'zeros',
}

def reset_subnet(subnet):
    for net in subnet:
        if hasattr(net, 'reset_parameters'):
            net.reset_parameters()


class MLP(torch.nn.Module):
    def __init__(self, num_layers, out_channels, acti_fini=False) -> None:
        super().__init__()
        layers = [layer for _ in range(num_layers) for layer in (
                Linear(-1, out_channels, **LINEAR_INIT), 
                torch.nn.ReLU(inplace=True))]
        if not acti_fini:
            layers = layers[:-1]
        self.lins = torch.nn.Sequential(*layers)
    
    @torch.jit.ignore
    def reset_parameters(self):
        for net in self.lins:
            if hasattr(net, 'reset_parameters'):
                net.reset_parameters()

    def forward(self, x):
        return self.lins(x)

