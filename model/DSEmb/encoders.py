# -*- coding: utf-8 -*-
# @Time : 2023/4/3 14:10
# @Author : Xiao Han
# @E-mail : hahahenha@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : encoders.py
# @Software: PyCharm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DSEmb.layers import GraphConvolution, GATL
from utils.parser_utils import get_dim_act


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self):
        super(Encoder, self).__init__()

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output

class GCN(Encoder):
    """
    Graph Convolution Networks.
    """

    def __init__(self, args):
        super(GCN, self).__init__()
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gc_layers.append(GraphConvolution(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*gc_layers)
        self.encode_graph = True

class GAT(Encoder):
    """
    Graph Attention Networks.
    """

    def __init__(self, args):
        super(GAT, self).__init__()
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gat_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            assert dims[i + 1] % args.n_heads == 0
            out_dim = dims[i + 1] // args.n_heads
            gat_layers.append(
                    GATL(in_dim, out_dim, args.dropout, args.alpha, args.n_heads))
        self.layers = nn.Sequential(*gat_layers)
        self.encode_graph = True