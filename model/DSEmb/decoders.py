# -*- coding: utf-8 -*-
# @Time : 2023/4/3 14:21
# @Author : Xiao Han
# @E-mail : hahahenha@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : decoders.py
# @Software: PyCharm

from torch import nn
import torch.nn.functional as F
from model.DSEmb.layers import GraphConvolution, GraphAttentionLayer, GATL


class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self):
        super(Decoder, self).__init__()

    def decode(self, x, adj):
        if self.decode_adj:
            input = (x, adj)
            probs, _ = self.cls.forward(input)
        else:
            probs = self.cls.forward(x)
        return probs

class GCNDecoder(Decoder):
    """
    Graph Convolution Decoder.
    """

    def __init__(self, args):
        super(GCNDecoder, self).__init__()
        act = lambda x: x
        self.cls = GraphConvolution(args.graph_dim, args.node_classes, args.dropout, act, args.bias)
        self.decode_adj = True


class GATDecoder(Decoder):
    """
    Graph Attention Decoder.
    """

    def __init__(self, args):
        super(GATDecoder, self).__init__()
        self.cls = GATL(args.graph_dim, args.node_classes, args.dropout, args.alpha, 1)
        self.decode_adj = True

model2decoder = {
    'GCN': GCNDecoder,
    'GAT': GATDecoder
}
