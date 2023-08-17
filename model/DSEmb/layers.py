# -*- coding: utf-8 -*-
# @Time : 2023/4/3 14:15
# @Author : Xiao Han
# @E-mail : hahahenha@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : layers.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):
    """
    Simple GCN layer.
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(GraphConvolution, self).__init__()
        # print('GCN in feas:', in_features)
        # print('GCN out feas:', out_features)
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, input):
        x, adj = input
        # print('GCN x shape:', x.shape)
        # print('GCN adj shape:', adj.shape)
        hidden = self.linear.forward(x)
        # print('GCN hidden1 shape:', hidden.shape)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        # print('GCN hidden2 shape:', hidden.shape)
        tmp_adj = adj.unsqueeze(0).repeat(x.shape[0], 1, 1)
        # print('GCN tmp_adj shape:', tmp_adj.shape)
        # if adj.is_sparse:
        #     support = torch.spmm(adj, hidden)
        # else:
        #     support = torch.bmm(tmp_adj, hidden)
        support = torch.bmm(tmp_adj, hidden)
        output = self.act(support), adj
        return output

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(
                self.in_features, self.out_features
        )


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b

class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

# class SpGraphAttentionLayer(nn.Module):
#     """
#     Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """
#
#     def __init__(self, in_features, out_features, dropout, alpha, activation):
#         super(SpGraphAttentionLayer, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#
#         self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#         nn.init.xavier_normal_(self.W.data, gain=1.414)
#
#         self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
#         nn.init.xavier_normal_(self.a.data, gain=1.414)
#
#         self.dropout = nn.Dropout(dropout)
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#         self.special_spmm = SpecialSpmm()
#         self.act = activation
#
#     def forward(self, input, adj):
#         for ind in range(len(input.shape[0])):
#
#             N = input.size()[1]
#             if adj.is_sparse:
#                 edge = adj._indices()
#             else:
#                 adj = adj.to_sparse()
#                 edge = adj._indices()
#
#             h = torch.matmul(input, self.W)
#             # h: N x out
#             assert not torch.isnan(h).any()
#
#             # Self-attention on the nodes - Shared attention mechanism
#             edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
#             # edge: 2*D x E
#
#             edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
#             assert not torch.isnan(edge_e).any()
#             # edge_e: E
#
#             ones = torch.ones(size=(N, 1))
#             if h.is_cuda:
#                 ones = ones.cuda()
#             e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), ones)
#             # e_rowsum: N x 1
#
#             edge_e = self.dropout(edge_e)
#             # edge_e: E
#
#             h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
#             assert not torch.isnan(h_prime).any()
#             # h_prime: N x out
#
#             h_prime = h_prime.div(e_rowsum)
#             # h_prime: N x out
#             assert not torch.isnan(h_prime).any()
#         return self.act(h_prime)
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
#
# class GraphAttentionLayer(nn.Module):
#     def __init__(self, input_dim, output_dim, dropout, activation, alpha, nheads, concat):
#         """Sparse version of GAT."""
#         super(GraphAttentionLayer, self).__init__()
#         self.dropout = dropout
#         self.output_dim = output_dim
#         self.attentions = [SpGraphAttentionLayer(input_dim,
#                                                  output_dim,
#                                                  dropout=dropout,
#                                                  alpha=alpha,
#                                                  activation=activation) for _ in range(nheads)]
#         self.concat = concat
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)
#
#     def forward(self, input):
#         x, adj = input
#         x = F.dropout(x, self.dropout, training=self.training)
#         if self.concat:
#             h = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         else:
#             h_cat = torch.cat([att(x, adj).view((-1, self.output_dim, 1)) for att in self.attentions], dim=2)
#             h = torch.mean(h_cat, dim=2)
#         h = F.dropout(h, self.dropout, training=self.training)
#         return (h, adj)


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    图注意力层
    input: (B,N,C_in)
    output: (B,N,C_out)
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征数
        self.out_features = out_features  # 节点表示向量的输出特征数
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数
        self.concat = concat  # 如果为true, 再进行elu激活

        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # 初始化

        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        """
        inp: input_fea [B,N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵  [N, N] 非零即一，数据结构基本知识
        """
        h = torch.matmul(inp, self.W)  # [B, N, out_features]
        N = h.size()[1]  # N 图的节点数

        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features), h.repeat(1, N, 1)], dim=-1).view(-1,
                                                                                                                    N,
                                                                                                                    N,
                                                                                                                    2 * self.out_features)
        # [B, N, N, 2*out_features]

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        # [B, N, N, 1] => [B, N, N] 图注意力的相关系数（未归一化）

        zero_vec = -1e12 * torch.ones_like(e)  # 将没有连接的边置为负无穷

        attention = torch.where(adj > 0, e, zero_vec)  # [B, N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=1)  # softmax形状保持不变 [B, N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [B, N, N].[B, N, out_features] => [B, N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GATL(nn.Module):
    def __init__(self, n_feat, n_hid, dropout, alpha, n_heads):
        """Dense version of GAT
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征。
        """
        super(GATL, self).__init__()
        self.dropout = dropout

        # 定义multi-head的图注意力层
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块

    def forward(self, input):
        x, adj = input
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)  # 将每个head得到的表示进行拼接
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        # print(x.shape)
        return (x, adj)