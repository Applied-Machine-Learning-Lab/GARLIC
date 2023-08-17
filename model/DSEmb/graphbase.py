# -*- coding: utf-8 -*-
# @Time : 2023/4/3 13:37
# @Author : Xiao Han
# @E-mail : hahahenha@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : graphbase.py
# @Software: PyCharm

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F

import model.DSEmb.encoders as encoders
from model.DSEmb.decoders import model2decoder
from utils.eval_utils import acc_f1

class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.encoder = getattr(encoders, args.graph_model)(args)

    def encode(self, x, adj):
        emb = self.encoder.encode(x, adj)
        return emb

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError

class NCModel(BaseModel):
    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.decoder = model2decoder[args.graph_model](args)
        if args.node_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        self.weights = torch.Tensor([1.] * args.node_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to('cuda:'+str(args.cuda))

    def decode(self, h, adj):
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output, dim=2)

    def compute_metrics(self, embeddings, adj, data_label):
        output = self.decode(embeddings, adj)
        # print('compute_metrics output shape:', output.shape)
        # print('compute_metrics data_label shape:', data_label.shape)
        loss = F.nll_loss(output.transpose(1, 2), data_label, self.weights)
        acc = 0
        f1 = 0
        cnt = 0
        for i in range(embeddings.shape[0]):
            tmp_acc, tmp_f1 = acc_f1(output[i], data_label[i], average=self.f1_average)
            cnt += 1
            acc += tmp_acc
            f1 += tmp_f1
        metrics = {'loss': loss, 'acc': acc / cnt, 'f1': f1 / cnt}
        return metrics