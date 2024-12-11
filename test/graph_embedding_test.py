# -*- coding: utf-8 -*-
# @Time : 2024/4/3 15:19
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : graph_embedding_test.py
# @Software: PyCharm
import logging
import time

import torch
from torch.utils.data import DataLoader

from config import parser
from model.DSEmb.graphbase import NCModel
from utils.data_utils import DS_data
from utils.train_utils import format_metrics

if __name__ == "__main__":
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    if not args.cuda == -1:
        args.device = 'cuda:' + str(args.cuda)

    args.multigraph_adj = '../' + args.multigraph_adj
    args.multigraph_fea = '../' + args.multigraph_fea
    DSDATA = DS_data(args)
    dataloader = DataLoader(DSDATA, batch_size=4, shuffle=True, num_workers=1)
    for i, batch in enumerate(dataloader):
        a, b, c, a_label, b_label, c_label = batch
        args.n_nodes = a.shape[1]
        args.feat_dim = a.shape[2]
        break
    adj_G2, adj_G5, adj_G10 = DSDATA.get_adj()

    model_G2 = NCModel(args)
    # model_G5 = NCModel
    # model_G10 = NCModel
    if not args.cuda == -1:
        model_G2 = model_G2.to(args.device)
        adj_G2 = adj_G2.to(args.device)
        adj_G5 = adj_G5.to(args.device)
        adj_G10 = adj_G10.to(args.device)

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    optimizer = torch.optim.Adam(params=model_G2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )

    for ep in range(args.epochs):
        for i, batch in enumerate(dataloader):
            t = time.time()
            a, b, c, a_label, b_label, c_label = batch
            if not args.cuda == -1:
                a = a.to(args.device)
                b = b.to(args.device)
                c = c.to(args.device)
                a_label = a_label.to(args.device)
                b_label = b_label.to(args.device)
                c_label = c_label.to(args.device)

            model_G2.train()
            optimizer.zero_grad()
            embeddings = model_G2.encode(a, adj_G2)
            # print('test train emb shape:', embeddings.shape)
            train_metrics = model_G2.compute_metrics(embeddings, adj_G2, a_label)
            train_metrics['loss'].backward()
            if args.grad_clip is not None:
                max_norm = float(args.grad_clip)
                all_params = list(model_G2.parameters())
                for param in all_params:
                    torch.nn.utils.clip_grad_norm_(param, max_norm)
            optimizer.step()
            lr_scheduler.step()
            if (i + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(i + 1),
                                       'lr: {}'.format(lr_scheduler.get_last_lr()[0]),
                                       format_metrics(train_metrics, 'train'),
                                       'time: {:.4f}s'.format(time.time() - t)
                                       ]))
