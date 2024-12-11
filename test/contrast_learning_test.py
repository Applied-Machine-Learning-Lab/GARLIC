# -*- coding: utf-8 -*-
# @Time : 2024/4/4 9:58
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : contrast_learning_test.py
# @Software: PyCharm
import time

import torch
import logging
import torch.nn as nn
from config import parser
from torch.utils.data import DataLoader

from model.RL.layers import Reward
from utils.data_utils import TrajData
from utils.eval_utils import acc_f1

if __name__ == "__main__":
    args = parser.parse_args()
    if not args.cuda == -1:
        args.device = 'cuda:' + str(args.cuda)
    args.trajectory_pos = '../' + args.trajectory_pos
    args.trajectory_neg = '../' + args.trajectory_neg

    logging.getLogger().setLevel(logging.INFO)

    contrast_mode_str = 'all'
    trajdata = TrajData(args, 0.3, contrast_mode=contrast_mode_str)
    dataloader = DataLoader(trajdata, batch_size=args.batch_size, shuffle=True, num_workers=1)

    in_dim = -1
    if contrast_mode_str == 'all':
        out_dim = 50
        avg = 'micro'
    else:
        out_dim = 2
        avg = 'binary'
    for i, batch in enumerate(dataloader):
        vid, state, action, reward = batch
        in_dim = state.shape[2] + action.shape[2]
        break

    model = Reward(in_dim, args.dim, out_dim, batch_size=args.batch_size)
    hidden = model.init_hidden()

    if not args.cuda == -1:
        model = model.to(args.device)
        hidden = hidden.to(args.device)

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )

    CEloss = nn.CrossEntropyLoss()
    loss = 0
    acc = 0
    f1 = 0
    cnt = 0
    for ep in range(args.epochs):
        for i, batch in enumerate(dataloader):
            t = time.time()
            vid, state, action, reward = batch
            # print('vid:', vid)
            if state.shape[0] < args.batch_size:
                continue
            if not args.cuda == -1:
                vid = vid.to(args.device)
                state = state.to(args.device)
                action = action.to(args.device)
            x = torch.cat([state, action], dim=2).float()
            vid = vid.long()

            model.train()
            optimizer.zero_grad()
            pred, _ = model.forward(x, hidden)
            tmp_loss = CEloss(pred, vid)
            tmp_acc, tmp_f1 = acc_f1(pred, vid, average=avg)
            loss += tmp_loss
            acc += tmp_acc
            f1 += tmp_f1
            cnt += 1
            loss.backward()
            if args.grad_clip is not None:
                max_norm = float(args.grad_clip)
                all_params = list(model.parameters())
                for param in all_params:
                    torch.nn.utils.clip_grad_norm_(param, max_norm)
            optimizer.step()
            lr_scheduler.step()

            model.eval()
            pred, _ = model.forward(x, hidden)
            pred = pred.cpu()
            # print('test:', pred)

            if (i + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(i + 1),
                                       'lr: {}'.format(lr_scheduler.get_last_lr()[0]),
                                       'loss: {}'.format(loss/cnt),
                                       'acc: {}'.format(acc/cnt),
                                       'f1: {}'.format(f1/cnt),
                                       'time: {:.4f}s'.format(time.time() - t)
                                       ]))
                acc = 0
                f1 = 0
                loss = 0
                cnt = 0
