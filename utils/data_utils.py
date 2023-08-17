# -*- coding: utf-8 -*-
# @Time : 2023/4/3 14:23
# @Author : Xiao Han
# @E-mail : hahahenha@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : data_utils.py
# @Software: PyCharm

import os
import pickle as pkl
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import sys

class DS_data(Dataset):
    def __init__(self, args):
        f = open(args.multigraph_adj, 'rb')
        adj_G2, adj_G5, adj_G10 = pkl.load(f)
        f.close()

        self.adj_G2 = adj_G2.astype(np.float32)
        self.adj_G5 = adj_G5.astype(np.float32)
        self.adj_G10 = adj_G10.astype(np.float32)

        f = open(args.multigraph_fea, 'rb')
        fea_G2, fea_G5, fea_G10 = pkl.load(f)
        f.close()

        self.fea_G2 = np.asarray(fea_G2).astype(np.float32)
        self.fea_G5 = np.asarray(fea_G5).astype(np.float32)
        self.fea_G10 = np.asarray(fea_G10).astype(np.float32)

        self.label_G2 = []
        self.label_G5 = []
        self.label_G10 = []
        for i in range(len(self.fea_G2)):
            temp_fea_G2 = self.fea_G2[i]
            tmp_label_G2 = []
            for j in range(len(temp_fea_G2)):
                tmp_fea_G2 = temp_fea_G2[j]
                if tmp_fea_G2[0] < tmp_fea_G2[1]:
                    tmp_label_G2.append(0)
                elif tmp_fea_G2[0] == tmp_fea_G2[1]:
                    tmp_label_G2.append(1)
                elif tmp_fea_G2[0] > tmp_fea_G2[1]:
                    tmp_label_G2.append(2)
                else:
                    assert tmp_fea_G2[0] > tmp_fea_G2[1]
            self.label_G2.append(tmp_label_G2)

            temp_fea_G5 = self.fea_G5[i]
            tmp_label_G5 = []
            for j in range(len(temp_fea_G5)):
                tmp_fea_G5 = temp_fea_G5[j]
                if tmp_fea_G5[0] < tmp_fea_G5[1]:
                    tmp_label_G5.append(0)
                elif tmp_fea_G5[0] == tmp_fea_G5[1]:
                    tmp_label_G5.append(1)
                elif tmp_fea_G5[0] > tmp_fea_G5[1]:
                    tmp_label_G5.append(2)
                else:
                    assert tmp_fea_G5[0] > tmp_fea_G5[1]
            self.label_G5.append(tmp_label_G5)


            temp_fea_G10 = self.fea_G10[i]
            tmp_label_G10 = []
            for j in range(len(temp_fea_G10)):
                tmp_fea_G10 = temp_fea_G10[j]
                if tmp_fea_G10[0] < tmp_fea_G10[1]:
                    tmp_label_G10.append(0)
                elif tmp_fea_G10[0] == tmp_fea_G10[1]:
                    tmp_label_G10.append(1)
                elif tmp_fea_G10[0] > tmp_fea_G10[1]:
                    tmp_label_G10.append(2)
                else:
                    assert tmp_fea_G10[0] > tmp_fea_G10[1]
            self.label_G10.append(tmp_label_G10)

        self.label_G2 = np.asarray(self.label_G2).astype(np.int64)
        self.label_G5 = np.asarray(self.label_G5).astype(np.int64)
        self.label_G10 = np.asarray(self.label_G10).astype(np.int64)

        assert len(self.fea_G2) == len(self.fea_G5)
        assert len(self.fea_G5) == len(self.fea_G10)
        assert len(self.fea_G10) == len(self.label_G2)
        assert len(self.label_G2) == len(self.label_G5)
        assert len(self.label_G5) == len(self.label_G10)

    def get_adj(self):
        return torch.as_tensor(self.adj_G2), torch.as_tensor(self.adj_G5), torch.as_tensor(self.adj_G10)

    def get_all_fea(self):
        return torch.as_tensor(self.fea_G2), torch.as_tensor(self.fea_G5), torch.as_tensor(self.fea_G10)

    def __getitem__(self, index):
        return self.fea_G2[index], self.fea_G5[index], self.fea_G10[index], self.label_G2[index], self.label_G5[index], self.label_G10[index]

    def __len__(self):
        return len(self.fea_G2)

class TrajData(Dataset):
    def __init__(self, args, rate = 0.5, contrast_mode='one'):
        f = open(args.trajectory_pos, 'rb')
        pos_list = pkl.load(f)
        f.close()
        self.traj_pos_list = pos_list

        f = open(args.trajectory_neg, 'rb')
        neg_list = pkl.load(f)
        f.close()
        self.traj_neg_list = neg_list

        self.rate = rate

        dd = {}
        cnt = 0
        for i in range(len(pos_list)):
            key = pos_list[i]['id']
            if key in dd.keys():
                if contrast_mode == 'all' or dd[key] == 0:
                    self.traj_pos_list[i]['id'] = dd[key]
                else:
                    self.traj_pos_list[i]['id'] = 1
            else:
                dd[key] = cnt
                if contrast_mode == 'all' or dd[key] == 0:
                    # print('oneoneone')
                    self.traj_pos_list[i]['id'] = dd[key]
                else:
                    # print('elseelseelse')
                    self.traj_pos_list[i]['id'] = 1
                cnt += 1

        for i in range(len(neg_list)):
            key = neg_list[i]['id']
            if key in dd.keys():
                if contrast_mode == 'all' or dd[key] == 0:
                    self.traj_neg_list[i]['id'] = dd[key]
                else:
                    self.traj_neg_list[i]['id'] = 1
            else:
                dd[key] = cnt
                if contrast_mode == 'all' or dd[key] == 0:
                    # print('oneoneone')
                    self.traj_neg_list[i]['id'] = dd[key]
                else:
                    # print('elseelseelse')
                    self.traj_neg_list[i]['id'] = 1
                cnt += 1
        # for i in range(len(self.traj_pos_list)):
        #     print('test traj_pos_list:', self.traj_pos_list[i]['id'])
        # for i in range(len(self.traj_neg_list)):
        #     print('test traj_neg_list:', self.traj_neg_list[i]['id'])

    def __getitem__(self, index):
        seed = random.random()
        if seed > self.rate:
            return self.traj_neg_list[index % len(self.traj_neg_list)]['id'], self.traj_neg_list[index % len(self.traj_neg_list)]['data'], self.traj_neg_list[index % len(self.traj_neg_list)]['label'], self.traj_neg_list[index % len(self.traj_neg_list)]['reward']
        else:
            return self.traj_pos_list[index % len(self.traj_pos_list)]['id'], self.traj_pos_list[index % len(self.traj_pos_list)]['data'], self.traj_pos_list[index % len(self.traj_pos_list)]['label'], self.traj_pos_list[index % len(self.traj_pos_list)]['reward']

    def __len__(self):
        return len(self.traj_pos_list) + len(self.traj_neg_list)


class TrajAllData(Dataset):
    def __init__(self, args):
        f = open(args.trajectory_all, 'rb')
        all_list = pkl.load(f)
        f.close()
        self.traj_list = all_list

        dd = {}
        cnt = 0
        for i in range(len(all_list)):
            key = all_list[i]['id']
            if key in dd.keys():
                self.traj_list[i]['id'] = dd[key]
            else:
                dd[key] = cnt
                self.traj_list[i]['id'] = dd[key]
                cnt += 1

        # for i in range(len(self.traj_list)):
        #     print('test traj_list:', self.traj_list[i]['id'])

    def __getitem__(self, index):
        return self.traj_list[index]['id'], self.traj_list[index]['data'], self.traj_list[index]['label'], self.traj_list[index]['reward']

    def __len__(self):
        return len(self.traj_list)

if __name__=='__main__':
    from torch.utils.data import DataLoader
    from config import parser

    args = parser.parse_args()

    # # DSDATA test
    # args.multigraph_adj = '../' + args.multigraph_adj
    # args.multigraph_fea = '../' + args.multigraph_fea
    # DSDATA = DS_data(args)
    # dataloader = DataLoader(DSDATA, batch_size=3, shuffle=True, num_workers=1)
    # adj_G2, adj_G5, adj_G10 = DSDATA.get_adj()
    # for i, batch in enumerate(dataloader):
    #     print(i)
    #     a, b, c, a_label, b_label, c_label = batch
    #     print(adj_G2.shape)
    #     print(a.shape)
    #     print(a_label.shape)
    #
    #     print(adj_G5.shape)
    #     print(b.shape)
    #     print(b_label.shape)
    #
    #     print(adj_G10.shape)
    #     print(c.shape)
    #     print(c_label.shape)


    # # TrajData Test
    # args.trajectory_pos = '../' + args.trajectory_pos
    # args.trajectory_neg = '../' + args.trajectory_neg
    # trajdata = TrajData(args, 0.3, 'all')
    # dataloader = DataLoader(trajdata, batch_size=4, shuffle=True, num_workers=1)
    # for i, batch in enumerate(dataloader):
    #     print(i)
    #     vid, data, label, reward = batch
    #     print('vid:', vid, vid.shape)
    #     print('data shape:', data.shape)
    #     print('label shape:', label.shape)
    #     print('reward shape:', reward.shape)

    # TrajAllData Test
    args.trajectory_all = '../' + args.trajectory_all
    trajdata = TrajAllData(args)
    dataloader = DataLoader(trajdata, batch_size=4, shuffle=True, num_workers=1)
    for i, batch in enumerate(dataloader):
        print(i)
        vid, data, label, reward = batch
        print('vid:', vid, vid.shape)
        print('data shape:', data.shape)
        print('label shape:', label.shape)
        print('reward shape:', reward.shape)