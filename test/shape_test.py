# -*- coding: utf-8 -*-
# @Time : 2024/4/3 13:23
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : shape_test.py
# @Software: PyCharm

import pandas as pd
import pickle as pkl
import numpy as np
from config import parser

if __name__ == "__main__":
    args = parser.parse_args()

    f = open('../'+args.multigraph_fea, 'rb')
    fea_G2, fea_G5, fea_G10 = pkl.load(f)
    f.close()

    f = open('../'+args.multigraph_adj, 'rb')
    adj_G2, adj_G5, adj_G10 = pkl.load(f)
    f.close()

    print('G2 adj shape:', adj_G2.shape)
    print('G5 adj shape:', adj_G5.shape)
    print('G10 adj shape:', adj_G10.shape)

    print('G2 fea shape:', len(fea_G2), 'x', fea_G2[0].shape)
    print('G5 fea shape:', len(fea_G5), 'x', fea_G5[0].shape)
    print('G10 fea shape:', len(fea_G10), 'x', fea_G10[0].shape)

    f = open('../'+args.trajectory_pos, 'rb')
    pos_data_list = pkl.load(f)
    f.close()

    f = open('../'+args.trajectory_neg, 'rb')
    neg_data_list = pkl.load(f)
    f.close()

    print('Positive data count:', len(pos_data_list))
    print('\tid type:', type(pos_data_list[0]['id']), 'data shape:', pos_data_list[0]['data'].shape, 'label shape:',
          pos_data_list[0]['label'].shape)
    print('Negative data count:', len(neg_data_list))
    print('\tid type:', type(neg_data_list[0]['id']), 'data shape:', neg_data_list[0]['data'].shape, 'label shape:',
          neg_data_list[0]['label'].shape)

    for carid in range(args.max_car_num):
        ## positive Trajectory data
        f = open('../data/hangzhou/one_traj/' + str(carid) + '.pkl', 'rb')
        data_list = pkl.load(f)
        f.close()
        MAX_FEE = 1.0
        for i in range(len(data_list)):
            if i == 0:
                tmp_data = data_list[i]['data']
                tmp_label = data_list[i]['label']
                tmp_static_reward = data_list[i]['reward']
                tmp_MAX_FEE = max(tmp_static_reward)
            else:
                tmp_data = np.r_[tmp_data, data_list[i]['data']]
                tmp_label = np.r_[tmp_label, data_list[i]['label']]
                tmp_static_reward = np.r_[tmp_static_reward, data_list[i]['reward']]
                if tmp_MAX_FEE < max(tmp_static_reward):
                    tmp_MAX_FEE = max(tmp_static_reward)
        print('car '+ str(carid) + ':', 'data shape:', tmp_data.shape, 'label shape:', tmp_label.shape)