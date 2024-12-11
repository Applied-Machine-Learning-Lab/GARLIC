# -*- coding: utf-8 -*-
# @Time : 2024/4/4 10:45
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : 20.multi_days_data_combine.py
# @Software: PyCharm

import os
import numpy as np
import pickle as pkl
from config import parser

if __name__ == "__main__":
    args = parser.parse_args()

    # TODO: CHANGE DATE
    END_DAY = 30

    TOTAL_DAYS = END_DAY - 18 + 1

    adj_G2 = None
    adj_G5 = None
    adj_G10 = None
    fea_G2 = None
    fea_G5 = None
    fea_G10 = None
    pos_data_list = []
    neg_data_list = []

    car_cnt = 0
    car_dict = {}
    total_data_list = []

    m = 9
    d = 18
    for cnt in range(TOTAL_DAYS):
        day = d + cnt
        if day > 30:
            day -= 30
            m += 1
        md_str = str(m).zfill(2) + str(d).zfill(2)

        dir = '../' + 'data/preprocess/' + md_str + '/'

        f = open(dir+'adjlist.pkl', 'rb')
        if cnt == 0:
            adj_G2, adj_G5, adj_G10 = pkl.load(f)
        else:
            tmp_adj_G2, tmp_adj_G5, tmp_adj_G10 = pkl.load(f)
            assert (tmp_adj_G2 == adj_G2).all()
            assert (tmp_adj_G5 == adj_G5).all()
            assert (tmp_adj_G10 == adj_G10).all()
        f.close()

        f = open(dir+'fealist.pkl', 'rb')
        if cnt == 0:
            fea_G2, fea_G5, fea_G10 = pkl.load(f)
            assert len(fea_G2) == int(86400/args.rebalance_time)
            assert len(fea_G5) == int(86400/args.rebalance_time)
            assert len(fea_G10) == int(86400/args.rebalance_time)
        else:
            tmp_fea_G2, tmp_fea_G5, tmp_fea_G10 = pkl.load(f)
            assert len(tmp_fea_G2) == int(86400/args.rebalance_time)
            assert len(tmp_fea_G5) == int(86400/args.rebalance_time)
            assert len(tmp_fea_G10) == int(86400/args.rebalance_time)
            fea_G2 = np.r_[fea_G2, tmp_fea_G2]
            fea_G5 = np.r_[fea_G5, tmp_fea_G5]
            fea_G10 = np.r_[fea_G10, tmp_fea_G10]
        f.close()

        f = open(dir + 'trajectory_pos.pkl', 'rb')
        tmp_pos_data_list = pkl.load(f)
        f.close()
        for data in tmp_pos_data_list:
            print(data)
            if data['data'].shape[0] < int(86400/args.rebalance_time):
                tmp_zero = np.zeros((int(86400/args.rebalance_time)-data['data'].shape[0],data['data'].shape[1]))
                tmp_data = np.r_[data['data'], tmp_zero]
            else:
                tmp_data = data['data']
            if data['label'].shape[0] < int(86400/args.rebalance_time):
                tmp_zero = np.zeros((int(86400/args.rebalance_time)-data['label'].shape[0],data['label'].shape[1]))
                tmp_label = np.r_[data['label'], tmp_zero]
            else:
                tmp_label = data['label']
            if data['reward'].shape[0] < int(86400/args.rebalance_time):
                tmp_zero = np.zeros(int(86400/args.rebalance_time)-data['reward'].shape[0])
                tmp_reward = np.r_[data['reward'], tmp_zero]
            else:
                tmp_reward = data['reward']
            pos_data_list.append({'id': data['id'], 'data': tmp_data, 'label': tmp_label, 'reward': tmp_reward})
            if data['id'] not in car_dict.keys():
                car_dict[data['id']] = car_cnt
                car_cnt += 1
            total_data_list.append({'id': car_dict[data['id']], 'data': tmp_data, 'label': tmp_label, 'reward': tmp_reward})

        f = open(dir + 'trajectory_neg.pkl', 'rb')
        tmp_neg_data_list = pkl.load(f)
        f.close()
        for data in tmp_neg_data_list:
            if data['data'].shape[0] < int(86400/args.rebalance_time):
                tmp_zero = np.zeros((int(86400/args.rebalance_time)-data['data'].shape[0],data['data'].shape[1]))
                tmp_data = np.r_[data['data'], tmp_zero]
            else:
                tmp_data = data['data']
            if data['label'].shape[0] < int(86400/args.rebalance_time):
                tmp_zero = np.zeros((int(86400/args.rebalance_time)-data['label'].shape[0],data['label'].shape[1]))
                tmp_label = np.r_[data['label'], tmp_zero]
            else:
                tmp_label = data['label']
            if data['reward'].shape[0] < int(86400/args.rebalance_time):
                tmp_zero = np.zeros(int(86400/args.rebalance_time)-data['reward'].shape[0])
                tmp_reward = np.r_[data['reward'], tmp_zero]
            else:
                tmp_reward = data['reward']
            neg_data_list.append({'id': data['id'], 'data': tmp_data, 'label': tmp_label, 'reward': tmp_reward})
            if data['id'] not in car_dict.keys():
                car_dict[data['id']] = car_cnt
                car_cnt += 1
            total_data_list.append({'id': car_dict[data['id']], 'data': tmp_data, 'label': tmp_label, 'reward': tmp_reward})
    print('G2 adj shape:', adj_G2.shape)
    print('G5 adj shape:', adj_G5.shape)
    print('G10 adj shape:', adj_G10.shape)

    f = open('../' + args.multigraph_adj, 'wb')
    adj_list = adj_G2, adj_G5, adj_G10
    pkl.dump(adj_list, f)
    f.close()

    print('G2 fea shape:', len(fea_G2), 'x', fea_G2[0].shape)
    print('G5 fea shape:', len(fea_G5), 'x', fea_G5[0].shape)
    print('G10 fea shape:', len(fea_G10), 'x', fea_G10[0].shape)

    f = open('../' + args.multigraph_fea, 'wb')
    fea_list = fea_G2, fea_G5, fea_G10
    pkl.dump(fea_list, f)
    f.close()

    print('Positive data count:', len(pos_data_list))
    print('\tid:', pos_data_list[0]['id'], 'data shape:', pos_data_list[0]['data'].shape, 'label shape:',
          pos_data_list[0]['label'].shape, 'reward shape:', pos_data_list[0]['reward'].shape)
    print('Negative data count:', len(neg_data_list))
    print('\tid[0]:', neg_data_list[0]['id'], 'data shape:', neg_data_list[0]['data'].shape, 'label shape:',
          neg_data_list[0]['label'].shape, 'reward shape:', neg_data_list[0]['reward'].shape)
    print('Total data count:', len(total_data_list))
    print('\tid[0]:', total_data_list[0]['id'], 'data shape:', total_data_list[0]['data'].shape, 'label shape:',
          total_data_list[0]['label'].shape, 'reward shape:', total_data_list[0]['reward'].shape)

    print('car cnt max:', car_cnt)

    f = open('../' + args.trajectory_pos, 'wb')
    pkl.dump(pos_data_list, f)
    f.close()

    f = open('../' + args.trajectory_neg, 'wb')
    pkl.dump(neg_data_list, f)
    f.close()

    f = open('../' + args.trajectory_all, 'wb')
    pkl.dump(total_data_list, f)
    f.close()

    if not os.path.exists('../data/hangzhou/one_traj/'):
        os.mkdir('../data/hangzhou/one_traj/')
    for i in range(args.max_car_num):
        data_list = []
        for data in total_data_list:
            if data['id'] == i:
                data_list.append({'id': data['id'], 'data': data['data'], 'label': data['label'], 'reward': data['reward']})
        f = open('../data/hangzhou/one_traj/' + str(i) + '.pkl', 'wb')
        pkl.dump(data_list, f)
        f.close()
