# -*- coding: utf-8 -*-
# @Time : 2024/3/30 9:28
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : 16.sample_trajectory.py
# @Software: PyCharm
import json
import math
import os
import pickle
import pandas as pd
import numpy as np

from config import parser
from utils.BDHconversion import dec_to_bin
from utils.Geohash import Geohash

########## 人工选车ID！！！ ##########

def pos_neg_ids(carIDs, default_val):
    pos_carID = []
    neg_carIDs = []
    if default_val == None:
        pos_carID.append(carIDs[0])
        neg_carIDs = carIDs[1:]
    elif type(default_val) == int:
        pos_carID.append(carIDs[default_val])
        for id in carIDs:
            if id != pos_carID:
                neg_carIDs.append(id)
    else:
        flag = True
        for id in carIDs:
            if id == default_val:
                pos_carID.append(id)
                flag = False
            elif id != carIDs[0]:
                neg_carIDs.append(id)
        if flag:
            pos_carID.append(carIDs[0])
        else:
            neg_carIDs.append(carIDs[0])
    return pos_carID, neg_carIDs

def sample_generate(args, carIDs, df, outfile, fillnan=0):
    f = open('../'+args.hg2_central_edge, 'r')
    hg2_json = json.load(f)
    f.close()
    hg2_zone_num = len(hg2_json.keys())
    hg2_width = len(dec_to_bin(str(hg2_zone_num)))
    f = open('../'+args.hg5_central_edge, 'r')
    hg5_json = json.load(f)
    f.close()
    hg5_zone_num = len(hg5_json.keys())
    hg5_width = len(dec_to_bin(str(hg5_zone_num)))
    f = open('../'+args.hg10_central_edge, 'r')
    hg10_json = json.load(f)
    f.close()
    hg10_zone_num = len(hg10_json.keys())
    hg10_width = len(dec_to_bin(str(hg10_zone_num)))

    data_list = []
    for id in carIDs:
        tmp_df = df[df['vID'] == id].fillna(fillnan)
        car_emb:np.array
        label_emb:np.array
        reward:np.array
        print(id)
        last_hg2 = 0
        for ts in range(int(tmp_df['timestep'].max()) + 1):
            r = float(tmp_df.loc[tmp_df['timestep'] == ts, 'orderfee'].values[0])

            hg2 = int(tmp_df.loc[tmp_df['timestep'] == ts, 'hg2'].values[0])
            hg2_emb_str = dec_to_bin(str(hg2)).zfill(hg2_width)
            hg2_emb = []
            for i in range(len(hg2_emb_str)):
                hg2_emb.append(int(hg2_emb_str[i]))
            hg2_emb = np.array(hg2_emb)[np.newaxis, :]
            # print('hg2_emb:', hg2_emb)

            hg5 = int(tmp_df.loc[tmp_df['timestep'] == ts, 'hg5'].values[0])
            hg5_emb_str = dec_to_bin(str(hg5)).zfill(hg5_width)
            hg5_emb = []
            for i in range(len(hg5_emb_str)):
                hg5_emb.append(int(hg5_emb_str[i]))
            hg5_emb = np.array(hg5_emb)[np.newaxis, :]
            # print('hg5_emb:', hg5_emb)

            hg10 = int(tmp_df.loc[tmp_df['timestep'] == ts, 'hg10'].values[0])
            hg10_emb_str = dec_to_bin(str(hg10)).zfill(hg10_width)
            hg10_emb = []
            for i in range(len(hg10_emb_str)):
                hg10_emb.append(int(hg10_emb_str[i]))
            hg10_emb = np.array(hg10_emb)[np.newaxis, :]
            # print('hg10_emb:', hg10_emb)

            lon = tmp_df.loc[tmp_df['timestep'] == ts, 'lon'].values[0]
            lat = tmp_df.loc[tmp_df['timestep'] == ts, 'lat'].values[0]
            geohash = Geohash()
            _, geo_emb = geohash.encode(lon=lon, lat=lat, precision=12)
            geo_emb = np.array(geo_emb)[np.newaxis, :]
            # print('geo_emb:', geo_emb)

            dis = tmp_df.loc[tmp_df['timestep'] == ts, 'dis'].values[0]
            dis = dis / (args.hg2_radius * 2)
            if dis > 1.0:
                dis = 1.0
            degree = tmp_df.loc[tmp_df['timestep'] == ts, 'degree'].values[0]

            degree = degree / (2 * math.pi)
            label = -1
            if last_hg2 > 0:
                if last_hg2 != hg2:
                    if degree < 1/6:
                        label = 1
                    elif degree < 1/3:
                        label = 2
                    elif degree < 1/2:
                        label = 3
                    elif degree < 2/3:
                        label = 4
                    elif degree < 5/6:
                        label = 5
                    else:
                        label = 6
                else:
                    label = 0


            emb_one_step = np.concatenate((hg2_emb, hg5_emb, hg10_emb, geo_emb), axis=1)
            if ts == 0:
                car_emb = emb_one_step
                label_emb = np.array([dis, degree, label])[np.newaxis, :]
                reward = np.array([r])
            else:
                car_emb = np.concatenate((car_emb, emb_one_step), axis=0)
                tmp_label_emb = np.array([dis, degree, label])[np.newaxis, :]
                label_emb = np.concatenate((label_emb, tmp_label_emb), axis=0)
                tmp_reward = np.array([r])
                reward = np.concatenate((reward, tmp_reward), axis=0)
            last_hg2 = hg2
        data_list.append({'id':id[3:],'data':car_emb, 'label': label_emb, 'reward':reward})
        # print('car_emb:', car_emb)
        print('car_emb_shape:', car_emb.shape)
        # print('label_emb:', label_emb)
        print('label_emb_shape:', label_emb.shape)
        print('reward_shape:', reward.shape)
    f = open(outfile, 'wb')
    pickle.dump(data_list, f)
    f.close()

    return

if __name__ == "__main__":
    args = parser.parse_args()

    f = open('../data/preprocess/tmp_carIDs.pkl', 'rb')
    carIDs = pickle.load(f)
    print(carIDs)

    default_val = '浙ATD753'
    pos_carIDs, neg_carIDs = pos_neg_ids(carIDs, default_val)
    print(pos_carIDs)
    print(neg_carIDs)

    df = pd.read_csv('../data/preprocess/taxi_region_data.csv')
    print(df.head())

    if not os.path.exists('../data/preprocess/' + args.table[-4:]):
        os.mkdir('../data/preprocess/' + args.table[-4:])

    pos_outfile = '../data/preprocess/' + args.table[-4:] + '/trajectory_pos.pkl'
    neg_outfile = '../data/preprocess/' + args.table[-4:] + '/trajectory_neg.pkl'

    sample_generate(args, pos_carIDs, df, pos_outfile)
    sample_generate(args, neg_carIDs, df, neg_outfile)


