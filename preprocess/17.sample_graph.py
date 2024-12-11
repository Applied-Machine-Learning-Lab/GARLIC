# -*- coding: utf-8 -*-
# @Time : 2024/3/30 11:16
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : 17.sample_graph.py
# @Software: PyCharm
import pickle

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from numpy import sort

from config import parser

if __name__ == '__main__':
    args = parser.parse_args()

    hg2_relation = pd.read_csv('../' + args.hg2_relation)
    hg5_relation = pd.read_csv('../' + args.hg5_relation)
    hg10_relation = pd.read_csv('../' + args.hg10_relation)

    G2 = nx.from_pandas_edgelist(hg2_relation, 'zoneID_from', 'zoneID_to')
    G5 = nx.from_pandas_edgelist(hg5_relation, 'zoneID_from', 'zoneID_to')
    G10 = nx.from_pandas_edgelist(hg10_relation, 'zoneID_from', 'zoneID_to')

    # Plot it
    # nx.draw_networkx(G2, with_labels=True)
    # plt.show()
    # nx.draw_networkx(G5, with_labels=True)
    # plt.show()
    # nx.draw_networkx(G10, with_labels=True)
    # plt.show()

    nodes2 = list(G2.nodes)
    sorted_node2 = sort(nodes2)
    adj_G2 = nx.to_numpy_matrix(G2, nodelist=sorted_node2)
    # print(adj_G2.shape)
    nodes5 = list(G5.nodes)
    sorted_node5 = sort(nodes5)
    adj_G5 = nx.to_numpy_matrix(G5, nodelist=sorted_node5)
    # print(adj_G5.shape)
    nodes10 = list(G10.nodes)
    sorted_node10 = sort(nodes10)
    adj_G10 = nx.to_numpy_matrix(G10, nodelist=sorted_node10)
    # print(adj_G10.shape)

    adj_list = adj_G2, adj_G5, adj_G10

    f = open('../data/preprocess/' + args.table[-4:] + '/adjlist.pkl', 'wb')
    pickle.dump(adj_list, f)
    f.close()

    DS_data = pd.read_csv('../'+args.tmp_demand_graph)

    fea_G2_list = []
    fea_G5_list = []
    fea_G10_list = []
    for i in range(int(86400/args.rebalance_time)):
        df2 = DS_data[(DS_data['timestep'] == i) & (DS_data['zoneID'] <= 4000)]
        df5 = DS_data[(DS_data['timestep'] == i) & (DS_data['zoneID'] <= 7000) & (DS_data['zoneID'] >= 5000)]
        df10 = DS_data[(DS_data['timestep'] == i) & (DS_data['zoneID'] >= 10000)]
        fea_G2_list.append(df2.loc[:, ['demand', 'supply']].values)
        fea_G5_list.append(df5.loc[:, ['demand', 'supply']].values)
        fea_G10_list.append(df10.loc[:, ['demand', 'supply']].values)
    fea_list = fea_G2_list, fea_G5_list, fea_G10_list
    f = open('../data/preprocess/' + args.table[-4:] + '/fealist.pkl', 'wb')
    pickle.dump(fea_list, f)
    f.close()

    print('date:', args.table[-4:])
    print('G2 adj shape:', adj_G2.shape)
    print('G5 adj shape:', adj_G5.shape)
    print('G10 adj shape:', adj_G10.shape)

    print('len fea:', len(fea_G2_list))
    print('G2 fea shape:', fea_G2_list[0].shape)
    print('G5 fea shape:', fea_G5_list[0].shape)
    print('G10 fea shape:', fea_G10_list[0].shape)