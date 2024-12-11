# -*- coding: utf-8 -*-
# @Time : 2024/3/29 9:38
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : 14.demand_graph.py
# @Software: PyCharm

import pandas as pd
import geopandas as gpd
from config import parser
from shapely.geometry import Point

if __name__ == '__main__':
    args = parser.parse_args()

    TIME_STEP = args.DS_window
    TOTOAL_DATETIME = 24 * 3600

    df_edge = pd.read_csv('../' + args.tmp_time_change_data)
    print(df_edge.head())

    g2 = gpd.read_file('../data/taxi_zones/hg2')
    g2 = g2.to_crs('EPSG:4326') # x y 转经纬度
    g5 = gpd.read_file('../data/taxi_zones/hg5')
    g5 = g5.to_crs('EPSG:4326')
    g10 = gpd.read_file('../data/taxi_zones/hg10')
    g10 = g10.to_crs('EPSG:4326')

    f = open('../'+args.tmp_demand_graph, 'w')
    f.write('timestep,zoneID,demand,supply\n')

    df_edge1 = df_edge.copy().sort_values(by='dropoff_datetime', ascending=True)
    print(df_edge1.head())

    cnt = 0
    cnt1 = 0
    timestep = 0
    region_demand_g2 = {}
    region_supply_g2 = {}
    for id in g2['id']:
        region_demand_g2[id] = 0
        region_supply_g2[id] = 0
    region_demand_g5 = {}
    region_supply_g5 = {}
    for id in g5['id']:
        region_demand_g5[id] = 0
        region_supply_g5[id] = 0
    region_demand_g10 = {}
    region_supply_g10 = {}
    for id in g10['id']:
        region_demand_g10[id] = 0
        region_supply_g10[id] = 0
    for time in range(TOTOAL_DATETIME):
        if time % TIME_STEP == 0:
            for id in g2['id']:
                f.write(str(timestep)+','+str(id)+','+str(region_demand_g2[id])+','+str(region_supply_g2[id])+'\n')
                region_demand_g2[id] = 0
                region_supply_g2[id] = 0
            for id in g5['id']:
                f.write(str(timestep)+','+str(id)+','+str(region_demand_g5[id])+','+str(region_supply_g5[id])+'\n')
                region_demand_g5[id] = 0
                region_supply_g5[id] = 0
            for id in g10['id']:
                f.write(str(timestep)+','+str(id)+','+str(region_demand_g10[id])+','+str(region_supply_g10[id])+'\n')
                region_demand_g10[id] = 0
                region_supply_g10[id] = 0
            timestep += 1
        tmp_cnt = cnt
        for i in range(tmp_cnt, len(df_edge)):
            ts = df_edge.loc[i, 'pickup_datetime']
            x = df_edge.loc[i, 'pickup_longitude']
            y = df_edge.loc[i, 'pickup_latitude']
            if int(ts) > time:
                break

            cnt = i + 1
            point = Point([x, y])
            for j in range(len(g2)):
                if point.within(g2.loc[j].geometry):
                    g2_id = g2.loc[j].id
                    # print(g2_id, 'has a demand')
                    region_demand_g2[g2_id] += 1
                    break
            for j in range(len(g5)):
                if point.within(g5.loc[j].geometry):
                    g5_id = g5.loc[j].id
                    region_demand_g5[g5_id] += 1
                    break
            for j in range(len(g10)):
                if point.within(g10.loc[j].geometry):
                    g10_id = g10.loc[j].id
                    region_demand_g10[g10_id] += 1
                    break

        tmp_cnt = cnt1
        for i in range(tmp_cnt, len(df_edge1)):
            ts = df_edge1.iloc[i, :]['dropoff_datetime']
            x1 = df_edge1.iloc[i, :]['dropoff_longitude']
            y1 = df_edge1.iloc[i, :]['dropoff_latitude']
            if int(ts) > time:
                break

            cnt1 = i + 1
            point = Point([x1, y1])
            for j in range(len(g2)):
                if point.within(g2.loc[j].geometry):
                    g2_id = g2.loc[j].id
                    # print(g2_id, 'has a demand')
                    region_supply_g2[g2_id] += 1
                    break
            for j in range(len(g5)):
                if point.within(g5.loc[j].geometry):
                    g5_id = g5.loc[j].id
                    region_supply_g5[g5_id] += 1
                    break
            for j in range(len(g10)):
                if point.within(g10.loc[j].geometry):
                    g10_id = g10.loc[j].id
                    region_supply_g10[g10_id] += 1
                    break
    f.close()