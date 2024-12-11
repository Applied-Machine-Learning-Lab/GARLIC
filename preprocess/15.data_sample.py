# -*- coding: utf-8 -*-
# @Time : 2024/3/29 17:21
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : 15.data_sample.py
# @Software: PyCharm

import math
import pickle

import pymysql
import pandas as pd
import geopandas as gpd
import pyproj
from shapely.ops import transform

from config import parser
from shapely.geometry import Point

from utils.taxi_utils import Taxis

if __name__ == '__main__':
    args = parser.parse_args()

    taxiid = pd.read_csv('../data/preprocess/taxi_id.csv')

    conn = pymysql.connect(host=args.host, port=args.port, user=args.usr, password=args.passwd, db=args.db,
                           cursorclass=pymysql.cursors.DictCursor)
    cursor = conn.cursor()

    data_count = {}
    for i in range(len(taxiid)):
        carID = taxiid.iloc[i, 0]
        # print('taxi:', carID)

        sql_str = 'select LONGI, LATI, GPS_TIME, RUN_STATUS from ' + args.table + ' where PLA_NO = \"' + carID + '\" and LONGI >= ' + str(args.west_lon) + ' and LONGI <= ' + str(
            args.east_lon) + ' and LATI >= ' + str(args.south_lat) + ' and LATI <= ' + str(args.north_lat) + ' order by GPS_TIME'
        # print('\t' + sql_str)

        flag = True
        count_down = 5
        while (flag and count_down):
            flag = False
            try:
                cursor.execute(sql_str)
                result = cursor.fetchall()

                res = pd.DataFrame(result)
                row_sum = len(res)

                df = res.iloc[:, :2]
                duplicated_sum = df.duplicated().sum()
                if duplicated_sum < 400 and row_sum > 300:
                    data_count[taxiid.iloc[i, 0]] = len(res)

            except Exception as e:
                print(e)
                # reconnect to the database
                conn = pymysql.connect(host=args.host, port=args.port, user=args.usr, password=args.passwd, db=args.db,
                                       cursorclass=pymysql.cursors.DictCursor)
                cursor = conn.cursor()
                flag = True
                count_down -= 1

    sorted_data = sorted(data_count.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    print(sorted_data[:500])

    carIDs = []
    cnt = 0
    for k, v in sorted_data:
        print(k, '\t', v)
        carIDs.append(k)
        cnt += 1
        if cnt == args.max_car_num:
            break
    f = open('../data/preprocess/tmp_carIDs.pkl', 'wb')
    pickle.dump(carIDs, f)
    f.close()

    print('\n\nStep 1: finished\n')

    TOTOAL_DATETIME = 24 * 3600

    # 设置经纬度起点坐标系
    in_proj = pyproj.CRS('EPSG:4326')
    # 设置投影坐标系
    out_proj = pyproj.CRS('EPSG:3857')
    project = pyproj.Transformer.from_crs(in_proj, out_proj, always_xy=True).transform
    project_inv = pyproj.Transformer.from_crs(out_proj, in_proj, always_xy=True).transform

    g2 = gpd.read_file('../data/taxi_zones/hg2')
    g5 = gpd.read_file('../data/taxi_zones/hg5')
    g10 = gpd.read_file('../data/taxi_zones/hg10')

    f = open('../data/preprocess/taxi_region_data.csv', 'w', encoding='utf8')
    f.write('timestep,vID,hg2,hg5,hg10,lon,lat,dis,degree,orderdis,orderfee\n')

    taxis = Taxis()

    for id in carIDs:
        print('carID:', id)
        sql_str = 'select LONGI, LATI, GPS_TIME, RUN_STATUS from ' + args.table + ' where PLA_NO = \"' + id + '\" order by GPS_TIME'
        flag = True
        count_down = 5
        while (flag and count_down):
            flag = False
            try:
                cursor.execute(sql_str)
                result = cursor.fetchall()
            except Exception as e:
                print(e)
                conn = pymysql.connect(host=args.host, port=args.port, user=args.usr, password=args.passwd,
                                       db=args.db,
                                       cursorclass=pymysql.cursors.DictCursor)
                cursor = conn.cursor()
                flag = True
                count_down -= 1

        cnt = 0
        time = args.rebalance_time
        last_time = 0
        last_point = None
        while (True):
            last_time = args.rebalance_time * cnt
            time = args.rebalance_time * (cnt + 1)
            if last_time > TOTOAL_DATETIME:
                break

            count = 0
            x_sum = 0.
            y_sum = 0.

            for row in result:
                status = row['RUN_STATUS']
                if taxis.get_status(id) <= 1 and status == 2:
                    taxis.set_pos(id, row['LONGI'], row['LATI'], pickup=True)
                elif status == 2:
                    taxis.set_pos(id, row['LONGI'], row['LATI'])
                else:
                    taxis.set_status(id, status)

                dt = row['GPS_TIME'].strftime("%Y-%m-%d %H:%M:%S")
                h = int(dt[11:13])
                m = int(dt[14:16])
                s = int(dt[17:19])
                ts = 3600 * h + 60 * m + s
                if ts < last_time:
                    continue
                elif ts >= time:
                    if count != 0:
                        x_mean = x_sum / count
                        y_mean = y_sum / count
                        point_xy = Point(x_mean, y_mean)
                        point_lonlat = transform(project_inv, point_xy)
                        hg2_id = ''
                        hg5_id = ''
                        hg10_id = ''
                        for j in range(len(g2)):
                            if projected_point.within(g2.loc[j].geometry):
                                hg2_id = int(g2.loc[j].id) - 2000
                                break
                        for j in range(len(g5)):
                            if projected_point.within(g5.loc[j].geometry):
                                hg5_id = int(g5.loc[j].id) - 5000
                                break
                        for j in range(len(g10)):
                            if projected_point.within(g10.loc[j].geometry):
                                hg10_id = int(g10.loc[j].id) - 10000
                                break
                        dis = ''
                        degree = ''
                        if last_point != None:
                            dis = point_xy.distance(last_point)
                            if dis == 0:
                                degree = 0
                            else:
                                x0 = last_point.x
                                y0 = last_point.y
                                x1 = point_xy.x
                                y1 = point_xy.y

                                val = (x1 - x0) / dis
                                if y1 >= y0:
                                    degree = math.acos(val)
                                elif y1 < y0:
                                    degree = math.pi + math.acos(-val)

                        f.write(str(cnt)+','+str(id)+','+str(hg2_id)+','+str(hg5_id)+','+str(hg10_id)+','+str(point_lonlat.x)+','+str(point_lonlat.y)+','+str(dis)+','+str(degree)+','+str(taxis.get_dis(id))+','+str(taxis.get_fee(id))+'\n')
                        last_point = point_xy
                    elif count == 0:
                        last_point = None
                        f.write(str(cnt) + ',' + str(id) + ',' + ',' + ',' + ',' + ',' + ',' + ',' + ',' + ',' + '\n')
                    break
                else:
                    lon = row['LONGI']
                    lat = row['LATI']
                    point = Point(lon, lat)
                    projected_point = transform(project, point)
                    x_sum += projected_point.x
                    y_sum += projected_point.y
                    count += 1
            cnt += 1

    conn.close()
    f.close()


    print('\n\nStep 2: finished\n')