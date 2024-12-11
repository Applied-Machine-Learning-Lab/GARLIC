# -*- coding: utf-8 -*-
# @Time : 2024/3/25 9:06
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : 04.demand_data_selection.py
# @Software: PyCharm
import math

import pymysql
import pandas as pd
from config import parser
from utils.sql_query import sql_to_file
from shapely.geometry import Point
import pyproj
from shapely.ops import transform

if __name__ == '__main__':
    args = parser.parse_args()

    # 设置经纬度起点坐标系
    in_proj = pyproj.CRS('EPSG:4326')
    # 设置投影坐标系
    out_proj = pyproj.CRS('EPSG:3857')
    project = pyproj.Transformer.from_crs(in_proj, out_proj, always_xy=True).transform
    project_inv = pyproj.Transformer.from_crs(out_proj, in_proj, always_xy=True).transform

    taxiid = pd.read_csv('../data/preprocess/processed_taxi_id.csv', encoding='utf8')

    fw = open('../' + args.ori_data, 'w', encoding='utf8')
    fw.write('VID,pickup_datetime,dropoff_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,distance,fee\n')

    conn = pymysql.connect(host=args.host, port=args.port, user=args.usr, password=args.passwd, db=args.db, cursorclass=pymysql.cursors.DictCursor)
    cursor = conn.cursor()

    data_count = {}
    car_id = 0
    for i in range(len(taxiid)):
        print('taxi:', taxiid.iloc[i].at['taxi_id'])

        sql_str = 'select LONGI, LATI, GPS_TIME, RUN_STATUS from ' + args.table + ' where PLA_NO = \"' + taxiid.iloc[i].at['taxi_id'] + '\" and LONGI >= ' + str(args.west_lon + 0.008) + ' and LONGI <= ' + str(args.east_lon - 0.006) + ' and LATI >= ' + str(args.south_lat + 0.011) + ' and LATI <= ' + str(args.north_lat - 0.005) + ' order by GPS_TIME'
        # print('\t'+sql_str)

        flag = True
        count_down = 5
        while (flag and count_down):
            flag = False
            try:
                cursor.execute(sql_str)
                result = cursor.fetchall()

                res = pd.DataFrame(result)
                row_sum = len(res)
                # print(res.head(70))

                # RUN_STATUS 1:空载 2:重载
                flag_pick = False
                last_status = 0
                pickup_time = ''
                pickup_lon = 0
                pickup_lat = 0
                for index, row in res.iterrows():
                    curr_status = row['RUN_STATUS']
                    # print(curr_status)
                    if last_status == 1 and curr_status == 2:
                        flag_pick = True
                        pickup_time = row['GPS_TIME'].strftime("%Y-%m-%d %H:%M:%S")
                        pickup_lon = row['LONGI']
                        pickup_lat = row['LATI']
                    elif flag_pick and curr_status == 1:
                        flag_pick = False

                        pickup_point = Point(pickup_lon, pickup_lat)
                        projected_pickup_point = transform(project, pickup_point)

                        dropoff_lon = row['LONGI']
                        dropoff_lat = row['LATI']

                        dropoff_point = Point(pickup_lon, pickup_lat)
                        projected_dropoff_point = transform(project, dropoff_point)

                        dis = math.fabs(projected_dropoff_point.x - projected_pickup_point.x) + math.fabs(projected_dropoff_point.y - projected_pickup_point.y)
                        dis = dis / 1000

                        fee = -1
                        if dis <= 3:
                            fee = 13.0
                        elif dis <= 10:
                            fee = 13.0 + (dis - 3) * 2.5
                        else:
                            fee = 30.5 + (dis - 10) * 3.75

                        fw.write(str(taxiid.iloc[i].at['id'])+','+pickup_time+','+row['GPS_TIME'].strftime("%Y-%m-%d %H:%M:%S")+','+str(pickup_lon)+','+str(pickup_lat)+','+str(dropoff_lon)+','+str(dropoff_lat)+','+str(dis)+','+str(fee)+'\n')
                    last_status = curr_status


            except Exception as e:
                print(e)
                # reconnect to the database
                conn = pymysql.connect(host=args.host, port=args.port, user=args.usr, password=args.passwd, db=args.db, cursorclass=pymysql.cursors.DictCursor)
                cursor = conn.cursor()
                flag = True
                count_down -= 1

    conn.close()
    fw.close()