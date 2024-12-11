# -*- coding: utf-8 -*-
# @Time : 2024/3/25 9:36
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : 01.data_sample.py
# @Software: PyCharm

import pandas as pd
import pymysql
from config import parser
from utils.sql_query import sql_to_file

# 完成后用arcgis看点位分布，微调路网框方位，重复步骤1

if __name__ == '__main__':
    args = parser.parse_args()

    data_dir = '../data/preprocess'

    time = [['2017-09-18 08:00:00','2017-09-18 08:01:00'],
            ['2017-09-18 13:00:00','2017-09-18 13:01:00'],
            ['2017-09-18 18:30:00','2017-09-18 18:31:00']
            ]

    conn = pymysql.Connect(host=args.host, port=args.port, user=args.usr, password=args.passwd, db=args.db)
    cursor = conn.cursor()

    for ti in range(3):
        file_name = 'location' + time[ti][0][10] + time[ti][0][11] + time[ti][0][13] + time[ti][0][14] + '.csv'
        sql_str = 'select LONGI, LATI from ' + args.table + ' where GPS_TIME >= \"' + time[ti][0] + '\" AND GPS_TIME <= \"' + time[ti][1] +'\"'
        print(sql_str)
        flag = True
        count_down = 5
        while (flag and count_down):
            flag = False
            try:
                cursor.execute(sql_str)
                result = cursor.fetchall()
                with open(data_dir + '/' + file_name, 'w', encoding='utf8') as f:
                    f.write('LON,LAT\n')
                    for row in result:
                        for i in range(len(row)):
                            if i > 0:
                                f.write(',')
                            f.write(str(row[i]))
                        f.write('\n')
                    f.close()
            except Exception as e:
                print(e)
                # reconnect to the database
                conn = pymysql.Connect(host=args.host, port=args.port, user=args.usr, password=args.passwd, db=args.db)
                cursor = conn.cursor()
                flag = True
                count_down -= 1
    conn.close()