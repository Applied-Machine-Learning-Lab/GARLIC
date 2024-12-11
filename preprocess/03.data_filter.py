# -*- coding: utf-8 -*-
# @Time : 2024/3/25 11:15
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : 03.data_filter.py
# @Software: PyCharm

import pymysql
import pandas as pd
from config import parser
from utils.sql_query import sql_to_file

if __name__ == '__main__':
    args = parser.parse_args()

    taxiid = pd.read_csv('../data/preprocess/taxi_id.csv')

    conn = pymysql.connect(host=args.host, port=args.port, user=args.usr, password=args.passwd, db=args.db, cursorclass=pymysql.cursors.DictCursor)
    cursor = conn.cursor()

    data_count = {}
    for i in range(len(taxiid)):

        sql_str = 'select LONGI, LATI, GPS_TIME, RUN_STATUS from ' + args.table + ' where PLA_NO = \"' + taxiid.iloc[i, 0] + '\" and LONGI >= ' + str(args.west_lon + 0.008) + ' and LONGI <= ' + str(args.east_lon - 0.006) + ' and LATI >= ' + str(args.south_lat + 0.011) + ' and LATI <= ' + str(args.north_lat - 0.005) + ' order by GPS_TIME'

        flag = True
        count_down = 5
        while (flag and count_down):
            flag = False
            try:
                cursor.execute(sql_str)
                result = cursor.fetchall()

                res = pd.DataFrame(result)
                row_sum = len(res)

                # print(res.head())
                df = res.iloc[:,:2]
                duplicated_sum = df.duplicated().sum()
                # print(df.head())
                if duplicated_sum < 400:
                    data_count[taxiid.iloc[i, 0]] = len(res)
                    # print('taxi:', taxiid.iloc[i, 0])
                    # print('\t' + sql_str)
                    # print('\t', row_sum, duplicated_sum)

            except Exception as e:
                print(e)
                # reconnect to the database
                conn = pymysql.connect(host=args.host, port=args.port, user=args.usr, password=args.passwd, db=args.db, cursorclass=pymysql.cursors.DictCursor)
                cursor = conn.cursor()
                flag = True
                count_down -= 1

    conn.close()

    sorted_data = sorted(data_count.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    print(sorted_data[:500])

    cnt = 0
    with open('../data/preprocess/processed_taxi_id.csv', 'w', encoding='utf8') as f:
        f.write('id,taxi_id\n')
        for k, v in sorted_data:
            print(k, '\t', v)
            if v < 200:
                break
            f.write(str(cnt)+',\"'+str(k) + '\"\n')
            cnt += 1
            # if cnt == 500:
            #     break