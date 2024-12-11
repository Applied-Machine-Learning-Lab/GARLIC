# -*- coding: utf-8 -*-
# @Time : 2024/3/25 14:55
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : 05.time_format_change.py
# @Software: PyCharm

import pymysql
import pandas as pd
from config import parser


if __name__ == '__main__':
    args = parser.parse_args()

    df = pd.read_csv('../'+args.ori_data, encoding='utf8')
    print(df.head())

    for i in range(0, len(df)):
        h = int(df.loc[i, 'dropoff_datetime'][11:13])
        m = int(df.loc[i, 'dropoff_datetime'][14:16])
        s = int(df.loc[i, 'dropoff_datetime'][17:19])
        # print('h:',h, 'm:', m, 's:', s)
        df.loc[i, 'pickup_datetime'] = 3600 * h + 60 * m + s

    for i in range(0, len(df)):
        h = int(df.loc[i, 'dropoff_datetime'][11:13])
        m = int(df.loc[i, 'dropoff_datetime'][14:16])
        s = int(df.loc[i, 'dropoff_datetime'][17:19])
        # print('h:',h, 'm:', m, 's:', s)
        df.loc[i, 'dropoff_datetime'] = 3600 * h + 60 * m + s
    print(df.head())
    df.sort_values("pickup_datetime", inplace=True)

    df = df[['pickup_datetime', 'dropoff_datetime', 'pickup_longitude', 'pickup_latitude',
             'dropoff_longitude', 'dropoff_latitude', 'distance', 'fee', 'VID']]
    df.to_csv('../'+args.tmp_time_change_data, index=False)


