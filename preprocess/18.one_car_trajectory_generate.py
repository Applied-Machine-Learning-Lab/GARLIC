# -*- coding: utf-8 -*-
# @Time : 2024/3/30 15:43
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : 18.one_car_trajectory_generate.py
# @Software: PyCharm

import datetime

import pymysql

from config import parser

if __name__ == '__main__':
    args = parser.parse_args()

    carID = '浙ATD753'

    h, m, s = map(int, args.start_time.split(':'))
    StartTime = datetime.timedelta(hours=h, minutes=m, seconds=s)
    h, m, s = map(int, args.end_time.split(':'))
    if h != 23:
        h = h + 1
    args.end_time = str(h).zfill(2) + ':' + str(m).zfill(2) + ':' + str(s).zfill(2)

    date = args.table[-4:]
    month = int(date[:2])
    day = int(date[-2:])
    gps_date = '2017-' + str(month) + '-' + str(day)

    conn = pymysql.connect(host=args.host, port=args.port, user=args.usr, password=args.passwd, db=args.db,
                           cursorclass=pymysql.cursors.DictCursor)
    cursor = conn.cursor()
    sql_str = 'select LONGI, LATI, GPS_TIME, RUN_STATUS from ' + args.table + ' where PLA_NO = \"' + carID + '\" and GPS_TIME >= \"' + gps_date + ' ' + args.start_time + '\" and GPS_TIME <= \"' + gps_date + ' ' + args.end_time + '\" order by GPS_TIME'
    print(sql_str)

    flag = True
    count_down = 5
    while (flag and count_down):
        flag = False
        try:
            cursor.execute(sql_str)
            result = cursor.fetchall()

            with open('../data/preprocess/' + args.table[-4:] + '/gps.csv', 'w') as f:
                f.write('timestep,lon,lat\n')
                for row in result:
                    dt = row['GPS_TIME'].strftime("%Y-%m-%d %H:%M:%S")
                    h = int(dt[11:13])
                    m = int(dt[14:15])
                    s = int(dt[17:19])
                    ts = 3600 * h + 60 * m + s

                    f.write(str(ts-int(StartTime.total_seconds()))+','+str(row['LONGI'])+','+str(row['LATI'])+'\n')

        except Exception as e:
            print(e)
            # reconnect to the database
            conn = pymysql.connect(host=args.host, port=args.port, user=args.usr, password=args.passwd, db=args.db,
                                   cursorclass=pymysql.cursors.DictCursor)
            cursor = conn.cursor()
            flag = True
            count_down -= 1

    conn.close()