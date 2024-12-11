# -*- coding: utf-8 -*-
# @Time : 2024/3/25 7:43
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : 00.taxi_id_file.py
# @Software: PyCharm

import pymysql
from config import parser
from utils.sql_query import sql_to_file

if __name__ == '__main__':
    args = parser.parse_args()

    data_dir = '../data/preprocess'
    file_name = 'taxi_id.csv'

    conn = pymysql.Connect(host=args.host, port=args.port, user=args.usr, password=args.passwd, db=args.db)
    cursor = conn.cursor()

    sql_str = 'select distinct PLA_NO from ' + args.table + ' where PLA_NO like "浙AT%"'

    flag = True
    count_down = 5
    while (flag and count_down):
        flag = False
        try:
            cursor.execute(sql_str)
            result = cursor.fetchall()
            with open(data_dir + '/' + file_name, 'w', encoding='utf8') as f:
                f.write('taxi_id\n')
                for row in result:
                    f.write(row[0]+'\n')
                f.close()
        except Exception as e:
            print(e)
            # reconnect to the database
            conn = pymysql.Connect(host=args.host, port=args.port, user=args.usr, password=args.passwd, db=args.db)
            cursor = conn.cursor()
            flag = True
            count_down -= 1
    conn.close()