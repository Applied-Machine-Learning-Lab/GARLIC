# -*- coding: utf-8 -*-
# @Time : 2023/3/25 7:46
# @Author : Xiao Han
# @E-mail : hahahenha@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : sql_query.py
# @Software: PyCharm

import pymysql

def sql_to_file(args, sql_str, dir, file_name):
    conn = pymysql.Connect(host=args.host, port=args.port, user=args.usr, password=args.passwd, db=args.db)
    cursor = conn.cursor()
    flag = True
    count_down = 5
    while (flag and count_down):
        flag = False
        try:
            cursor.execute(sql_str)
            result = cursor.fetchall()
            with open(dir+'/'+file_name, 'w') as f:
                for i in range(len(result[0])):
                    if i > 0:
                        f.write(',')
                    f.write('col'+str(i+1))
                f.write('\n')
                flag = False
                for row in result:
                    for i in range(len(row)):
                        if i > 0:
                            f.write(',')
                        f.write(row[i])
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
    return not flag