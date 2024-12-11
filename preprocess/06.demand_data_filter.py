# -*- coding: utf-8 -*-
# @Time : 2024/3/25 15:22
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : 06.demand_data_filter.py
# @Software: PyCharm

import pandas as pd
from config import parser

if __name__ == '__main__':
    args = parser.parse_args()

    df = pd.read_csv('../' + args.tmp_time_change_data, encoding='utf8')
    print(df.head())

    INTERVAL = 4

    for i in range(25 - INTERVAL):
        time_s = i * 3600
        time_e = time_s + INTERVAL * 3600
        df_res = df[(df['pickup_datetime'] >= time_s) & (df['pickup_datetime'] <= time_e)]
        print(str(i)+'点-'+str(i+4)+'点:'+str(len(df_res)))

# 0点-4点:795
# 1点-5点:628
# 2点-6点:578
# 3点-7点:618
# 4点-8点:822
# 5点-9点:1091
# 6点-10点:1278
# 7点-11点:1499
# 8点-12点:1593
# 9点-13点:1624
# 10点-14点:1670
# 11点-15点:1620
# 12点-16点:1514
# 13点-17点:1431
# 14点-18点:1390
# 15点-19点:1416
# 16点-20点:1570
# 17点-21点:1710
# 18点-22点:1712
# 19点-23点:1596
# 20点-24点:1262