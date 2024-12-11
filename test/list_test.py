# -*- coding: utf-8 -*-
# @Time : 2024/4/4 11:06
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : list_test.py
# @Software: PyCharm
from config import parser

list = [1, 2, 3]
list2 = [2,3,4]
print(list + list2)

args = parser.parse_args()

print(args.table[-4:])

date = args.table[-4:]
month = int(date[:2])
day = int(date[-2:])
print(month, day)

ss_str = '2017-09-18 00:00:00'
print(ss_str[11:13])