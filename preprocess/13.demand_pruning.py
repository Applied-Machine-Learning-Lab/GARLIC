# -*- coding: utf-8 -*-
# @Time : 2024/3/25 22:17
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : 13.demand_pruning.py
# @Software: PyCharm

import datetime
import pandas as pd
from config import parser
import xml.etree.ElementTree as ET

args = parser.parse_args()

#### 剪枝，筛选出固定时间间隔的demandtrips数据，保存为args.person_demand_after
h, m, s = map(int, args.start_time.split(':'))
StartTime = datetime.timedelta(hours=h, minutes=m, seconds=s)
h, m, s = map(int, args.end_time.split(':'))
EndTime = datetime.timedelta(hours=h, minutes=m, seconds=s)
if int(StartTime.total_seconds()) > 0:
    tree = ET.parse('../'+args.person_demand_before)
    root = tree.getroot()
    with open('../'+args.person_demand_after, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes>\n\t')
        for person in root.findall('person'):

            depart = person.attrib['depart']
            if int(EndTime.total_seconds()) > int(depart):
                # id=person.attrib['id']
                depart = str(int(depart) - int(StartTime.total_seconds()))
                person.set('depart', depart)
                if int(person.attrib['depart']) < 0:
                    id_0 = person.attrib['id']
                else:
                    id = person.attrib['id']
                    id = str(int(id) - int(id_0) - 1)
                    person.set('id', id)
                    f.write(ET.tostring(person, encoding='utf-8').decode('utf-8'))

        f.write('</routes>')
    data = pd.read_csv('../'+args.tmp_person_data)
    data = data.loc[(data['pickup_datetime'] >= int(StartTime.total_seconds())) & (
                data['pickup_datetime'] < int(EndTime.total_seconds()))]
    data.pickup_datetime = data.pickup_datetime - int(StartTime.total_seconds())

    data.to_csv('../'+args.tmp_person_data2, index=False)
else:
    df1 = pd.read_csv('../'+args.tmp_person_data)
    df1.to_csv('../'+args.tmp_person_data2, index=False)
    tree = ET.parse('../'+args.person_demand_before)
    tree.write('../'+args.person_demand_after)