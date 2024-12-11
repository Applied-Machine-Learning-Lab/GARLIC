# -*- coding: utf-8 -*-
# @Time : 2024/3/25 21:38
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : 12.sumo_personDemandtrips_generate.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import pandas as pd
from config import parser
from xml.dom.minidom import Document

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import sumolib
from sumolib import checkBinary  # noqa
import traci  # noqa

args = parser.parse_args()

### 启动sumo
traci.start(["sumo", "-c", "../" + args.pre_sim_file, "--tripinfo-output.write-unfinished", "--tripinfo-output",
                 "../"+args.trip_info, "--stop-output", "../"+args.stop_info])

### 地理经纬度信息转换为sumo可用的edge信息
df_edge = pd.read_csv('../'+args.tmp_time_change_data)
print(df_edge.head())

df_edge.insert(1,'fe',"")
df_edge.insert(1,'te',"")
df_edge1 =df_edge.copy()
# print(df_edge1.head())

net = sumolib.net.readNet('../'+args.net_file, withInternal=False)

cnt = 0
for i in range(0,len(df_edge1)):
    x=df_edge.loc[i,'pickup_longitude']
    y=df_edge.loc[i,'pickup_latitude']
    x1=df_edge.loc[i,'dropoff_longitude']
    y1=df_edge.loc[i,'dropoff_latitude']
    # print(traci.simulation.convertRoad(x,y,isGeo = True))
    cnt+=1

    # df_edge1.loc[i,'fe'] = traci.simulation.convertRoad(x,y,isGeo = True)[0]
    xx, yy = net.convertLonLat2XY(x, y)
    edges = net.getNeighboringEdges(xx, yy,args.hg2_radius)
    if len(edges) > 0:
        # edge, dist in edges
        distancesAndEdges = sorted(edges, key=(lambda x:x[1]), reverse=False)
        closestEdge, dist = distancesAndEdges[0]
        edgeid = closestEdge.getID()
        df_edge1.loc[i, 'fe'] = edgeid

    # df_edge1.loc[i,'te'] = traci.simulation.convertRoad(x1,y1,isGeo = True)[0]
    xx, yy = net.convertLonLat2XY(x1, y1)
    edges = net.getNeighboringEdges(xx, yy, args.hg2_radius)
    if len(edges) > 0:
        # edge, dist in edges
        distancesAndEdges = sorted(edges, key=(lambda x: x[1]), reverse=False)
        closestEdge, dist = distancesAndEdges[0]
        edgeid = closestEdge.getID()
        df_edge1.loc[i,'te'] = edgeid

    cnt+=1
print(cnt)
df_edge1.head()
df_edge1.to_csv('../'+args.tmp_edge_done_data,index=False)

### 利用sumolib工具getClosestLanePosDist()，对数据进行地理位置最近距离估算
data2 = pd.read_csv('../'+args.tmp_edge_done_data)

# parse the net
edges = net.getEdges(False)
data2['fpos'] = -1
data2['tpos'] = -1
data2['fdist'] = -1
data2['tdist'] = -1
for i in range(0, len(data2)):
    fe = data2.loc[i, 'fe']
    if fe == "" or fe == None or type(fe) != str:
        continue
    fromX = data2.loc[i, 'pickup_longitude']
    fromY = data2.loc[i, 'pickup_latitude']

    fromEdge = net.getEdge(fe)

    fromLaneNo, fromPos, fromDist = fromEdge.getClosestLanePosDist(net.convertLonLat2XY(fromX, fromY))
    data2.loc[i, 'fpos'] = fromPos
    data2.loc[i, 'fdist'] = fromDist

    te = data2.loc[i, 'te']
    if te == "" or te == None or type(te) != str:
        continue
    toX = data2.loc[i, 'dropoff_longitude']
    toY = data2.loc[i, 'dropoff_latitude']

    toEdge = net.getEdge(te)

    toLaneNo, toPos, toDist = toEdge.getClosestLanePosDist(net.convertLonLat2XY(toX, toY))
    data2.loc[i, 'tpos'] = toPos
    data2.loc[i, 'tdist'] = toDist

    # if i % 100 == 0:
    #     print(i)
data2.to_csv('../'+args.tmp_person_data, index=False)
df1 = data2.describe()
print(df1.head())

### 将最终处理好的数据写入到xml中，得到最终convert-demandtrips.xml文件
doc = Document()

# 创建根节点
routes = doc.createElement('routes')
# 修改或添加节点中元素内容
df =pd.read_csv('../'+args.tmp_person_data)
print('person_data_len:', len(df))
# 根节点插入dom树
doc.appendChild(routes)

# 每一组信息先创建节点<order>，然后插入到父节点<modify_node>下
cnt = 0
for i in range(0,len(df)):
    id = str(cnt)
    depart = str(df.loc[i,"pickup_datetime"])
    fe = str(df.loc[i,"fe"])
    te = str(df.loc[i,"te"])
    fpos = str(df.loc[i,"fpos"])
    tpos = str(df.loc[i,"tpos"])

    try:
        if len(traci.simulation.findRoute(fromEdge=fe, toEdge=te, vType="DEFAULT_VEHTYPE").edges) > 0 and len(traci.simulation.findRoute(fromEdge=te, toEdge=fe, vType="DEFAULT_VEHTYPE").edges) > 0:
            person = doc.createElement('person')
            person.setAttribute("id", id)
            person.setAttribute("depart", depart)
            person.setAttribute("departPos", fpos)
            routes.appendChild(person)

            personTrip = doc.createElement('personTrip')
            personTrip.setAttribute('from', fe)
            personTrip.setAttribute('to', te)
            personTrip.setAttribute('arrivalPos', tpos)
            personTrip.setAttribute('modes', "taxi")
            person.appendChild(personTrip)
            cnt += 1
    except:
        print(i, 'Route error!!!')
        continue
print(cnt)

# 将dom对象写入本地xml文件
with open('../'+args.person_demand_before, 'w') as f:
    doc.writexml(f, indent='',addindent='  ',newl='\n',encoding='UTF-8')

traci.close()