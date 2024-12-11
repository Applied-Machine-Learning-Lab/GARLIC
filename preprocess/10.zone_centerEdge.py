# -*- coding: utf-8 -*-
# @Time : 2024/3/25 22:22
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : 10.zone_centerEdge.py
# @Software: PyCharm


#### 启动sumo

import os
import sys
import geopandas as gpd
import shapely.geometry.point as Point

# we need to import python modules from the $SUMO_HOME/tools directory
from config import parser

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import sumolib
from sumolib import checkBinary  # noqa
import traci  # noqa


args = parser.parse_args()

net = sumolib.net.readNet('../'+args.net_file, withInternal=False)


# traci.start(["sumo", "-c", "../"+args.pre_sim_file, "--tripinfo-output.write-unfinished", "--tripinfo-output",
#                  "../"+args.trip_info, "--stop-output", "../"+args.stop_info])

f = open('../'+args.hg2_central_edge, 'w', encoding='utf8')
f.write('{\n')
g2 = gpd.read_file('../data/taxi_zones/hg2')
g2 = g2.to_crs('EPSG:4326')
flag = True
for row in g2.itertuples():
    if flag:
        flag = False
    else:
        f.write(',\n')
    point:Point = row.geometry.centroid
    zoneid = row.id
    x, y = net.convertLonLat2XY(point.x, point.y)
    # edgeid = traci.simulation.convertRoad(point.x, point.y, isGeo=True)[0]
    # print('\tOri:',zoneid, edgeid)
    edges = net.getNeighboringEdges(x, y, args.hg2_radius)
    if len(edges) > 0:
        # edge, dist in edges
        distancesAndEdges = sorted(edges, key=(lambda x:x[1]), reverse=False)
        closestEdge, dist = distancesAndEdges[0]
        edgeid = closestEdge.getID()
        print('\tModi:',zoneid, edgeid)
        f.write('\t\"'+str(zoneid)+'\":\t\"'+str(edgeid)+'\"')
    else:
        radius = args.hg2_radius * 2
        while(True):
            edges = net.getNeighboringEdges(x, y, radius)
            if len(edges) > 0:
                # edge, dist in edges
                distancesAndEdges = sorted(edges, key=(lambda x: x[1]), reverse=False)
                closestEdge, dist = distancesAndEdges[0]
                edgeid = closestEdge.getID()
                print('\tModi:', zoneid, edgeid)
                f.write('\t\"' + str(zoneid) + '\":\t\"' + str(edgeid) + '\"')
                break
            radius = radius * 2
f.write('\n}\n')
f.close()

f = open('../'+args.hg5_central_edge, 'w', encoding='utf8')
f.write('{\n')
g5 = gpd.read_file('../data/taxi_zones/hg5')
g5 = g5.to_crs('EPSG:4326')
flag = True
for row in g5.itertuples():
    if flag:
        flag = False
    else:
        f.write(',\n')
    point:Point = row.geometry.centroid
    zoneid = row.id
    x, y = net.convertLonLat2XY(point.x, point.y)
    # edgeid = traci.simulation.convertRoad(point.x, point.y, isGeo=True)[0]
    # print('\tOri:',zoneid, edgeid)
    edges = net.getNeighboringEdges(x, y, args.hg5_radius)
    if len(edges) > 0:
        # edge, dist in edges
        distancesAndEdges = sorted(edges, key=(lambda x:x[1]), reverse=False)
        closestEdge, dist = distancesAndEdges[0]
        edgeid = closestEdge.getID()
        print('\tModi:',zoneid, edgeid)
        f.write('\t\"'+str(zoneid)+'\":\t\"'+str(edgeid)+'\"')
    else:
        radius = args.hg5_radius * 2
        while(True):
            edges = net.getNeighboringEdges(x, y, radius)
            if len(edges) > 0:
                # edge, dist in edges
                distancesAndEdges = sorted(edges, key=(lambda x: x[1]), reverse=False)
                closestEdge, dist = distancesAndEdges[0]
                edgeid = closestEdge.getID()
                print('\tModi:', zoneid, edgeid)
                f.write('\t\"' + str(zoneid) + '\":\t\"' + str(edgeid) + '\"')
                break
            radius = radius * 2
f.write('\n}\n')
f.close()

f = open('../'+args.hg10_central_edge, 'w', encoding='utf8')
f.write('{\n')
g10 = gpd.read_file('../data/taxi_zones/hg10')
g10 = g10.to_crs('EPSG:4326')
flag = True
for row in g10.itertuples():
    if flag:
        flag = False
    else:
        f.write(',\n')
    point:Point = row.geometry.centroid
    zoneid = row.id
    x, y = net.convertLonLat2XY(point.x, point.y)
    # edgeid = traci.simulation.convertRoad(point.x, point.y, isGeo=True)[0]
    # print('\tOri:',zoneid, edgeid)
    edges = net.getNeighboringEdges(x, y, args.hg10_radius)
    if len(edges) > 0:
        # edge, dist in edges
        distancesAndEdges = sorted(edges, key=(lambda x:x[1]), reverse=False)
        closestEdge, dist = distancesAndEdges[0]
        edgeid = closestEdge.getID()
        print('\tModi:',zoneid, edgeid)
        f.write('\t\"'+str(zoneid)+'\":\t\"'+str(edgeid)+'\"')
    else:
        radius = args.hg10_radius * 2
        while(True):
            edges = net.getNeighboringEdges(x, y, radius)
            if len(edges) > 0:
                # edge, dist in edges
                distancesAndEdges = sorted(edges, key=(lambda x: x[1]), reverse=False)
                closestEdge, dist = distancesAndEdges[0]
                edgeid = closestEdge.getID()
                print('\tModi:', zoneid, edgeid)
                f.write('\t\"' + str(zoneid) + '\":\t\"' + str(edgeid) + '\"')
                break
            radius = radius * 2
f.write('\n}\n')
f.close()