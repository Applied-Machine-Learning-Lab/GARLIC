# -*- coding: utf-8 -*-
# @Time : 2024/3/25 20:44
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : 09.adjacency_zones_relation_find.py
# @Software: PyCharm

from shapely.geometry import Polygon
import geopandas as gpd
import matplotlib.pyplot as plt
from config import parser

args = parser.parse_args()

# direction:
#         2
#     3       1
#         0
#     4       6
#         5

f = open('../' + args.hg2_relation, 'w', encoding='utf8')
f.write('zoneID_from,zoneID_to,direction\n')
g2 = gpd.read_file('../data/taxi_zones/hg2')
for row1 in g2.itertuples():
    hexagon1:Polygon = row1.geometry
    id1 = row1.id
    for row2 in g2.itertuples():
        hexagon2: Polygon = row2.geometry
        id2 = row2.id
        if id1 != id2 and hexagon1.distance(hexagon2) == 0:
            point1 = hexagon1.centroid
            x1 = point1.x
            y1 = point1.y
            point2 = hexagon2.centroid
            x2 = point2.x
            y2 = point2.y
            dir = -1
            if y2 > y1:
                if x2 > x1:
                    dir = 1
                elif x2 == x1:
                    dir = 2
                elif x2 < x1:
                    dir = 3
                else:
                    print('error!!!')
            elif y2 < y1:
                if x2 > x1:
                    dir = 6
                elif x2 == x1:
                    dir = 5
                elif x2 < x1:
                    dir = 4
                else:
                    print('error!!!')
            else:
                print('error!!!')
            f.write(str(id1)+','+str(id2)+','+str(dir)+'\n')
f.close()

f = open('../' + args.hg5_relation, 'w', encoding='utf8')
f.write('zoneID_from,zoneID_to,direction\n')
g5 = gpd.read_file('../data/taxi_zones/hg5')
for row1 in g5.itertuples():
    hexagon1:Polygon = row1.geometry
    id1 = row1.id
    for row2 in g5.itertuples():
        hexagon2: Polygon = row2.geometry
        id2 = row2.id
        if id1 != id2 and hexagon1.distance(hexagon2) == 0:
            point1 = hexagon1.centroid
            x1 = point1.x
            y1 = point1.y
            point2 = hexagon2.centroid
            x2 = point2.x
            y2 = point2.y
            dir = -1
            if y2 > y1:
                if x2 > x1:
                    dir = 1
                elif x2 == x1:
                    dir = 2
                elif x2 < x1:
                    dir = 3
                else:
                    print('error!!!')
            elif y2 < y1:
                if x2 > x1:
                    dir = 6
                elif x2 == x1:
                    dir = 5
                elif x2 < x1:
                    dir = 4
                else:
                    print('error!!!')
            else:
                print('error!!!')
            f.write(str(id1)+','+str(id2)+','+str(dir)+'\n')
f.close()

f = open('../' + args.hg10_relation, 'w', encoding='utf8')
f.write('zoneID_from,zoneID_to,direction\n')
g10 = gpd.read_file('../data/taxi_zones/hg10')
for row1 in g10.itertuples():
    hexagon1:Polygon = row1.geometry
    id1 = row1.id
    for row2 in g10.itertuples():
        hexagon2: Polygon = row2.geometry
        id2 = row2.id
        if id1 != id2 and hexagon1.distance(hexagon2) == 0:
            point1 = hexagon1.centroid
            x1 = point1.x
            y1 = point1.y
            point2 = hexagon2.centroid
            x2 = point2.x
            y2 = point2.y
            dir = -1
            if y2 > y1:
                if x2 > x1:
                    dir = 1
                elif x2 == x1:
                    dir = 2
                elif x2 < x1:
                    dir = 3
                else:
                    print('error!!!')
            elif y2 < y1:
                if x2 > x1:
                    dir = 6
                elif x2 == x1:
                    dir = 5
                elif x2 < x1:
                    dir = 4
                else:
                    print('error!!!')
            else:
                print('error!!!')
            f.write(str(id1)+','+str(id2)+','+str(dir)+'\n')
f.close()