# -*- coding: utf-8 -*-
# @Time : 2024/3/27 9:45
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : 11.zone_edge_relation.py
# @Software: PyCharm

import sumolib
import geopandas as gpd
from shapely.geometry import Polygon, Point

from config import parser

args = parser.parse_args()

g2 = gpd.read_file('../data/taxi_zones/hg2')
g2 = g2.to_crs('EPSG:4326')

g5 = gpd.read_file('../data/taxi_zones/hg5')
g5 = g5.to_crs('EPSG:4326')

g10 = gpd.read_file('../data/taxi_zones/hg10')
g10 = g10.to_crs('EPSG:4326')

f = open('../'+args.zone_edge, 'w')
f.write('edgeID, hg2_zoneID, hg5_zoneID, hg10_zoneID\n')

net = sumolib.net.readNet('../'+args.net_file, withInternal=False)
for edge in sumolib.xml.parse('../'+args.net_file, ['edge']):
    if net.hasEdge(edge.id):
        f.write('\"'+str(edge.id)+'\",\"')
        x, y = net.getEdge(edge.id).getFromNode().getCoord()
        lon, lat = net.convertXY2LonLat(x,y)
        point = Point(lon,lat)
        for g2_row in g2.itertuples():
            hexagon2: Polygon = g2_row.geometry
            g2_id = g2_row.id
            if hexagon2.contains(point):
                f.write(str(g2_id))
                break
        f.write('\",\"')
        for g5_row in g5.itertuples():
            hexagon5: Polygon = g5_row.geometry
            g5_id = g5_row.id
            if hexagon5.contains(point):
                f.write(str(g5_id))
                break
        f.write('\",\"')
        for g10_row in g10.itertuples():
            hexagon10: Polygon = g10_row.geometry
            g10_id = g10_row.id
            if hexagon10.contains(point):
                f.write(str(g10_id))
                break
        f.write('\"\n')

f.close()