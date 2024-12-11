# -*- coding: utf-8 -*-
# @Time : 2024/3/25 9:20
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : 02.osm_generate.py
# @Software: PyCharm
import os

import osmnx as ox

from config import parser
import matplotlib.pyplot as plt

args = parser.parse_args()

G_old = ox.graph_from_bbox(args.north_lat, args.south_lat, args.east_lon, args.west_lon, network_type='drive')
G_projected = ox.project_graph(G_old, to_crs='EPSG:4326')
fig, ax = ox.plot_graph(G_projected, node_size=2, node_color='r', edge_color='w', edge_linewidth=0.2)
plt.show()
# ox.save_graph_shapefile(G, filepath="../data/taxi_zones/out/")
ox.save_graph_xml(G_projected, filepath='../' + args.road_centerline)
ox.save_graph_shapefile(G_projected, filepath='../data/road_centerline/shp')

os.system("netconvert --osm-files ../data/road_centerline/road_centerline.osm --output-file ../data/sumo_files/convert.net.xml --geometry.remove --roundabouts.guess --ramps.guess --junctions.join --tls.guess-signals --tls.discard-simple --tls.join")
