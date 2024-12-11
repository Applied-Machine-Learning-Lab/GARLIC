# -*- coding: utf-8 -*-
# @Time : 2024/3/25 17:25
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : 08.taxi_zones_show.py
# @Software: PyCharm

import geopandas as gpd
import matplotlib.pyplot as plt


FIGURE_WIDTH = 50
FIGURE_HEIGHT = 50

all_areas = gpd.read_file(r'../data/taxi_zones/all')
road_network = gpd.read_file(r'../data/road_centerline/shp/edges.shp')
road_network = road_network.to_crs('EPSG:3857')
g2 = gpd.read_file('../data/taxi_zones/hg2')
g5 = gpd.read_file('../data/taxi_zones/hg5')
g10 = gpd.read_file('../data/taxi_zones/hg10')
ax = g2.plot(fc='none',ec='grey',figsize=(FIGURE_WIDTH,FIGURE_HEIGHT))
g5.plot(fc='none',ec='blue',ax=ax,alpha=0.5)
g10.plot(fc='none',ec='orange',ax=ax, alpha=0.5)
road_network.plot(fc='none', ec='red',ax=ax, alpha=0.15)
# all_areas.plot(fc='green', ec='red',ax=ax, alpha=0.3)
plt.savefig('../logs/pic/all_zones.png')
plt.show()

ax = g2.plot(fc='none',ec='black',linewidth=5,figsize=(FIGURE_WIDTH,FIGURE_HEIGHT))
# road_network.plot(fc='none', ec='green',ax=ax, alpha=0.1)
# # 标注每个网格的id
# for row in g2.itertuples():
#     centroid = row.geometry.centroid
#     ax.text(centroid.x, centroid.y, row.id, ha='center', va='center')
plt.savefig('../logs/pic/hg2_zones.png',transparent = True)
plt.show()

ax = g5.plot(fc='none',ec='black',linewidth=5,figsize=(FIGURE_WIDTH,FIGURE_HEIGHT))
# road_network.plot(fc='none', ec='green',ax=ax, alpha=0.1)
# # 标注每个网格的id
# for row in g5.itertuples():
#     centroid = row.geometry.centroid
#     ax.text(centroid.x, centroid.y, row.id, ha='center', va='center')
plt.savefig('../logs/pic/hg5_zones.png',transparent = True)
plt.show()

ax = g10.plot(fc='none',ec='black',linewidth=5,figsize=(FIGURE_WIDTH,FIGURE_HEIGHT))
# road_network.plot(fc='none', ec='green',ax=ax, alpha=0.1)
# # 标注每个网格的id
# for row in g10.itertuples():
#     centroid = row.geometry.centroid
#     ax.text(centroid.x, centroid.y, row.id, ha='center', va='center')
plt.savefig('../logs/pic/hg10_zones.png',transparent = True)
plt.show()