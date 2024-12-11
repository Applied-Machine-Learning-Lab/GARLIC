# -*- coding: utf-8 -*-
# @Time : 2024/3/25 16:20
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : 07.hexagon_grids_generate.py
# @Software: PyCharm

import geopandas as gpd
import matplotlib.pyplot as plt

import warnings
from config import parser
from utils.hexagon_grid import create_Honeycomb_Polygon
from shapely.geometry import Polygon
import os
import shutil

def produce_rect_by_conner(L_lon_value, R_lon_value, L_Lat_value, T_lat_value):
    """左下角逆时针开始"""
    LL = (L_lon_value, L_Lat_value)  # 左下角
    RL = (R_lon_value, L_Lat_value)
    RT = (R_lon_value, T_lat_value)
    LT = (L_lon_value, T_lat_value)
    return Polygon([LL, RL, RT, LT])

def move(old_path, new_path):
    # if not os.path.exists(target_dir):
    #     os.makedirs(target_dir)
    # else:
    #     os.removedirs(target_dir)
    #     os.makedirs(target_dir)
    print(old_path)
    print(new_path)
    filelist = os.listdir(old_path)  # 列出该目录下的所有文件,listdir返回的文件列表是不包含路径的。
    print(filelist)
    for file in filelist:
        src = os.path.join(old_path, file)
        dst = os.path.join(new_path, file)
        print('src:', src)
        print('dst:', dst)
        shutil.move(src, dst)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    args = parser.parse_args()

    all_areas = gpd.GeoSeries([produce_rect_by_conner(args.west_lon, args.east_lon, args.south_lat, args.north_lat)],
                                index=['all_area'],  # 构建一个索引字段
                                crs='EPSG:4326',  # 坐标系是：WGS 1984
                                )
    all_areas = all_areas.to_crs('EPSG:3857')
    all_areas.to_file('../data/taxi_zones/all')
    all_areas.plot()
    # plt.show()


    create_Honeycomb_Polygon(vectorfile=r'../data/taxi_zones/all',outfile='../data/taxi_zones/hg2/hg2.shp',gridwidth=args.hg2_radius,gridheight=args.hg2_radius)
    create_Honeycomb_Polygon(vectorfile=r'../data/taxi_zones/all',outfile='../data/taxi_zones/hg5/hg5.shp',gridwidth=args.hg5_radius,gridheight=args.hg5_radius)
    create_Honeycomb_Polygon(vectorfile=r'../data/taxi_zones/all',outfile='../data/taxi_zones/hg10/hg10.shp',gridwidth=args.hg10_radius,gridheight=args.hg10_radius)