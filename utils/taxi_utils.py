# -*- coding: utf-8 -*-
# @Time : 2023/4/17 9:58
# @Author : Xiao Han
# @E-mail : hahahenha@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : taxi_utils.py
# @Software: PyCharm

import math
import pandas as pd
from config import parser
from shapely.geometry import Point
import pyproj
from shapely.ops import transform

class Taxis:
    def __init__(self):
        # 设置经纬度起点坐标系
        in_proj = pyproj.CRS('EPSG:4326')
        # 设置投影坐标系
        out_proj = pyproj.CRS('EPSG:3857')
        self.project = pyproj.Transformer.from_crs(in_proj, out_proj, always_xy=True).transform
        self.project_inv = pyproj.Transformer.from_crs(out_proj, in_proj, always_xy=True).transform

        # car ID : float
        self.start_pos_x = {}
        self.start_pos_y = {}
        self.cur_pos_x = {}
        self.cur_pos_y = {}

        self.dis = {}
        self.status = {}

        self.fee = {}

    def set_pos(self, vehID, lon, lat, isGeo=True, pickup=False):
        if isGeo:
            point = Point(lon, lat)
            projected_point = transform(self.project, point)
            self.cur_pos_x[vehID] = projected_point.x
            self.cur_pos_y[vehID] = projected_point.y
        else:
            self.cur_pos_x[vehID] = lon
            self.cur_pos_y[vehID] = lat

        if pickup:
            self.status[vehID] = 2
            self.start_pos_x[vehID] = self.cur_pos_x[vehID]
            self.start_pos_y[vehID] = self.cur_pos_y[vehID]

        if self.status[vehID] == 2:
            dis = math.fabs(self.start_pos_x[vehID] - self.cur_pos_x[vehID]) + math.fabs(self.start_pos_y[vehID] - self.cur_pos_y[vehID])
            dis = dis / 1000

            fee = -1
            if dis <= 3:
                fee = 13.0
            elif dis <= 10:
                fee = 13.0 + (dis - 3) * 2.5
            else:
                fee = 30.5 + (dis - 10) * 3.75
            self.dis[vehID] = dis
            self.fee[vehID] = fee
        else:
            self.dis[vehID] = 0.0
            self.fee[vehID] = 0.0

    def set_status(self, vehID, status):
        self.status[vehID] = status
        if status != 2:
            self.start_pos_x[vehID] = -1
            self.start_pos_y[vehID] = -1
            self.dis[vehID] = 0.0
            self.fee[vehID] = 0.0

    def get_fee(self, vehID):
        if vehID not in self.fee.keys():
            self.fee[vehID] = 0.0
        return self.fee[vehID]

    def get_dis(self, vehID):
        if vehID not in self.dis.keys():
            self.dis[vehID] = 0.0
        return self.dis[vehID]

    def get_status(self, vehID):
        if vehID not in self.status.keys():
            self.status[vehID] = 0
        return self.status[vehID]


