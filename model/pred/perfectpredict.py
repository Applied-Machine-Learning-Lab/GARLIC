# -*- coding: utf-8 -*-
# @Time : 2024/3/21 11:24
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : perfectpredict.py
# @Software: PyCharm
import json

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from config import parser


class PerfectPred(object):

    def __init__(self, args) -> None:
        self.data = pd.read_csv(args.tmp_person_data2)
        with open(args.hg2_central_edge, 'r') as centralEdgesFile:
            centralEdges: dict = json.load(centralEdgesFile)
            tmp_region_list = list(centralEdges.keys())
            self.region_list =  [ int(x) for x in tmp_region_list]
            # print(self.region_list)
        self.region_list.sort()
        geodata = gpd.read_file(args.hg2_zones_shp)
        geodata = geodata.loc[geodata.id.isin(self.region_list)]
        self.dataWGS84 = geodata.to_crs('EPSG:4326')
        del geodata

    def getPrediction(self, now, span):
        span = now + span
        df = self.data.loc[(self.data['pickup_datetime'] >= now) & (self.data['pickup_datetime'] < span)]
        point_list = [Point(df.pickup_longitude[i], df.pickup_latitude[i]) for i in df.index.values]

        id_list = [self.dataWGS84.contains(i)[self.dataWGS84.contains(i) == True].index[0] + 1 for i in point_list]
        # print(id_list)
        count_list = [id_list.count(i) for i in self.region_list]
        demand_list = dict(zip(self.region_list, count_list))
        del df, point_list, id_list, count_list
        return demand_list


if __name__ == "__main__":
    args = parser.parse_args()
    args.tmp_person_data2 = '../../' + args.tmp_person_data2
    args.hg2_zones_shp = '../../' + args.hg2_zones_shp
    args.hg2_central_edge = '../../' + args.hg2_central_edge
    pp = PerfectPred(args)
    obj = pp.getPrediction(0, 60 * 15)
    # obj=pp.span_time(0,10*60)
    print(obj)



