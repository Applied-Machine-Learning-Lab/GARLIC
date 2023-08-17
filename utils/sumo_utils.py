# -*- coding: utf-8 -*-
# @Time : 2023/3/21 9:48
# @Author : Xiao Han
# @E-mail : hahahenha@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : sumo_utils.py
# @Software: PyCharm

import time
import copy
import traci
import pickle
import random
import json
import sumolib
import datetime
import subprocess
import numpy as np
import pandas as pd
import geopandas as gpd
from numpy import sort

from config import parser
from shapely.geometry import Point

from model.dispatch.greedyclosest import GreedyCloestEuclidean
from model.pred.perfectpredict import PerfectPred
from model.dispatch.dispatch_base import DispatchModel, Decision
from model.rebalance.rebalance_base import RebalanceModel
from utils.BDHconversion import dec_to_bin
from utils.Geohash import Geohash
from utils.Logger import LogPrinter
from utils.RW_Lock import RWLock

GREEN = (0,188,0,255)
RED = (255,0,0,255)
BROWN = (149,82,0,255)

BIGINT = 9999999999

class Controller(object):

    def __init__(self, args) -> None:
        self.args = args
        self.geohash = Geohash()
        time_s_str = args.start_time.split(':')
        self.start_t = int(int(time_s_str[0]) * 3600 + int(time_s_str[1]) * 60 + int(time_s_str[2]))
        self._rebalanceModel = RebalanceModel()
        self._dispatchModel = DispatchModel()   #TODO:change method
        self._predictor = PerfectPred(self.args) #TODO:change method
        self._rebalanceWindowRVCnt = 0
        self._rebalanceWindowRVCntObj = 0
        self._rebalancingPool = {}          # 正在执行rebalance任务的车辆记录
        self._servingPool = {}              # 正在执行serive任务的车辆记录
        self._openReservationPool = {}      # 等待匹配的订单
        self._regionDemandRecord = {}       # 记录DSAGGREGATEWINDOW内各个region产生的累计需求数
        self._regionSupplyRecord = {}       # 记录DSAGGREGATEWINDOW内各个region新产生的累计供应数
        self._centralEdges = {}             # region的中心位置对应的路（边）
        self._errorReservationRec = {}      # 记录订单的起点终点是否（双向）连通
        self._outOfRegionBoundTimes = 0     # 需求产生点超出运营区域的次数
        self._regionalCosts = None
        self._rebalanceLenRank = None
        self._rebalanceTasks = None
        self._net = sumolib.net.readNet(args.net_file)
        self._lastTimeVacantFleet = set()
        self._lastFleetSZ = 0
        self._logprinter = LogPrinter(self.args.log_dir)
        self._dataToPrint = {'regionDemandSupplyRealtimeInfo': None,
                             'regionDemandSupplyAggregateInfo': {'startTime': None, 'timeSpan': None, 'regionD': None,
                                                                 'regionS': None},
                             'rebalanceinfo': [], 'servicepredinfo': None, 'taxistatnuminfo': None,
                             'regionDemandSupplyPredInfo': {'startTime': None, 'timeSpan': None, 'regionD': None,
                                                            'regionS': None}}

        # 区域中心点到边的映射
        with open(args.hg2_central_edge, 'r') as centralEdgesFile:
            self._centralEdges: dict = json.load(centralEdgesFile)
        old_keys = set(self._centralEdges.keys())
        for old_key in old_keys:
            new_key = int(old_key)
            self._centralEdges[new_key] = str(self._centralEdges[old_key])
            del self._centralEdges[old_key]
        self._regionProjector = RegionProjector(self.args.hg2_zones_shp ,self._centralEdges.keys())

        # hg5区域中心点到边的映射
        with open(args.hg5_central_edge, 'r') as centralEdgesFile:
            self._centralEdgesM: dict = json.load(centralEdgesFile)
        old_keys = set(self._centralEdgesM.keys())
        for old_key in old_keys:
            new_key = int(old_key)
            self._centralEdgesM[new_key] = str(self._centralEdgesM[old_key])
            del self._centralEdgesM[old_key]
        self._regionProjectorM = RegionProjector(self.args.hg5_zones_shp, self._centralEdgesM.keys(), scale='M')

        # hg10区域中心点到边的映射
        with open(args.hg10_central_edge, 'r') as centralEdgesFile:
            self._centralEdgesH: dict = json.load(centralEdgesFile)
        old_keys = set(self._centralEdgesH.keys())
        for old_key in old_keys:
            new_key = int(old_key)
            self._centralEdgesH[new_key] = str(self._centralEdgesH[old_key])
            del self._centralEdgesH[old_key]
        self._regionProjectorH = RegionProjector(self.args.hg10_zones_shp, self._centralEdgesH.keys(), scale='H')

        # 初始化区域需求量记录
        self._regionDemandRecord = dict.fromkeys(self._regionProjector.regionIDList, 0)
        self._regionDemandRecordM = dict.fromkeys(self._regionProjectorM.regionIDList, 0)
        self._regionDemandRecordH = dict.fromkeys(self._regionProjectorH.regionIDList, 0)
        # 初始化区域供应量到达数记录
        self._regionSupplyRecord = dict.fromkeys(self._centralEdges.keys(), 0)
        self._regionSupplyRecordM = dict.fromkeys(self._centralEdgesM.keys(), 0)
        self._regionSupplyRecordH = dict.fromkeys(self._centralEdgesH.keys(), 0)

        temp_dict = dict.fromkeys(self._centralEdges.keys())
        vehNumEachRegion = int(self.args.taxi_size / len(self._centralEdges))
        sum = 0
        for idx, (key, value) in enumerate(temp_dict.items()):
            if idx == len(temp_dict) - 1:
                temp_dict[key] = self.args.taxi_size - sum
            else:
                sum += vehNumEachRegion
                temp_dict[key] = vehNumEachRegion

        self.routes = dict.fromkeys(self._regionProjector.regionIDList)
        self.routesM = dict.fromkeys(self._regionProjectorM.regionIDList)
        self.routesH = dict.fromkeys(self._regionProjectorH.regionIDList)

        # 添加车队起始路线，初始化所有车辆在区域中心道路上
        for regionID, edgeID in self._centralEdges.items():
            routeID = 'ROUTE' + str(regionID)
            traci.route.add(routeID=routeID, edges=[edgeID])
            self.routes[regionID] = routeID

        self.ALL_CARS = []
        vCnt = 0
        # 往每个区域中心点添加车辆
        for regionID in self._regionProjector.regionIDList:
            routeID = self.routes[regionID]
            regionVehNum = temp_dict[regionID]
            while regionVehNum > 0:
                vCnt += 1
                assert vCnt <= self.args.taxi_size
                taxiID = 'TAXI' + str(vCnt)
                self.ALL_CARS.append(taxiID)
                traci.vehicle.add(vehID=taxiID, routeID=routeID, typeID='taxi', depart='now')
                # 设置路由模式
                traci.vehicle.setRoutingMode(vehID=taxiID, routingMode=traci.constants.ROUTING_MODE_AGGREGATED)

                regionVehNum -= 1
        self.ALL_CARS = set(self.ALL_CARS)
        self.NET = sumolib.net.readNet(self.args.net_file, withInternal=False)

    def step(self):
        ### 决策间隔为STEPTIME
        now = traci.simulation.getTime()
        if now == self.args.sim_time:
            return -1
        if now % self.args.step != 0:
            return 1
        ### 供给信息收集
        print('-------------------' + '第' + str(now) + '步' + '-------------------')
        real_time = self.start_t + now
        print('real_time (sec):', real_time)
        # 0 (empty) : taxi is idle
        # 1 (pickup): taxi is en-route to pick up a customer
        # 2 (occupied): taxi has customer on board and is driving to drop-off
        # 3 (pickup + occupied): taxi has customer on board but will pick up more customers, carID also shows in mode 1 or 2
        # -1: (return all taxis regardless of state)
        allFleet = set(traci.vehicle.getTaxiFleet(-1))
        IDlist = set(traci.vehicle.getIDList())

        if allFleet != (allFleet & IDlist):  # TODO manage teleports
            print("\nVehicle %s is being teleported, skip to next step" %
                  (allFleet - set(traci.vehicle.getIDList())))
            return -2  # if a vehicle is being teleported skip to next step
        begin0 = time.time()

        begin = time.time()
        PoolingTaskInfo.stepBegin()
        end = time.time()
        print(f'stepBegin用时：%f s' % (end - begin))

        # 空车（包括vacant，rebalancing）集合
        emptyFleet = traci.vehicle.getTaxiFleet(0)

        # 移除掉意外退出路网的车辆记录
        nowFleetSZ = len(allFleet)

        if (nowFleetSZ < self._lastFleetSZ):
            self._servingPool = {key: values for key, values in self._servingPool.items()
                                 if key in allFleet}
            self._rebalancingPool = {key: values for key, values in self._rebalancingPool.items()
                                     if key in allFleet}

        self._lastFleetSZ = nowFleetSZ

        # 将_servingPool中刚完成订单任务的车辆对应记录删除
        oldServingPool = self._servingPool
        self._servingPool = {key: values for key, values in self._servingPool.items()
                             if key not in emptyFleet}

        for key in (oldServingPool.keys() - self._servingPool.keys()):
            predArrivalTime = oldServingPool[key].estArrivalTime
            actArrivalTime = now
            self._logprinter.updateServicePredRecord(key, predArrivalTime=predArrivalTime,
                                                     actArrivalTime=actArrivalTime)

        # 将_rebalancingPool中完成再平衡任务的车辆对应记录删除
        oldRebalancingPool = self._rebalancingPool
        for key in oldRebalancingPool.keys():
            vX0, vY0 = traci.vehicle.getPosition(vehID=key)
            vX0, vY0 = traci.simulation.convertGeo(vX0, vY0, fromGeo=False)
            # 找出车辆所在位置对应区域
            regionID = self._regionProjector.inWhichRegion(x=vX0, y=vY0, isCache=True)
            # print('test:', regionID)
            # print('test desArea:', oldRebalancingPool[key].desArea)
            if str(regionID) == str(oldRebalancingPool[key].desArea):
                self._rebalancingPool.pop(key)

        finishedRebalanceVehs = set(oldRebalancingPool.keys()) - set(self._rebalancingPool.keys())
        for vID in finishedRebalanceVehs:
            # arrivalTime有至多StepTime的误差
            self._dataToPrint['rebalanceinfo'].append({'vehID': vID, 'isDepart': False, 'arrivalTime': now})

        # 删除emptyFleet中同时在_rebalancingPool里出现过的车辆ID删除
        emptyFleet = [vID for vID in emptyFleet if vID not in self._rebalancingPool.keys()]
        emptyFleet = set(emptyFleet)

        for vID in emptyFleet - self._lastTimeVacantFleet:
            # 空闲车辆用绿色highlight
            traci.vehicle.highlight(vehID=vID, color=GREEN, size=self.args.high_light, type=1)

        # 记录各状态出租车数量
        self._dataToPrint['taxistatnuminfo'] = {'time': now, 'vacantNum': len(emptyFleet),
                                                'servingNum': len(self._servingPool),
                                                'rebalancingNum': len(self._rebalancingPool)}

        print('vacant车辆数：')
        print(len(emptyFleet))
        print('serving车辆数：')
        print(len(self._servingPool))
        print('rebalancing车辆数：')
        print(len(self._rebalancingPool))
        print('总车辆数：')
        print(nowFleetSZ)

        ### 需求信息收集
        newReservations = traci.person.getTaxiReservations(1)
        # 将新到达的订单放入_openReservationPool
        for res in newReservations:
            assert res not in self._openReservationPool
            self._openReservationPool[res.id] = res

        print('开放订单数：')
        print(len(self._openReservationPool))

        assert nowFleetSZ == len(emptyFleet) + len(self._servingPool) + len(self._rebalancingPool)

        # 记录预测区间内各个区域的累计需求到达数
        for res in newReservations:
            pickupEdge = res.fromEdge
            pickupPos = res.departPos
            # if res.persons[0] == "43":
            #     print(res)
            x, y = traci.simulation.convert2D(edgeID=pickupEdge, pos=pickupPos, toGeo=True)
            inRegion = self._regionProjector.inWhichRegion(x, y, isCache=True)
            if inRegion is not None:
                self._regionDemandRecord[inRegion] += 1
            else:
                self._outOfRegionBoundTimes += 1

        # 记录本step各区域的实时未满足需求量
        nowRegionDemand = dict.fromkeys(self._centralEdges.keys(), 0)
        nowRegionDemandM = dict.fromkeys(self._centralEdgesM.keys(), 0)
        nowRegionDemandH = dict.fromkeys(self._centralEdgesH.keys(), 0)
        for resID, resObj in self._openReservationPool.items():
            pickupEdge = resObj.fromEdge
            pickupPos = resObj.departPos
            x, y = traci.simulation.convert2D(edgeID=pickupEdge, pos=pickupPos, toGeo=True)
            inRegion = self._regionProjector.inWhichRegion(x, y, isCache=True)
            inRegionM = self._regionProjectorM.inWhichRegion(x, y, isCache=True)
            inRegionH = self._regionProjectorH.inWhichRegion(x,y, isCache=True)
            if inRegion is not None:
                nowRegionDemand[inRegion] += 1
            else:
                pass
            if inRegionM is not None:
                nowRegionDemandM[inRegionM] += 1
            else:
                pass
            if inRegionH is not None:
                nowRegionDemandH[inRegionH] += 1
            else:
                pass
        G2_Dkeys = sorted(nowRegionDemand)
        G2_D = []
        for key in G2_Dkeys:
            G2_D.append(nowRegionDemand[key])
        G5_Dkeys = sorted(nowRegionDemandM)
        G5_D = []
        for key in G5_Dkeys:
            G5_D.append(nowRegionDemandM[key])
        G10_Dkeys = sorted(nowRegionDemandH)
        G10_D = []
        for key in G10_Dkeys:
            G10_D.append(nowRegionDemandH[key])
        # print('Demand G2:', G2_D)
        # print('Demand G5:', G5_D)
        # print('Demand G10:', G10_D)

        self._dataToPrint['regionDemandSupplyRealtimeInfo'] = {'time': now, 'regionD': nowRegionDemand}

        ### 供给信息收集
        # 记录本step各区域的可用空车数
        nowRegionSupply = dict.fromkeys(self._centralEdges.keys(), 0)
        nowRegionSupplyM = dict.fromkeys(self._centralEdgesM.keys(), 0)
        nowRegionSupplyH = dict.fromkeys(self._centralEdgesH.keys(), 0)

        newVancantFleet = emptyFleet - self._lastTimeVacantFleet
        nowSDic = dict.fromkeys(self._centralEdges.keys())
        for key in nowSDic.keys():
            nowSDic[key] = set()
        for vID in emptyFleet:
            # 空闲车辆当前地理坐标(x, y)
            vX0, vY0 = traci.vehicle.getPosition(vehID=vID)
            vX0, vY0 = traci.simulation.convertGeo(vX0, vY0, fromGeo=False)  # return lon,lat    if True return x,y
            _, geo_emb = self.geohash.encode(lon=vX0, lat=vY0, precision=12)

            # 找出车辆所在位置对应区域
            # print('Controller.step:', vX0, vY0)
            regionID = self._regionProjector.inWhichRegion(x=vX0, y=vY0, isCache=True)
            regionIDM = self._regionProjectorM.inWhichRegion(x=vX0, y=vY0, isCache=True)
            regionIDH = self._regionProjectorH.inWhichRegion(x=vX0, y=vY0, isCache=True)

            # 对于每一个vID,所在2,5,10区域信息以及经纬度信息
            # emb_one_step = np.concatenate((hg2_emb, hg5_emb, hg10_emb, geo_emb), axis=1)
            hg2_emb_str = dec_to_bin(str(regionID-2000)).zfill(self._regionProjector.get_embwidth)
            hg2_emb = []
            for i in range(len(hg2_emb_str)):
                hg2_emb.append(int(hg2_emb_str[i]))
            hg2_emb = np.array(hg2_emb)[np.newaxis, :]
            hg5_emb_str = dec_to_bin(str(regionIDM-5000)).zfill(self._regionProjectorM.get_embwidth)
            hg5_emb = []
            for i in range(len(hg5_emb_str)):
                hg5_emb.append(int(hg5_emb_str[i]))
            hg5_emb = np.array(hg5_emb)[np.newaxis, :]
            hg10_emb_str = dec_to_bin(str(regionIDH-10000)).zfill(self._regionProjectorH.get_embwidth)
            hg10_emb = []
            for i in range(len(hg10_emb_str)):
                hg10_emb.append(int(hg10_emb_str[i]))
            hg10_emb = np.array(hg10_emb)[np.newaxis, :]
            emb_one_step = np.concatenate((hg2_emb, hg5_emb, hg10_emb, np.array(geo_emb)[np.newaxis, :]), axis=1)
            print('emb_one_step shape:', emb_one_step.shape)

            if regionID is not None:
                nowRegionSupply[regionID] += 1
                nowSDic[regionID].add(vID)
                if vID in newVancantFleet:
                    self._regionSupplyRecord[regionID] += 1
            else:
                pass
            if regionIDM is not None:
                nowRegionSupplyM[regionIDM] += 1
            else:
                pass
            if regionIDH is not None:
                nowRegionSupplyH[regionIDH] += 1
            else:
                pass
        G2_Skeys = sorted(nowRegionSupply)
        G2_S = []
        for key in G2_Skeys:
            G2_S.append(nowRegionSupply[key])
        G5_Skeys = sorted(nowRegionSupplyM)
        G5_S = []
        for key in G5_Skeys:
            G5_S.append(nowRegionSupplyM[key])
        G10_Skeys = sorted(nowRegionSupplyH)
        G10_S = []
        for key in G10_Skeys:
            G10_S.append(nowRegionSupplyH[key])
        # print('Supply G2:', G2_S)
        # print('Supply G5:', G5_S)
        # print('Supply G10:', G10_S)

        G2_DS = np.array(list(zip(G2_D, G2_S)))
        G5_DS = np.array(list(zip(G5_D, G5_S)))
        G10_DS = np.array(list(zip(G10_D, G10_S)))

        print('G2_DS shape:', G2_DS.shape)
        print('G5_DS shape:', G5_DS.shape)
        print('G10_DS shape:', G10_DS.shape)

        self._dataToPrint['regionDemandSupplyRealtimeInfo']['regionS'] = nowRegionSupply

        # 记录各区域供需信息
        if now % self.args.DS_window == 0 and now > 0:
            self._dataToPrint['regionDemandSupplyAggregateInfo']['startTime'] = now - self.args.DS_window
            self._dataToPrint['regionDemandSupplyAggregateInfo']['timeSpan'] = self.args.DS_window
            self._dataToPrint['regionDemandSupplyAggregateInfo']['regionD'] = self._regionDemandRecord
            self._dataToPrint['regionDemandSupplyAggregateInfo']['regionS'] = self._regionSupplyRecord
            # 需求记录清零
            self._regionDemandRecord = dict.fromkeys(self._centralEdges.keys(), 0)
            # 以每个区域当前空车数初始化供给记录
            self._regionSupplyRecord = dict.fromkeys(self._centralEdges.keys(), 0)
            for regionID in self._regionSupplyRecord.keys():
                self._regionSupplyRecord[regionID] = len(nowSDic[regionID])

        ### 再平衡决策
        decision, vIDFRegion = self._rebalance(now, vacantFleet=emptyFleet, nowVehDic=nowSDic)

        if self.args.balance:
            setOutVehCnt = self._excecuteRebalance(now, emptyFleet=emptyFleet, decision=decision,
                                                   vehIDFromRegion=vIDFRegion)
            self._rebalanceWindowRVCnt += setOutVehCnt
            print('本平衡调度时间窗内累计发出平衡车辆数：')
            print('\t已发：%d' % (self._rebalanceWindowRVCnt))
            print('\t目标:%d' % (self._rebalanceWindowRVCntObj))

        ### 决策（匹配）

        if self.args.model == 'GreedyClosest':
            decision = GreedyCloestEuclidean(vacantFleet=emptyFleet, openReservationPool=self._openReservationPool, traci=traci)
            print(f'决策+派单任务下达时间：%f s' % (end - begin))
        else:
            exit('dispatch decision')
        # print('Controller.decision:', decision.service)

        ### 得到决策结果后
        self._excecuteDispatch(now, emptyFleet=emptyFleet, decision=decision, isShared=self.args.isshared)

        # 更新nowSDic
        for key, value in nowSDic.items():
            nowSDic[key] = value & emptyFleet

        self._lastTimeVacantFleet = emptyFleet

        self._printLog(now)

        end0 = time.time()
        print(f'本step运行总时间：%f s' % (end0 - begin0))

        PoolingTaskInfo.stepFinish()
        return 0

    def _predictDemand(self, now, timeSpan, demand) -> dict:
        pred = self._predictor.getPrediction(now, timeSpan, demand)
        # result = dict.fromkeys(self._centralEdges.keys(), 0)
        # for key in result.keys():
        #     if key in pred.keys():
        #         result[key] = pred[key]
        return pred

    def _excecuteRebalance(self, now, emptyFleet, decision, vehIDFromRegion):
        # 再平衡
        successTimes = 0
        for vID, regionID in decision.rebalance.items():
            actualEdge = self._centralEdges[regionID]
            # stops = traci.vehicle.getStops(vID)
            # assert len(stops) == 1
            # originalStopData = stops[0]
            # vehNowEdge = traci.lane.getEdgeID(originalStopData.lane)

            vehNowEdge = traci.vehicle.getRoadID(vID)

            try:
                if vehNowEdge == actualEdge:
                    continue
                traci.vehicle.changeTarget(vID, actualEdge)
                # obj1 = traci.vehicle.getStops(vID)
                # if len(obj1) > 0:
                #     assert len(obj1) == 1
                #     traci.vehicle.replaceStop(vID, 0, '')
                # traci.vehicle.setStop(vehID=vID, flags=STOPFLAG, edgeID=actualEdge)
            except Exception as ex:
                print(ex)
                exit('再平衡')

            # 计算并记录预计到达时间
            # mode = traci.vehicle.getRoutingMode(vehID=vID)
            edges = traci.vehicle.getRoute(vID)
            estEnrouteTime = 0
            for edge in edges:
                traveltime = float((traci.vehicle.getParameter(vID, f"device.rerouting.edge:%s" % (edge))))
                assert traveltime != -1
                estEnrouteTime += traveltime

            routeLength = traci.vehicle.getDrivingDistance(vehID=vID, edgeID=actualEdge, pos=1.0)

            self._rebalancingPool[vID] = EnrouteInfo(desArea=regionID, estArrivalTime=(now + estEnrouteTime))

            self._dataToPrint['rebalanceinfo'].append({'isDepart': True, 'departTime': now, 'vehID': vID,
                                                       'fromRegion': vehIDFromRegion[vID], 'toRegion': regionID,
                                                       'fromEdge': vehNowEdge, 'toEdge': actualEdge,
                                                       'routeLength': routeLength, 'predDuration': estEnrouteTime})
            # 平衡车辆用棕色highlight
            traci.vehicle.highlight(vehID=vID, color=BROWN, size=self.args.high_light)
            emptyFleet.remove(vID)
            successTimes += 1
        return successTimes

    def _excecuteDispatch(self, now, emptyFleet, decision, isShared=False):
        # 订单匹配
        for vID, value in decision.service.items():
            if not isShared:
                resID = value
                resStage = self._openReservationPool[resID]
                isSuccess = PoolingTaskInfo.changeTaskQ(resObj=resStage, vID=vID)

                if not isSuccess:
                    continue

                ### 派单成功
                # 订单目的地的地理坐标(x, y)
                x, y = traci.simulation.convert2D(edgeID=resStage.toEdge, pos=resStage.arrivalPos, toGeo=True)
                # 目的地区域ID
                regionID = self._regionProjector.inWhichRegion(x=x, y=y)

                self._servingPool[vID] = EnrouteInfo(desArea=regionID, estArrivalTime=-1)

                # 从open订单池中删除相应订单记录
                self._openReservationPool.pop(resID)
                # 服务车辆用红色highlight
                traci.vehicle.highlight(vehID=vID, color=RED, size=self.args.high_light)
                emptyFleet.remove(vID)
                # print('\t_excecuteDispatch:', vID, 'is removed')
            else:
                # 拼车
                resID = value['resID']
                pickPos, dropPos = value['insertPositions']
                resStage = self._openReservationPool[resID]
                isSuccess = PoolingTaskInfo.changeTaskQ(resObj=resStage, vID=vID,
                                                        pickupInsertPos=pickPos, dropoffInsertPos=dropPos)
                if not isSuccess:
                    continue

                ### 派单成功
                # desResID = PoolingTaskInfo.poolingRecords[vID].taskQ[-1]    # 终点订单id
                desResID = PoolingTaskInfo.getPoolingRecordValue(vID).taskQ[-1]  # 终点订单id

                desResObj = PoolingTaskInfo.getClosedResRecord(desResID)

                # 订单目的地的地理坐标(x, y)
                x, y = traci.simulation.convert2D(edgeID=desResObj.toEdge, pos=desResObj.arrivalPos, toGeo=True)
                # 目的地区域ID
                regionID = self._regionProjector.inWhichRegion(x=x, y=y)
                self._servingPool[vID] = EnrouteInfo(desArea=regionID, estArrivalTime=-1)

                # 从open订单池中删除相应订单记录
                self._openReservationPool.pop(resID)

                if vID in emptyFleet:
                    # 服务车辆用红色highlight
                    traci.vehicle.highlight(vehID=vID, color=RED, size=self.args.high_light)
                    emptyFleet.remove(vID)

    def _rebalance(self, now, vacantFleet, nowVehDic):
        # 决策阶段
        if now % self.args.rebalance_time == 0:
            self._rebalanceWindowRVCnt = 0
            ### 计算Cij
            pre_value = self._predictDemand(now, self.args.pred_time, nowVehDic)
            self._regionalCosts = {}
            # 初始化_rebalanceLenRank
            self._rebalanceLenRank = {}
            regionIDLst = list(self._centralEdges.keys())

            start = time.time()
            print("区域间路径计算：")
            print("...")

            for regionID1 in regionIDLst:
                self._rebalanceLenRank[regionID1] = []
                for regionID2 in regionIDLst:
                    if regionID1 != regionID2:
                        centerEdge1 = self._centralEdges[regionID1]
                        centerEdge2 = self._centralEdges[regionID2]
                        stgRou = traci.simulation.findRoute(fromEdge=centerEdge1, toEdge=centerEdge2,
                                                            vType='taxi', depart=-1,
                                                            routingMode=traci.constants.ROUTING_MODE_AGGREGATED, )
                        self._regionalCosts[(regionID1, regionID2)] = stgRou.travelTime

                        # 以每个区域到其他区域的距离大小进行排序（由大到小）
                        foundIndex = len(self._rebalanceLenRank[regionID1])
                        for idx, regionID3 in enumerate(self._rebalanceLenRank[regionID1]):
                            if self._regionalCosts[(regionID1, regionID3)] < stgRou.length:
                                foundIndex = idx
                                break
                        self._rebalanceLenRank[regionID1].insert(foundIndex, regionID2)
            end = time.time()
            print(f"计算完毕，用时：%d s" % (end - start))

            ### si
            latestBound = now + self.args.pred_time
            # 未来供应量
            futureSupply = self._predictFutureSupply()
            # 未来+现在供应量
            sDict = futureSupply.copy()

            for regionID in sDict.keys():
                sDict[regionID] += len(nowVehDic[regionID])

            ### λi
            lambdaDict = pre_value

            # log文件数据
            self._dataToPrint['regionDemandSupplyPredInfo']['startTime'] = now
            self._dataToPrint['regionDemandSupplyPredInfo']['timeSpan'] = self.args.pred_time
            self._dataToPrint['regionDemandSupplyPredInfo']['regionD'] = pre_value
            self._dataToPrint['regionDemandSupplyPredInfo']['regionS'] = sDict

            result = self._rebalanceModel.decide(now, regionalCosts=self._regionalCosts, regionSupplyRates=sDict,
                                                 regionResArrivalRates=lambdaDict)
            # self._rebalanceTasks = result
            print('sumo utils, rebalance result:', result)
            self._rebalanceTasks = dict.fromkeys(self._centralEdges.keys())
            # 生成平衡任务
            # debug
            rcnt = 0
            for fromRegionID in self._rebalanceTasks.keys():
                self._rebalanceTasks[fromRegionID] = {'target': [0, 0.0], 'real': [0, 0], 'taskLst': [], 'stepLen': -1}
                totRebalanceVehCnt = 0
                for toRegionID in self._rebalanceLenRank[fromRegionID]:
                    # 计划在平衡调度区间内要从fromRegionID调FromTo辆车到toRegionID
                    if toRegionID == fromRegionID:
                        continue
                    rFromTo = int(result[(fromRegionID, toRegionID)])
                    if rFromTo > 0:
                        # 如果>0，就安排这个任务（根据路程距离由远到近的排序）
                        rcnt += rFromTo
                        totRebalanceVehCnt += rFromTo
                        self._rebalanceTasks[fromRegionID]['taskLst'].append((toRegionID, rFromTo))

                # 每step要完成的任务量
                self._rebalanceTasks[fromRegionID]['stepLen'] = (totRebalanceVehCnt / self.args.rebalance_task_time) * self.args.step

            self._rebalanceTasks = dict(item for item in self._rebalanceTasks.items() if len(item[1]['taskLst']) > 0)
            self._rebalanceWindowRVCntObj = rcnt
            return Decision({}, {}), {}
        else:

            rebalanceDecision = {}
            vIDFRegion = {}
            for fRegionID in self._rebalanceTasks.keys():
                # 移动target指针
                stepLen = self._rebalanceTasks[fRegionID]['stepLen']
                targetPos = self._rebalanceTasks[fRegionID]['target'][0]
                targetCompleted = self._rebalanceTasks[fRegionID]['target'][1]
                realPos = self._rebalanceTasks[fRegionID]['real'][0]
                realCompleted = self._rebalanceTasks[fRegionID]['real'][1]
                taskLst = self._rebalanceTasks[fRegionID]['taskLst']

                if targetPos != len(taskLst):
                    accum = 0
                    accum += taskLst[targetPos][1] - targetCompleted
                    while accum < stepLen:
                        targetPos += 1
                        targetCompleted = 0
                        if targetPos == len(taskLst):
                            break
                        accum += taskLst[targetPos][1] - targetCompleted
                    else:
                        if targetPos == len(taskLst):
                            break
                        targetCompleted = taskLst[targetPos][1] - (accum - stepLen)
                        if targetCompleted == taskLst[targetPos][1]:
                            targetPos += 1
                            targetCompleted = 0
                    # 更新
                    self._rebalanceTasks[fRegionID]['target'][0] = targetPos
                    self._rebalanceTasks[fRegionID]['target'][1] = targetCompleted

                if realPos != len(taskLst):
                    # accum = 0
                    # accum += taskLst[targetPos][1] - targetCompleted
                    while len(nowVehDic[fRegionID]) > 0 and (
                            realPos != targetPos or (realPos == targetPos and realCompleted < int(targetCompleted))):
                        toRegionID = taskLst[realPos][0]
                        if realPos != targetPos:
                            num1 = taskLst[realPos][1] - realCompleted
                            if len(nowVehDic[fRegionID]) >= num1:
                                for _ in range(num1):
                                    vID = nowVehDic[fRegionID].pop()
                                    rebalanceDecision[vID] = toRegionID
                                    vIDFRegion[vID] = fRegionID
                                ##到达最大临界值进一，记得!!!
                                realPos += 1
                                realCompleted = 0
                            else:
                                num2 = len(nowVehDic[fRegionID])
                                for _ in range(num2):
                                    vID = nowVehDic[fRegionID].pop()
                                    rebalanceDecision[vID] = toRegionID
                                    vIDFRegion[vID] = fRegionID
                                realCompleted += num2
                        else:
                            # realPos == targetPos and realCompleted < int(targetCompleted)的情况
                            gap = int(targetCompleted) - realCompleted
                            sendout = min(gap, len(nowVehDic[fRegionID]))

                            for _ in range(sendout):
                                vID = nowVehDic[fRegionID].pop()
                                rebalanceDecision[vID] = toRegionID
                                vIDFRegion[vID] = fRegionID
                            realCompleted += sendout

                    # 更新指针记录
                    self._rebalanceTasks[fRegionID]['real'][0] = realPos
                    self._rebalanceTasks[fRegionID]['real'][1] = realCompleted

            return Decision({}, rebalanceDecision), vIDFRegion

    def _predictFutureSupply(self):
        with open(self.args.tmp_curr_routes, 'w+') as rouFile:
            sumolib.xml.writeHeader(outf=rouFile, root='routes')
            # rouFile.write("<routes>\n")
            for vID, enrouteInfo in (self._servingPool | self._rebalancingPool).items():
                typeID = 'taxi'
                route = traci.vehicle.getRoute(vehID=vID)

                # routeID = 'initRoute_' + vID
                fEdgeIndex = traci.vehicle.getRouteIndex(vehID=vID)
                if fEdgeIndex == -1:
                    fEdgeIndex = 0
                routeStr = ''

                for idx, edge in enumerate(route):
                    if idx < fEdgeIndex:
                        continue
                    routeStr += f'%s' % (edge)
                    if idx != len(route) - 1:
                        routeStr += ' '

                # stops = traci.vehicle.getStops(vehID=vID)
                # assert len(stops) > 0
                # lastStop = stops[-1]
                #
                # # debug
                # if len(stops) == 1:
                #     pass

                rouFile.write("""\t<vehicle id="%s" depart="%s">\n """
                              % (vID, str(0)))
                rouFile.write("""\t\t<route edges="%s">\n""" % (routeStr))

                # rouFile.write("""\t\t\t<stop lane="%s" endPos="%s" actType="%s" parking="true"/>\n"""%(str(lastStop.lane), str(lastStop.endPos), str(lastStop.actType)))

                rouFile.write("""\t\t</route>\n""")

                rouFile.write("""\t</vehicle>\n""")
            rouFile.write("</routes>\n")
            rouFile.close()

        sumoBinary = sumolib.checkBinary('sumo')

        subprocess.call([sumoBinary, '-n', self.args.net_file,
                         '--additional-files', self.args.add_file,
                         '--route-files', self.args.tmp_curr_routes,
                         # '--device.taxi.idle-algorithm', 'randomCircling',
                         '--tripinfo-output.write-unfinished',
                         '--vehroute-output', self.args.tmp_res_routes,
                         '--vehroute-output.write-unfinished',
                         '--tripinfo-output', self.args.trip_info,
                         '--stop-output', self.args.stop_info,
                         '--vehroute-output.cost',
                         '--vehroute-output.exit-times',
                         '--gui-settings-file', self.args.sumo_gui_setting,
                         # '--vehroute-output.dua',
                         '--end', str(self.args.pred_time)])

        futureSupply = dict.fromkeys(self._centralEdges.keys(), 0)
        for vID, arrival in sumolib.xml.parse_fast(self.args.tmp_res_routes, 'vehicle', ['id', 'arrival']):

            if vID in self._servingPool.keys():
                desArea = self._servingPool[vID].desArea
            elif vID in self._rebalancingPool.keys():
                desArea = self._rebalancingPool[vID].desArea
            else:
                raise Exception('_predictFutureSupply')
            if desArea is not None:
                futureSupply[desArea] += 1

        return futureSupply

    def _printLog(self, now):
        if self._dataToPrint['taxistatnuminfo'] is not None:
            info = self._dataToPrint['taxistatnuminfo']
            self._logprinter.updateTaxiStatNumRecord(time=info['time'], vacantNum=info['vacantNum'],
                                                     servingNum=info['servingNum'],
                                                     rebalancingNum=info['rebalancingNum'])
            self._dataToPrint['taxistatnuminfo'] = None

        if self._dataToPrint['regionDemandSupplyRealtimeInfo'] is not None:
            time = self._dataToPrint['regionDemandSupplyRealtimeInfo']['time']
            regionD = self._dataToPrint['regionDemandSupplyRealtimeInfo']['regionD']
            regionS = self._dataToPrint['regionDemandSupplyRealtimeInfo']['regionS']
            self._logprinter.updateRealTimeEachRegionDemandSupply(time=time, regionD=regionD, regionS=regionS)
            self._dataToPrint['regionDemandSupplyRealtimeInfo'] = None

        if now % self.args.DS_window == 0 and now > 0:
            startTime = self._dataToPrint['regionDemandSupplyAggregateInfo']['startTime']
            timeSpan = self._dataToPrint['regionDemandSupplyAggregateInfo']['timeSpan']
            regionD = self._dataToPrint['regionDemandSupplyAggregateInfo']['regionD']
            regionS = self._dataToPrint['regionDemandSupplyAggregateInfo']['regionS']
            self._logprinter.updateAggregateEachRegionDemandSupply(startTime=startTime, timespan=timeSpan,
                                                                   regionD=regionD, regionS=regionS)

        if now % self.args.pred_time == 0:
            startTime = self._dataToPrint['regionDemandSupplyPredInfo']['startTime']
            timeSpan = self._dataToPrint['regionDemandSupplyPredInfo']['timeSpan']
            regionD = self._dataToPrint['regionDemandSupplyPredInfo']['regionD']
            regionS = self._dataToPrint['regionDemandSupplyPredInfo']['regionS']
            self._logprinter.updatePredEachRegionDemandSupply(startTime=startTime, timespan=timeSpan, regionD=regionD,
                                                              regionS=regionS)

        if len(self._dataToPrint['rebalanceinfo']) > 0:
            for infoDic in self._dataToPrint['rebalanceinfo']:
                isDepart = infoDic['isDepart']
                if isDepart:
                    vehID = infoDic['vehID']
                    departTime = infoDic['departTime']
                    fromRegion = infoDic['fromRegion']
                    toRegion = infoDic['toRegion']
                    fromEdge = infoDic['fromEdge']
                    toEdge = infoDic['toEdge']
                    predDuration = infoDic['predDuration']
                    routeLength = infoDic['routeLength']
                    self._logprinter.updateRebalanceRecord(vehID=vehID, isDepart=True, departTime=departTime,
                                                           fromRegion=fromRegion,
                                                           toRegion=toRegion, fromEdge=fromEdge, toEdge=toEdge,
                                                           routeLength=routeLength, predDuration=predDuration)
                else:
                    vehID = infoDic['vehID']
                    arrivalTime = infoDic['arrivalTime']
                    self._logprinter.updateRebalanceRecord(vehID=vehID, isDepart=isDepart, arrivalTime=arrivalTime)

            self._dataToPrint['rebalanceinfo'] = []

        self._logprinter.printout()

    def finish(self):
        print('仿真完成。')
        self._logprinter.finish()

class RegionProjector(object):

    def __init__(self, shp_file, regions, scale='L') -> None:
        # 读shp文件
        self._gdf = gpd.read_file(shp_file)
        self._regions = set(regions)
        self.region_num = len(regions)
        self.emb_width = len(dec_to_bin(str(self.region_num)))
        # print('RegionProjector', self._regions)
        self._gdf = self._gdf.to_crs('EPSG:4326')
        # 结果缓存
        self._resultCache = {}
        if scale == 'L':
            self.scale = 2000
        elif scale == 'M':
            self.scale = 5000
        elif scale == 'H':
            self.scale = 10000

    # 返回坐标(x,y)是否在剪枝后region中，返回region number或None
    def inWhichRegion(self, x, y, isCache=False):
        if isCache and (x, y) in self._resultCache:
            return self._resultCache[(x, y)]
        series = self._gdf.contains(Point(x, y))
        inRegions = series[series == True]
        if len(inRegions) > 0:
            result = self.scale + inRegions.index[0] + 1  # region的编号比index大1 + 2000 (hg2)
            if isCache:
                self._resultCache[(x, y)] = result
            return result
        else:
            return None

    @property
    def get_embwidth(self):
        return self.emb_width

    @property
    def regionCentroids(self):
        return self._gdf.centroid

    @property
    def regionIDList(self):  # 运营region集合
        return self._regions

class EnrouteInfo(object):

    def __init__(self, desArea: str, estArrivalTime: float) -> None:
        self._desArea = desArea  # 最终到达区域（id）
        self._estArrivalTime = estArrivalTime  # 估计到达时间（s)

    @property
    def desArea(self):
        return self._desArea

    @property
    def estArrivalTime(self):
        return self._estArrivalTime

class PoolingTaskInfo(EnrouteInfo):
    # 只包括serving车辆
    _poolingRecords = {}

    # 关闭的订单
    _closedResRecords = {}

    # 乘客到所在订单的映射
    _persons2Res = {}

    _pairDict = {}

    _pairMissedEdges = set()

    _lockTaskCache = RWLock()

    _errorVehs = set()

    _args = {}

    @staticmethod
    def getPoolingRecordsLen():
        return len(PoolingTaskInfo._poolingRecords)

    @staticmethod
    def getPoolingRecordKeys():
        return PoolingTaskInfo._poolingRecords.keys()

    @staticmethod
    def getPoolingRecordItems():
        lst = list(PoolingTaskInfo._poolingRecords.items())
        return copy.deepcopy(lst)

    @staticmethod
    def getPoolingRecordValue(key):
        return copy.deepcopy(PoolingTaskInfo._poolingRecords[key])

    @staticmethod
    def getClosedResRecord(key):
        return copy.deepcopy(PoolingTaskInfo._closedResRecords[key])

    @staticmethod
    def getPoolingRecords():
        return copy.deepcopy(PoolingTaskInfo._poolingRecords)

    @staticmethod
    def closedResRecords() -> dict:
        return PoolingTaskInfo._closedResRecords

    @staticmethod
    def getEdgePairTimeCost(fedge: str, tedge: str):
        if fedge == tedge:
            return 0.0
        else:
            key = fedge + '_' + tedge
            if key in PoolingTaskInfo._pairDict:
                return PoolingTaskInfo._pairDict[fedge + '_' + tedge]
            else:
                # raise Exception('getEdgePairTimeCost')
                stg = traci.simulation.findRoute(fromEdge=fedge, toEdge=tedge, vType='taxi')
                if len(stg.edges) == 0:
                    PoolingTaskInfo._pairDict[key] = float('inf')
                    return float('inf')
                else:
                    PoolingTaskInfo._pairDict[key] = stg.travelTime
                    return stg.travelTime

    @staticmethod
    def init(args):
        PoolingTaskInfo._args = args
        PoolingTaskInfo._lockTaskCache.w_acquire()
        if PoolingTaskInfo._args.model == 'InsertionHeuristic':
            PoolingTaskInfo._readEdgePairs()
            with open(PoolingTaskInfo._args.pair_err_log, 'w') as outf:
                outf.close()
            with open(PoolingTaskInfo._args.err_log, 'w') as outf:
                outf.close()

    @staticmethod
    def stepBegin():
        # 每一步都要调用（通过len(poolingRecords)检查是否和主模块的记录一样
        servingFleet = set(traci.vehicle.getTaxiFleet(-1)) - set(traci.vehicle.getTaxiFleet(0))
        allFleet = set(traci.vehicle.getTaxiFleet(-1))

        # 清除掉跑出路网的车
        for taxiId in set(PoolingTaskInfo._poolingRecords.keys()):
            if taxiId not in allFleet:
                PoolingTaskInfo._poolingRecords.pop(taxiId)

        for taxiId in set(PoolingTaskInfo._poolingRecords.keys()):  # 不转成set循环会有bug
            # 得到这辆出租车当前任务队列中的所有顾客id
            currentCustomersSet = set()
            x = traci.vehicle.getParameter(taxiId, 'device.taxi.currentCustomers')
            x = x.split()
            for personId in x:
                currentCustomersSet.add(str(personId))

            # 根据currentCustomersSet，对_myTaskQueueCache进行更新
            taskQueue = PoolingTaskInfo._poolingRecords[taxiId].taskQ
            # last step的订单乘客集合
            oldCustomersSet = set()
            for resId in set(taskQueue):
                oldCustomersSet |= set(PoolingTaskInfo._closedResRecords[resId].persons)
            # 找出已完成的订单
            # if oldCustomersSet != currentCustomersSet:
                # print('stepBegin,taxiID:', taxiId)
                # print('\toldCustomersSet:', oldCustomersSet)
                # print('\tcurrentCustomersSet:', currentCustomersSet)
            assert oldCustomersSet >= currentCustomersSet
            finishedCustomersSet = oldCustomersSet - currentCustomersSet
            if len(finishedCustomersSet) > 0:
                print('finishedCustomersSet')
                finishedResIdSet = set()  # 已完成的订单
                for resId in set(taskQueue):
                    personsTup = PoolingTaskInfo._closedResRecords[resId].persons
                    resPersons = set(personsTup)  # debug: 检查resPersons的类型
                    # numFinished = 0
                    if resPersons <= finishedCustomersSet:
                        # 删除已完成的订单记录
                        PoolingTaskInfo._closedResRecords.pop(resId)
                        deletedId = PoolingTaskInfo._persons2Res.pop(personsTup)
                        assert deletedId == resId
                        finishedResIdSet.add(resId)

                # 剔除原本任务队列中已完成的订单
                if len(finishedResIdSet) > 0:
                    taskQueue = [resId for resId in taskQueue if resId not in finishedResIdSet]
                    PoolingTaskInfo._poolingRecords[taxiId]._taskQ = taskQueue

                assert len(taskQueue) % 2 == 0

                if len(taskQueue) == 0:
                    # 已完成所有任务
                    PoolingTaskInfo._poolingRecords.pop(taxiId)

        # assert len(servingFleet) == len(PoolingTaskInfo._poolingRecords)

        # 使得所有PoolingTaskInfo对象的_currPos失效
        for vID, info in PoolingTaskInfo._poolingRecords.items():
            info._currPos = -1

        # 此时TaskRecord的currPos仍需要更新
        for taxiId in PoolingTaskInfo._poolingRecords.keys():
            onBoardCustomers = set(traci.vehicle.getPersonIDList(taxiId))
            taskQueue = PoolingTaskInfo._poolingRecords[taxiId].taskQ
            # 找到车上乘客所属的订单
            occurResSet = set()
            for persons, resId in PoolingTaskInfo._persons2Res.items():
                if onBoardCustomers >= set(persons):
                    occurResSet.add(resId)

            foundPos = -1
            for idx, resId in enumerate(taskQueue):
                if resId in occurResSet:
                    occurResSet.remove(resId)
                else:
                    # bugggggg处理
                    if len(occurResSet) != 0:
                        if taxiId not in PoolingTaskInfo._errorVehs:
                            PoolingTaskInfo._errorVehs.add(taxiId)
                        break
                    # idx即id为taxiId的出租车正在执行的任务序号
                    foundPos = idx
                    break

            # assert foundPos >= 0

            PoolingTaskInfo._poolingRecords[taxiId]._currPos = foundPos

        for taxiID in PoolingTaskInfo._errorVehs:
            PoolingTaskInfo._poolingRecords.pop(taxiID)
            with open(PoolingTaskInfo._args.err_log, 'a') as outf:
                outf.write(f'Timestep: %f, stepBegin, 出错taxiID：%s\n'
                           % (traci.simulation.getTime(), taxiID))
                outf.close()

        PoolingTaskInfo._errorVehs = set()

        # PoolingTaskInfo的静态变量已更新完毕
        PoolingTaskInfo._lockTaskCache.w_release()

    @staticmethod
    def stepFinish():
        PoolingTaskInfo._lockTaskCache.w_acquire()

    ### 以下是成员方法
    def __init__(self, taskQueue: list, initPos=0) -> None:
        # desArea未确定
        super().__init__(None, None)
        self._taskQ = taskQueue
        self._currPos = initPos

    # 同时负责bookkeeping和执行接口的调用
    @staticmethod
    def changeTaskQ(resObj, vID, pickupInsertPos=0, dropoffInsertPos=0) -> bool:
        PoolingTaskInfo._lockTaskCache.w_acquire()
        try:
            if vID in PoolingTaskInfo._poolingRecords:
                isEmptyCar = False
                info: PoolingTaskInfo = PoolingTaskInfo._poolingRecords[vID]
                # pickupInsertPos, dropoffInsertPos要符合一定的规则
                if pickupInsertPos > dropoffInsertPos or dropoffInsertPos > len(info.taskQ) \
                        or pickupInsertPos < info.currPos:
                    raise Exception('changeTaskQ')
                tempTaskQueue = copy.deepcopy(info._taskQ)  # 插入尝试
                currentPos = info._currPos
            else:
                # 这是一辆空车
                isEmptyCar = True
                # pickupInsertPos, dropoffInsertPos要符合一定的规则
                pickupInsertPos = 0
                dropoffInsertPos = 0
                tempTaskQueue = []  # 插入尝试
                currentPos = 0

            # 先插入pickupInsertPos
            tempTaskQueue.insert(pickupInsertPos, resObj.id)
            dropoffInsertPos += 1
            tempTaskQueue.insert(dropoffInsertPos, resObj.id)

            ### 检查插入后连通性
            odPairs = set()

            pickupFormerPos = pickupInsertPos - 1
            pickupLatterPos = pickupInsertPos + 1
            if pickupFormerPos == currentPos - 1:
                # 起点插在了第一个订单前面（或者这个订单派给了一辆空车）
                # buggy
                if isEmptyCar:
                    # 如果这辆车是空车（闲车）
                    # stops = traci.vehicle.getStops(vID)
                    # assert len(stops) == 1
                    # originalStopData = stops[0]
                    # vehNowEdge = traci.lane.getEdgeID(originalStopData.lane)

                    vehNowEdge = traci.vehicle.getRoadID(vID)
                    pickupFormerEdge = vehNowEdge
                else:
                    if traci.vehicle.getStopState(vID) == 0:
                        # 一般是正常状态（车在路线上跑）
                        # buggy
                        vehOnEdge = traci.vehicle.getRoadID(vID)
                        if vehOnEdge == '' or vehOnEdge[0] == ':':
                            # laneID = traci.vehicle.getLaneID(vID)
                            route = traci.vehicle.getRoute(vehID=vID)
                            rIdx = traci.vehicle.getRouteIndex(vID)
                            vehOnEdge = route[rIdx]
                            # raise Exception('changeTaskQ')
                        else:
                            # debug
                            print('ok')
                        pickupFormerEdge = vehOnEdge
                    else:
                        # 如果这辆车恰好停靠在路边让乘客上下车
                        with open(PoolingTaskInfo._args.err_log, 'a') as outf:
                            outf.write(f'Timestep: %f, changeTaskQ 返回 False, \
                                        原因：这辆车恰好停靠在路边让乘客上下车。\nargs = [resObjID = %s, vID = %s, \
                                        pickupInsertPos = %d, dropoffInsertPos = %d]\n'
                                       % (
                                       traci.simulation.getTime(), resObj.id, vID, pickupInsertPos, dropoffInsertPos))
                            outf.close()
                        return False
            else:
                # 起点前有订单任务
                pickupFormerEdge, _ = PoolingTaskInfo.getTaskPointEdge(tempTaskQ=tempTaskQueue, pos=pickupFormerPos)

            pickupEdge = resObj.fromEdge

            if pickupLatterPos == dropoffInsertPos:
                pickupLatterEdge = resObj.toEdge  # 插入订单的终点紧接着起点
            else:
                pickupLatterEdge, _ = PoolingTaskInfo.getTaskPointEdge(tempTaskQ=tempTaskQueue, pos=pickupLatterPos)

            odPairs.add((pickupFormerEdge, pickupEdge))
            odPairs.add((pickupEdge, pickupLatterEdge))

            dropoffFormerPos = dropoffInsertPos - 1
            dropoffLatterPos = dropoffInsertPos + 1
            if dropoffFormerPos == pickupInsertPos:
                dropoffFormerEdge = resObj.fromEdge
            else:
                dropoffFormerEdge, _ = PoolingTaskInfo.getTaskPointEdge(tempTaskQ=tempTaskQueue, pos=dropoffFormerPos)

            dropoffEdge = resObj.toEdge

            if dropoffLatterPos == len(tempTaskQueue):
                # 任务队列队尾放下这个新订单
                dropoffLatterEdge = None
            else:
                dropoffLatterEdge, _ = PoolingTaskInfo.getTaskPointEdge(tempTaskQ=tempTaskQueue, pos=dropoffLatterPos)

            odPairs.add((dropoffFormerEdge, dropoffEdge))
            if dropoffLatterEdge is not None:
                odPairs.add((dropoffEdge, dropoffLatterEdge))

            vType = traci.vehicle.getTypeID(vehID=vID)
            for pair in odPairs:
                # 每个od对都必须相通
                stgRes = traci.simulation.findRoute(fromEdge=pair[0], toEdge=pair[1], vType=vType)
                stgRes_reverse = traci.simulation.findRoute(fromEdge=pair[1], toEdge=pair[0], vType=vType)
                if len(stgRes.edges) == 0 or len(stgRes_reverse.edges) == 0:
                    # 汽车完成这个订单序列或者执行完毕之后“铁定”回不来
                    with open(PoolingTaskInfo._args.err_log, 'a') as outf:
                        outf.write(f'Timestep: %f, changeTaskQ 返回 False, \
                                    原因：汽车完成这个订单序列或者执行完毕之后“铁定”回不来。\
                                    \nargs = [resObjID = %s, vID = %s, pickupInsertPos = %d, dropoffInsertPos = %d]\n'
                                   % (traci.simulation.getTime(), resObj.id, vID, pickupInsertPos, dropoffInsertPos))
                        outf.close()
                    return False

            # 执行派单命令
            try:
                traci.vehicle.dispatchTaxi(vID, tempTaskQueue)
            except Exception as ex:
                print(ex)
                # exit('dispatchTaxi调用失败')
                print('dispatchTaxi调用失败')
                # return False

            # 将修改写入字典
            PoolingTaskInfo._poolingRecords[vID] = PoolingTaskInfo(taskQueue=tempTaskQueue, initPos=currentPos)
            PoolingTaskInfo._persons2Res[resObj.persons] = resObj.id
            PoolingTaskInfo._closedResRecords[resObj.id] = resObj
            return True

        finally:
            PoolingTaskInfo._lockTaskCache.w_release()

    @staticmethod
    def getTaskPointEdge(tempTaskQ, pos):
        appearTimes = tempTaskQ[:pos].count(tempTaskQ[pos])
        if appearTimes == 0:
            # 这是一个pickup点
            atPosEdge = PoolingTaskInfo._closedResRecords[tempTaskQ[pos]].fromEdge
        elif appearTimes == 1:
            # 这是一个dropoff点
            atPosEdge = PoolingTaskInfo._closedResRecords[tempTaskQ[pos]].toEdge
        else:
            raise Exception('changeTaskQ')
        return atPosEdge, appearTimes

    @property
    def taskQ(self):
        return self._taskQ

    @property
    def currPos(self):
        return self._currPos
