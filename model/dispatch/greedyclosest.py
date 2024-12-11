# -*- coding: utf-8 -*-
# @Time : 2024/3/21 14:10
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : greedyclosest.py
# @Software: PyCharm

import numpy as np

from model.dispatch.dispatch_base import Decision


def GreedyCloestEuclidean(vacantFleet, openReservationPool, traci):
    vIDLst = list(vacantFleet)
    resIDLst = list(openReservationPool.keys())

    distanceVRLst = []

    vEdgePos = {}
    for vID in vIDLst:
        vX0, vY0 = traci.vehicle.getPosition(vehID=vID)
        vEdgePos[vID] = {'coord': (vX0, vY0)}

    resEdgePosEnterTime = {}
    for resID in resIDLst:
        toRes = []
        res = openReservationPool[resID]
        resAtEdge = res.fromEdge
        resAtPos = res.departPos
        rX, rY = traci.simulation.convert2D(resAtEdge, resAtPos, toGeo=False)
        resEnterTime = res.depart
        resEdgePosEnterTime[resID] = {'edge': resAtEdge, 'pos': resAtPos, 'coord': (rX, rY), 'enterTime': resEnterTime}

    # 先来先服务
    resIDLst = sorted(resIDLst, key=lambda resID: resEdgePosEnterTime[resID]['enterTime'])

    # 算出每个订单和每辆车之间的曼哈顿距离
    for resID in resIDLst:
        toRes = []
        for vID in vIDLst:
            vX, vY = vEdgePos[vID]['coord']
            rX, rY = resEdgePosEnterTime[resID]['coord']
            dist = abs(vX - rX) + abs(vY - rY)
            toRes.append(dist)
        distanceVRLst.append(toRes)

    distanceVRArr = np.array(distanceVRLst)
    matchMatrix = np.zeros([len(resIDLst), len(vIDLst)])
    i = 0
    assignedVCnt = 0

    while i < len(resIDLst):
        if assignedVCnt == len(vIDLst):
            break
        toRes = distanceVRArr[i]
        result = np.where(toRes == np.amin(toRes))
        j = result[0][0]
        matchMatrix[i][j] = 1
        distanceVRArr[:, j] = np.inf
        i += 1
        assignedVCnt += 1
        # if assignedVCnt == len(vIDLst):
        #     break

    service = {}
    rebalance = {}

    for i in range(len(resIDLst)):
        for j in range(len(vIDLst)):
            if matchMatrix[i][j] == 1:
                matchedResID = resIDLst[i]
                matchedVID = vIDLst[j]
                service[matchedVID] = matchedResID

    return Decision(service=service, rebalance=rebalance)