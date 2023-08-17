# -*- coding: utf-8 -*-
# @Time : 2023/3/21 9:56
# @Author : Xiao Han
# @E-mail : hahahenha@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : Logger.py
# @Software: PyCharm


from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse

# we need to import python modules from the $SUMO_HOME/tools directory

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import sumolib

import xml.etree.ElementTree as ET
import xml


class LogPrinter(object):
    def __init__(self, logdir) -> None:
        self._logDirPath = logdir
        self._eachRegionDemandSupplyRealtimePath = self._logDirPath + r'/each_region_demand_supply_realtime.xml'
        self._rebalanceRecordPath = self._logDirPath + r'/rebalance_record.xml'
        self._servicePredRecordPath = self._logDirPath + r'/service_pred_record.xml'
        self._taxiStatNumRecordPath = self._logDirPath + r'/taxi_stat_num_record.xml'
        self._eachRegionDemandSupplyAggregatePath = self._logDirPath + r'/each_region_demand_supply_aggregate.xml'
        self._eachRegionDemandSupplyPredPath = self._logDirPath + r'/each_region_demand_supply_pred.xml'

        self._settingsDict = {}

        self._dataEachRegionDemandSupplyRealtime = ET.Element("root")
        self._dataEachRegionDemandSupplyAggregate = ET.Element("root")
        self._dataEachRegionDemandSupplyPred = ET.Element("root")
        self._dataRebalanceRecord = ET.Element("root")
        self._dataRebalanceRecord2Print = ET.Element("root")
        self._dataServicePredRecord = ET.Element("root")
        self._dataTaxiStatNumRecord = ET.Element("root")

        self._interRegionCost = {}

        with open(self._eachRegionDemandSupplyRealtimePath, 'w') as outf:
            sumolib.xml.writeHeader(outf=outf, root='regiondemandsupplyinfo')
            outf.close()
        with open(self._rebalanceRecordPath, 'w') as outf:
            sumolib.xml.writeHeader(outf=outf, root='rebalanceinfo')
            outf.close()
        with open(self._servicePredRecordPath, 'w') as outf:
            sumolib.xml.writeHeader(outf=outf, root='servicepredinfo')
            outf.close()
        with open(self._taxiStatNumRecordPath, 'w') as outf:
            sumolib.xml.writeHeader(outf=outf, root='taxistatnuminfo')
            outf.close()
        with open(self._eachRegionDemandSupplyAggregatePath, 'w') as outf:
            sumolib.xml.writeHeader(outf=outf, root='regiondemandsupplyinfo')
            outf.close()
        with open(self._eachRegionDemandSupplyPredPath, 'w') as outf:
            sumolib.xml.writeHeader(outf=outf, root='regiondemandsupplyinfo')
            outf.close()

    def updatePredEachRegionDemandSupply(self, startTime, timespan, regionD, regionS):
        parent = ET.Element('regiondemandsupply', attrib={'startTime': str(startTime), 'timeSpan': str(timespan)})
        for key, value in regionD.items():
            demandNum = value
            supplyNum = regionS[key]
            child = ET.Element('region', attrib={'id': str(key), 'demand': str(demandNum), 'supply': str(supplyNum)})
            parent.append(child)
        self._dataEachRegionDemandSupplyPred.append(parent)

    def updateAggregateEachRegionDemandSupply(self, startTime, timespan, regionD, regionS):
        parent = ET.Element('regiondemandsupply', attrib={'startTime': str(startTime), 'timeSpan': str(timespan)})
        for key, value in regionD.items():
            demandNum = value
            supplyNum = regionS[key]
            child = ET.Element('region', attrib={'id': str(key), 'demand': str(demandNum), 'supply': str(supplyNum)})
            parent.append(child)
        self._dataEachRegionDemandSupplyAggregate.append(parent)

    def updateRealTimeEachRegionDemandSupply(self, time, regionD, regionS):
        parent = ET.Element('regiondemandsupply', attrib={'time': str(time)})
        for key, value in regionD.items():
            demandNum = value
            supplyNum = regionS[key]
            child = ET.Element('region', attrib={'id': str(key), 'demand': str(demandNum), 'supply': str(supplyNum)})
            parent.append(child)
        self._dataEachRegionDemandSupplyRealtime.append(parent)

    def updateRebalanceRecord(self, vehID, departTime=None, fromRegion=None,
                              toRegion=None, fromEdge=None, toEdge=None,
                              routeLength=None, predDuration=None, isDepart=True,
                              arrivalTime=None):
        if isDepart:
            child = ET.Element('rebalancetask', attrib={'departTime': str(departTime),
                                                        'vehID': str(vehID), 'fromRegion': str(fromRegion),
                                                        'toRegion': str(toRegion), 'fromEdge': str(fromEdge),
                                                        'toEdge': str(toEdge), 'routeLength': str(routeLength),
                                                        'predDuration': str(predDuration)})
            self._dataRebalanceRecord.append(child)
        else:
            children = self._dataRebalanceRecord.findall(f".//*[@vehID='%s']" % (vehID))
            if len(children) != 1:
                raise Exception('updateRebalanceRecord')
            child = children[0]
            actualDuration = arrivalTime - float(child.attrib['departTime'])
            child.set('arrivalTime', str(arrivalTime))
            child.set('actualDuration', str(actualDuration))
            # child.set('actualLength', str(actualLength))
            self._dataRebalanceRecord.remove(child)
            self._dataRebalanceRecord2Print.append(child)

    def updateServicePredRecord(self, vehID, predArrivalTime, actArrivalTime):
        child = ET.Element('servicetask', attrib={'vehID': str(vehID),
                                                  'predArrivalTime': str(predArrivalTime),
                                                  'actArrivalTime': str(actArrivalTime)})
        self._dataServicePredRecord.append(child)

    def updateTaxiStatNumRecord(self, time, vacantNum, servingNum, rebalancingNum):
        child = ET.Element('taxistatnum', attrib={'time': str(time), 'vacantNum': str(vacantNum),
                                                  'servingNum': str(servingNum),
                                                  'rebalancingNum': str(rebalancingNum)})
        self._dataTaxiStatNumRecord.append(child)

    def printout(self):
        # 各状态车辆数
        with open(self._taxiStatNumRecordPath, 'a') as outf:
            for child in self._dataTaxiStatNumRecord:
                # print(str(ET.tostring(child)))
                outf.writelines(['\t' + ET.tostring(child).decode('UTF-8') + '\n'])

            outf.close()
            del self._dataTaxiStatNumRecord
            self._dataTaxiStatNumRecord = ET.Element("root")

        # 各区域现有供给和开放需求数
        with open(self._eachRegionDemandSupplyRealtimePath, 'a') as outf:
            for child in self._dataEachRegionDemandSupplyRealtime:
                attrstr = ''
                for key, value in child.attrib.items():
                    attrstr += ' ' + key + '=' + '"' + value + '"'
                outf.writelines(['\t' + f'<%s%s>' % (child.tag, attrstr) + '\n'])

                for subelement in child.iter('region'):
                    outf.writelines(['\t\t' + ET.tostring(subelement).decode('UTF-8') + '\n'])

                outf.writelines(['\t' + f'</%s>' % (child.tag) + '\n'])
            outf.close()
            del self._dataEachRegionDemandSupplyRealtime
            self._dataEachRegionDemandSupplyRealtime = ET.Element("root")

        # 平衡调度记录
        with open(self._rebalanceRecordPath, 'a') as outf:
            for child in self._dataRebalanceRecord2Print:
                outf.writelines(['\t' + ET.tostring(child).decode('UTF-8') + '\n'])
            outf.close()
            del self._dataRebalanceRecord2Print
            self._dataRebalanceRecord2Print = ET.Element("root")

        # 服务预测记录
        with open(self._servicePredRecordPath, 'a') as outf:
            for child in self._dataServicePredRecord:
                outf.writelines(['\t' + ET.tostring(child).decode('UTF-8') + '\n'])
            outf.close()
            del self._dataServicePredRecord
            self._dataServicePredRecord = ET.Element("root")

        # 各区域在一定时间窗内现有累计需求量和累计到达供应量
        with open(self._eachRegionDemandSupplyAggregatePath, 'a') as outf:
            for child in self._dataEachRegionDemandSupplyAggregate:
                attrstr = ''
                for key, value in child.attrib.items():
                    attrstr += ' ' + key + '=' + '"' + value + '"'
                outf.writelines(['\t' + f'<%s%s>' % (child.tag, attrstr) + '\n'])

                for subelement in child.iter('region'):
                    outf.writelines(['\t\t' + ET.tostring(subelement).decode('UTF-8') + '\n'])

                outf.writelines(['\t' + f'</%s>' % (child.tag) + '\n'])
            outf.close()
            del self._dataEachRegionDemandSupplyAggregate
            self._dataEachRegionDemandSupplyAggregate = ET.Element("root")

        # 各区域在一定时间窗内现有累计需求量和累计到达供应量的预计值
        with open(self._eachRegionDemandSupplyPredPath, 'a') as outf:
            for child in self._dataEachRegionDemandSupplyPred:
                attrstr = ''
                for key, value in child.attrib.items():
                    attrstr += ' ' + key + '=' + '"' + value + '"'
                outf.writelines(['\t' + f'<%s%s>' % (child.tag, attrstr) + '\n'])

                for subelement in child.iter('region'):
                    outf.writelines(['\t\t' + ET.tostring(subelement).decode('UTF-8') + '\n'])

                outf.writelines(['\t' + f'</%s>' % (child.tag) + '\n'])
            outf.close()
            del self._dataEachRegionDemandSupplyPred
            self._dataEachRegionDemandSupplyPred = ET.Element("root")

    def finish(self):
        with open(self._taxiStatNumRecordPath, 'a') as outf:
            outf.writelines(['</taxistatnuminfo>'])
            outf.close()
        with open(self._eachRegionDemandSupplyRealtimePath, 'a') as outf:
            outf.writelines(['</regiondemandsupplyinfo>'])
            outf.close()
        with open(self._rebalanceRecordPath, 'a') as outf:
            outf.writelines(['</rebalanceinfo>'])
            outf.close()
        with open(self._servicePredRecordPath, 'a') as outf:
            outf.writelines(['</servicepredinfo>'])
            outf.close()
        with open(self._eachRegionDemandSupplyAggregatePath, 'a') as outf:
            outf.writelines(['</regiondemandsupplyinfo>'])
            outf.close()
        with open(self._eachRegionDemandSupplyPredPath, 'a') as outf:
            outf.writelines(['</regiondemandsupplyinfo>'])
            outf.close()


if __name__ == "__main__":
    lp = LogPrinter()

    for time in range(10):
        lp.updateTaxiStatNumRecord(time=time, vacantNum=12, servingNum=3, rebalancingNum=4)

    lp.updateAggregateEachRegionDemandSupply(startTime=2, timespan=10, regionD={3: 0, 1: 20, 2: 7},
                                             regionS={3: 0, 2: 9, 1: 5})

    lp.updatePredEachRegionDemandSupply(startTime=2, timespan=10, regionD={3: 0, 1: 20, 2: 7},
                                        regionS={3: 0, 2: 9, 1: 5})

    lp.updateRebalanceRecord(isDepart=True, departTime=15.0, vehID='taxi2', fromRegion=142, toRegion=100,
                             fromEdge='123', toEdge='122', predDuration=500, routeLength=1500.4)
    lp.updateRebalanceRecord(isDepart=False, vehID='taxi2', arrivalTime=652)

    lp.updateServicePredRecord(vehID='taxi0', resID='12345', departTime=3.0, predDuration=24124, predLength=1234.2)

    lp.printout()

    lp.finish()

    # tree = ET.parse(r'C:\Users\HW\Desktop\41\sim-ult-3.23 - 副本 - 副本\data\mylogs\taxi_stat_num_record.xml')
    # print(11)