# -*- coding: utf-8 -*-
# @Time : 2023/3/21 13:51
# @Author : Xiao Han
# @E-mail : hahahenha@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : dispatch_base.py
# @Software: PyCharm

import copy
from munkres import Munkres
from ortools.graph import pywrapgraph

class Decision(object):
    def __init__(self, service:dict, rebalance:dict) -> None:
        self._service = service
        self._rebalance = rebalance

    @property
    def service(self):
        return self._service

    @property
    def rebalance(self):
        return self._rebalance

class DispatchModel(object):
    def __init__(self) -> None:
        self._m = Munkres()

    def decideHungarian(self, workerLst: list, taskLst: list, costMatrix: list) -> list:
        # begin = time.time()
        indexes = self._m.compute(costMatrix)
        # end = time.time()
        # print(f'司乘匹配算法用时：%f' % (end - begin))
        returnResult = {}
        for tup in indexes:
            workerName = workerLst[tup[0]]
            taskName = taskLst[tup[1]]
            returnResult[workerName] = taskName
        return returnResult

    def decideMinFlowCost(self, workerLst: list, taskLst: list, costMatrix: list) -> list:
        # 准备solver的input
        self._min_cost_flow = pywrapgraph.SimpleMinCostFlow()
        numWorker = len(workerLst)
        numTask = len(taskLst)
        isLegal = False

        if numTask > numWorker:
            # worker不足的情况
            fakeWorkerNum = numTask - numWorker
            fakeTaskNum = -1
            finalNum = numTask  # 两边图其中一边的节点数
        elif numTask == numWorker:
            fakeWorkerNum = -1
            fakeTaskNum = -1
            finalNum = numTask  # 两边图其中一边的节点数
        else:
            # worker有剩余的情况
            fakeWorkerNum = -1
            fakeTaskNum = numWorker - numTask
            finalNum = numWorker  # 两边图其中一边的节点数

        # 构造source和worker之间的边
        start_nodes = [0 for _ in range(finalNum)]
        for i in range(1, finalNum + 1):
            start_nodes += [i for _ in range(finalNum)]
        start_nodes += [i for i in range(finalNum + 1, finalNum + finalNum + 1)]

        # 构造worker和task之间的边
        end_nodes = [i for i in range(1, finalNum + 1)]
        for _ in range(finalNum):
            tempLst = [k for k in range(finalNum + 1, finalNum + finalNum + 1)]
            end_nodes += tempLst
        end_nodes += [finalNum + finalNum + 1 for _ in range(finalNum)]

        capacities = [1 for _ in range(finalNum)] + [1 for _ in
                                                     range(finalNum * finalNum)] + [1 for _ in range(finalNum)]

        newCostMatrix = copy.deepcopy(costMatrix)
        # 拓展成本矩阵
        if numTask > numWorker:
            for i in range(fakeWorkerNum):
                workerCost = [0 for _ in range(finalNum)]
                newCostMatrix.append(workerCost)
        if numTask < numWorker:
            for lst in newCostMatrix:
                lst += [0 for _ in range(fakeTaskNum)]

        costs = [0 for _ in range(finalNum)]
        for i in range(finalNum):
            costs += newCostMatrix[i]
        costs += [0 for _ in range(finalNum)]
        # costs = (costs)

        source = 0
        sink = finalNum + finalNum + 1
        tasks = finalNum
        supplies = [tasks] + [0 for _ in range(2 * finalNum)] + [-tasks]

        # Add each arc.
        for i in range(len(start_nodes)):
            self._min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i],
                                                              end_nodes[i], capacities[i],
                                                              costs[i])
        # Add node supplies.
        for i in range(len(supplies)):
            self._min_cost_flow.SetNodeSupply(i, supplies[i])

        status = self._min_cost_flow.Solve()
        isLegal = True

        result = {}
        if isLegal is True:
            if status == self._min_cost_flow.OPTIMAL:
                print('Total cost = ', self._min_cost_flow.OptimalCost())
                print()
                for arc in range(self._min_cost_flow.NumArcs()):
                    # Can ignore arcs leading out of source or into sink.
                    if self._min_cost_flow.Tail(arc) != source and self._min_cost_flow.Head(
                            arc) != sink and (self._min_cost_flow.Tail(arc) <= numWorker
                                              and self._min_cost_flow.Head(arc) <= max(numTask, numWorker) + numTask):

                        # Arcs in the solution have a flow value of 1. Their start and end nodes
                        # give an assignment of worker to task.
                        if self._min_cost_flow.Flow(arc) > 0 and self._min_cost_flow.UnitCost(arc) > 0:
                            print('Worker %d assigned to task %d.  Cost = %d' %
                                  (self._min_cost_flow.Tail(arc), self._min_cost_flow.Head(arc),
                                   self._min_cost_flow.UnitCost(arc)))

                            tail = self._min_cost_flow.Tail(arc)
                            head = self._min_cost_flow.Head(arc)

                            result[workerLst[tail - 1]] = taskLst[head - max(numTask, numWorker) - 1]

            else:
                print('There was an issue with the min cost flow input.')
                print(f'Status: {status}')
                raise Exception('DispatchModel')
        else:
            raise Exception('DispatchModel')

        return result