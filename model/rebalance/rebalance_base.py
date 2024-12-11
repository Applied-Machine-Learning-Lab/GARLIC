# -*- coding: utf-8 -*-
# @Time : 2024/3/21 13:48
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : rebalance_base.py
# @Software: PyCharm

from ortools.linear_solver import pywraplp

class RebalanceModel(object):

    def decide(self, now, regionalCosts, regionSupplyRates, regionResArrivalRates):
        mapper = list(regionSupplyRates.keys())
        data = {}
        data['num_region'] = len(mapper)
        data['obj_coeffs_regionalCost'] = regionalCosts
        data['constants_regionSupplyRates'] = regionSupplyRates
        data['constants_regionResArrivalRates'] = regionResArrivalRates
        data['threshold'] = 0

        solver = pywraplp.Solver.CreateSolver('GLOP')

        # 定义决策变量
        flowVars = []
        for i in range(data['num_region']):
            row = []
            for j in range(data['num_region']):
                row.append(solver.NumVar(0, solver.infinity(), f'r_%d_%d' % (i, j)))
            flowVars.append(row)

        if sum(data['constants_regionSupplyRates'].values()) >= sum(data['constants_regionResArrivalRates'].values()):
            # 总供给大于等于总需求的情况
            # 定义约束
            for i in range(data['num_region']):
                row = [var for idx, var in enumerate(flowVars[i]) if idx != i]
                col = []
                for j in range(data['num_region']):
                    if j != i:
                        col.append(flowVars[j][i])
                print(data['constants_regionSupplyRates'][mapper[i]])
                print(data['constants_regionResArrivalRates'][mapper[i]])
                solver.Add(data['constants_regionSupplyRates'][mapper[i]] + sum(row) - sum(col) -
                           data['constants_regionResArrivalRates'][mapper[i]] >= -data['threshold'])

        else:
            # 总供给小于总需求的情况
            # 定义约束
            for i in range(data['num_region']):
                row = [var for idx, var in enumerate(flowVars[i]) if idx != i]
                col = []
                for j in range(data['num_region']):
                    if j != i:
                        col.append(flowVars[j][i])

                solver.Add(data['constants_regionSupplyRates'][mapper[i]] + sum(row) - sum(col) -
                           data['constants_regionResArrivalRates'][mapper[i]] <= 0)

                # bounds = (sum(data['constants_regionSupplyRates'].values()) - sum(data['constants_regionResArrivalRates'].values())) / data['num_region']
                # solver.Add(data['constants_regionSupplyRates'][mapper[i]] + sum(row) - sum(col) - data['constants_regionResArrivalRates'][mapper[i]] <= bounds)
                # 已有+流入 >= 流出
                # solver.Add(data['constants_regionSupplyRates'][mapper[i]] + sum(row) - sum(col) >= 0)
            # 定义目标函数
            # 两次规划求解
            # 第一次最小化供需差
            # 可能要设置阈值
            # solver2 = pywraplp.Solver.CreateSolver('GLOP')
        for i in range(data['num_region']):
            solver.Add(flowVars[i][i] == 0)
        # 定义目标函数
        solver.Minimize(sum(
            flowVars[i][j] * data['obj_coeffs_regionalCost'][(mapper[i], mapper[j])] for i in range(data['num_region'])
            for j in range(data['num_region']) if i != j))

        # # 第二次最小化调度成本
        # solver.Minimize(sum)

        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            print('Solution:')
            print('Objective value =', solver.Objective().Value())
            # print('x =', x.solution_value())
            # print('y =', y.solution_value())
            resultFlows = {}
            for i in range(data['num_region']):
                for j in range(data['num_region']):
                    if i != j:
                        fromRegionID = mapper[i]
                        toRegionID = mapper[j]
                        value = flowVars[i][j].solution_value()
                        resultFlows[(fromRegionID, toRegionID)] = value
            return resultFlows
        else:
            print('The problem does not have an optimal solution.')
            return None