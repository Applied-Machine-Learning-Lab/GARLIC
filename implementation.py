# -*- coding: utf-8 -*-
# @Time : 2024/3/27 10:04
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : implementation.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import traci
import torch
from config import parser
from sumolib import checkBinary
from utils.sumo_utils import Controller, PoolingTaskInfo

def run(args):
    """execute the TraCI control loop"""
    controller = Controller(args)
    step = 0
    # 模拟进行4个小时
    PoolingTaskInfo.init(args)
    while step < 15000:
        flag = controller.step()
        if flag == -1:
            break
        traci.simulationStep()
        step += 1
    traci.close()
    controller.finish()
    sys.stdout.flush()

# this is the main entry point of this script
if __name__ == "__main__":
    # we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    args = parser.parse_args()
    args.device = torch.device("cpu")
    if args.cuda >= 0:
        args.device = torch.device("cuda:" + str(args.cuda))

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if args.nogui:
        sumoBinary = checkBinary('sumo')

        traci.start(["sumo", "-c", args.sim_file, "--tripinfo-output.write-unfinished", "--tripinfo-output",
                     args.trip_info, "--stop-output", args.stop_info, '--vehroute-output', args.vehicle_route,
                     "--vehroute-output.exit-times", "--vehroute-output.write-unfinished", "--quit-on-end",
                     "--start"])
    else:
        sumoBinary = checkBinary('sumo-gui')

        # this is the normal way of using traci. sumo is started as a
        # subprocess and then the python script connects and runs
        traci.start(["sumo-gui", "-c", args.sim_file, "--tripinfo-output.write-unfinished", "--tripinfo-output",
                     args.trip_info, "--stop-output", args.stop_info, '--vehroute-output', args.vehicle_route,
                     "--vehroute-output.exit-times", "--vehroute-output.write-unfinished", "--quit-on-end",
                     "--start"])
    run(args)