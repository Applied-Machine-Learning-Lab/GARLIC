# -*- coding: utf-8 -*-
# @Time : 2024/4/3 16:09
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : train_utils.py
# @Software: PyCharm
import math

import torch
from torch import nn
import torch.nn.functional as F


def format_metrics(metrics, split):
    """Format metric in metric dict for logging."""
    return " ".join(
            ["{}_{}: {:.4f}".format(split, metric_name, metric_val) for metric_name, metric_val in metrics.items()])

def select_action(out_act, true_act=0):
    dis = out_act[0, 0]
    degree = out_act[0, 1]
    if dis <= 0.5:
        action = 0
        tmp_degree = 1/12
    else:
        if degree < 1/6:
            action = 1
            tmp_degree = degree
        elif degree < 1/3:
            action = 2
            tmp_degree = torch.min(degree - 1/6, 1/3 - degree)
        elif degree < 1/2:
            action = 3
            tmp_degree = torch.min(degree - 1/3, 1/2 - degree)
        elif degree < 2/3:
            action = 4
            tmp_degree = torch.min(degree - 1/2, 2/3 - degree)
        elif degree < 5/6:
            action = 5
            tmp_degree = torch.min(degree - 2/3, 5/6 - degree)
        else:
            action = 6
            tmp_degree = torch.min(degree - 5/6, 1.0 - degree)
    action_prob = (0.5 * torch.abs(dis - 0.5)) + (3 * tmp_degree)
    if true_act == action:
        action_prob += 0.5
    return action, action_prob