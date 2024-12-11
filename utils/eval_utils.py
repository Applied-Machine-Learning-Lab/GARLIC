# -*- coding: utf-8 -*-
# @Time : 2024/4/3 14:11
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : eval_utils.py
# @Software: PyCharm

import torch
from sklearn.metrics import average_precision_score, accuracy_score, f1_score

def acc_f1(output, labels, average='binary'):
    preds = output.max(1)[1].type_as(labels)
    if preds.is_cuda:
        preds = preds.cpu()
        labels = labels.cpu()
    accuracy = accuracy_score(preds, labels)
    f1 = f1_score(preds, labels, average=average)
    return accuracy, f1


def action_accuracy(out_act_preds, true_action):
    cnt = 0
    sum = 0
    for i in range(out_act_preds.shape[0]):
        true_act = true_action[i].int()
        dis = out_act_preds[i, 0]
        degree = out_act_preds[i, 1]
        if dis <= 0.5:
            action = 0
        else:
            if degree < 1 / 6:
                action = 1
            elif degree < 1 / 3:
                action = 2
            elif degree < 1 / 2:
                action = 3
            elif degree < 2 / 3:
                action = 4
            elif degree < 5 / 6:
                action = 5
            else:
                action = 6
        if action == true_act:
            cnt += 1
        if true_act >= 0:
            sum += 1
    if sum > 0:
        return cnt / sum
    else:
        return -1