# -*- coding: utf-8 -*-
# @Time : 2024/4/3 17:18
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : pytorch operator.py
# @Software: PyCharm

# import torch
# from torch import nn
#
# d = torch.rand(1, 2)
# a = torch.rand(5, 2, 3)
# b = torch.rand(3, 4)
# print(a.shape)
# c = torch.matmul(a, b)
# print(c.shape)
# e = torch.matmul(d, a)
# print(e.shape)
#
# f = torch.rand(5, 2, 6)
# g = torch.cat([a, f], dim=2)
# print(g.shape)

import torch

a = torch.rand(4, 3, 28, 28)
print(a[0].shape)  # 取到第一个维度
print(a[0, 0].shape)  # 取到二个维度
print(a[1, 2, 2, 4])  # 具体到某个元素
print(a[:,0,:,:].unsqueeze(1).shape)

a = a.view(1, -1)
print(a.shape)

b = torch.rand(1, 128)
print(b.shape)

print(torch.cat([a,b],dim=1).shape)