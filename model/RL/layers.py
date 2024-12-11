# -*- coding: utf-8 -*-
# @Time : 2024/4/18 10:30
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : layers.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, num_state, num_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.action_head = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, num_state):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.state_value = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value

class Reward(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, out_size, n_layers=1, batch_size=1):
        super(Reward, self).__init__()

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.out_size = out_size

        # 这里指定了BATCH FIRST,所以输入时BATCH应该在第一维度
        self.gru = torch.nn.GRU(input_dim, hidden_size, n_layers, batch_first=True, bidirectional=True)

        # 加了一个线性层，全连接
        self.fc1 = torch.nn.Linear(hidden_size * 2, hidden_size)
        # 加入了第二个全连接层
        self.fc2 = torch.nn.Linear(hidden_size, out_size)

    def forward(self, inputs, hidden):
        # hidden 就是上下文输出，output 就是 RNN 输出
        output, hidden = self.gru(inputs, hidden)
        # output是所有隐藏层的状态，hidden是最后一层隐藏层的状态
        output = self.fc1(output)
        output = self.fc2(output)

        # 仅仅获取 time seq 维度中的最后一个向量
        # the last of time_seq
        output = output[:, -1, :]

        return output, hidden

    def init_hidden(self, batch_size=None):
        # 这个函数写在这里，有一定迷惑性，这个不是模型的一部分，是每次第一个向量没有上下文，在这里捞一个上下文，仅此而已。
        if batch_size == None:
            batch_size = self.batch_size
        hidden = torch.autograd.Variable(
            torch.zeros(2 * self.n_layers, batch_size, self.hidden_size))
        return hidden