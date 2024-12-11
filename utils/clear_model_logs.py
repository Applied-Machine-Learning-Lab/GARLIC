# -*- coding: utf-8 -*-
# @Time : 2024/4/19 14:32
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : clear_model_logs.py
# @Software: PyCharm

import os
import shutil

def clear(pre_str = ''):
    log_dir = pre_str + 'logs/'

    # step 1
    dir = log_dir + 'tensorboard/0'
    if os.path.exists(dir):
        shutil.rmtree(dir)

    # step 2
    dir = log_dir + 'param'
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)
    print('Log dir cleaned!')


if __name__ == '__main__':
    clear('../')