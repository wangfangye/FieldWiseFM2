#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Author:wangfy
@project:PNNConvModel
@Time:2020/9/6 9:27 下午
'''


from sklearn.metrics import log_loss
import random

def compute(p_tr,p_pre):
    return log_loss(p_tr,p_pre)


if __name__ == '__main__':
    p_tr = random.choices([0,1],k=1000000)
    p_pre = [random.random() for _ in range(1000000)]
    print(p_pre)
    print(compute(p_tr,p_pre))