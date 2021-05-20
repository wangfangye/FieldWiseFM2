#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Author:wangfy
@project:DLRec
@Time:2020/4/17 8:31 下午
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import FeaturesLinear

class LogisticRegression(torch.nn.Module):
    def __init__(self,field_dims):
        super(LogisticRegression, self).__init__()
        self.linear = FeaturesLinear(field_dims)
        
    def forward(self,x):
        """
        :param x: [B,num_fields]
        :return: [B,1]
        """
        x = self.linear(x)
        return x

if __name__ == '__main__':
    import numpy as np
    field_dims = [3,4]

    input = np.array([[1,2],[0,1],[1,3],[1,2],[0,1],[1,3]])
    input = torch.from_numpy(input)
    # input = input.tolist()
    model = LogisticRegression(field_dims)

    out = model(input)
    print(out)
    print(out)



