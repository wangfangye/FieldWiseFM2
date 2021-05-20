#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Author:wangfy
@project:PNNConvModel
@Time:2021/3/18 7:06 下午
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import FeaturesEmbedding

class FmFM(nn.Module):
    def __init__(self,field_dims,embed_dim):
        super(FmFM, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims,embed_dim)



    def forward(self,x):


        return x