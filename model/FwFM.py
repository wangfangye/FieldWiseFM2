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
from layer import FeaturesLinear, FactorizationMachine, FeaturesEmbedding, FeaturesEmbeddingWithGlobalIn,FwFMInterLayer

class FwFM(nn.Module):
    # 在二阶权重上存在
    def __init__(self,field_dims,embed_dim):
        super(FwFM, self).__init__()
        self.lr = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims,embed_dim)
        self.fwfm = FwFMInterLayer(field_dims)


    def forward(self,x):
        emb_x = self.embedding(x)  # (B,nf,embed_size)
        x = self.fc(x) + self.fwfm(emb_x)
        return x