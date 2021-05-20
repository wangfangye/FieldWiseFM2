#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Author:wangfy
@project:PNNConvModel
@Time:2021/4/27 9:26 上午
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_de import FIELD_NUMS
from torchsummary import summary

from layer import FeaturesLinear, FactorizationMachine, FeaturesEmbedding, IFMLayer


class IFM(nn.Module):
    """
    """
    def __init__(self,field_dims,embed_dim, embed_dims, field_len=10):
        """
        :param field_dims: list, 每个field的
        :param embed_dim:
        """
        super(IFM, self).__init__()

        self.fc = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.ifm = IFMLayer(field_dims, embed_dim, embed_dims=embed_dims,field_len=field_len)

        self.fm = FactorizationMachine(embed_dim,reduce_sum=True)
        # self.last_fc = nn.Linear(2,1)

    def forward(self,x):
        """
        :param x:  Long tensor of size ``(batch_size, num_fields)``
        :return:  Long tensor [batch_size,1]
        """
        emb_x = self.embedding(x) # (B,nf,embed_size)
        emb_x, x_att = self.ifm(emb_x)
        x = self.fc(x, weights=x_att) + self.fm(emb_x)
        return x

    def get_l2_loss(self, lambdas=1e-5):
        regularization_loss = 0
        for parameter in self.ifm.parameters():
            regularization_loss += torch.sum(torch.pow(parameter,2))
        return lambdas * regularization_loss