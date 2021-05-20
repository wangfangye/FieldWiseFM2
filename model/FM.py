#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Author:wangfy
@project:DLRec
@Time:2020/4/16 5:20 下午
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_de import FIELD_NUMS
from torchsummary import summary

from layer import FeaturesLinear, FactorizationMachine, FeaturesEmbedding, FeaturesEmbeddingWithGlobalIn


class FactorizationMachineModel(nn.Module):
    """
    wfy test
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """
    def __init__(self,field_dims,embed_dim):
        """
        :param field_dims: list, 每个field的
        :param embed_dim:
        """
        super(FactorizationMachineModel, self).__init__()
        self.fc = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.fm = FactorizationMachine(embed_dim, reduce_sum=True)

    def forward(self,x):
        """
        :param x:  Long tensor of size ``(batch_size, num_fields)``
        :return:  Long tensor [batch_size,1]
        """
        emb_x = self.embedding(x) # (B,nf,embed_size)
        x = self.fc(x) + self.fm(emb_x)
        return x

    def get_l2_loss(self, lambdas=1e-5):
        regularization_loss = 0
        for parameter in self.embedding.parameters():
            regularization_loss += torch.sum(torch.pow(parameter,2))
        return lambdas*regularization_loss



class GFRLFM(nn.Module):
    """
    wfy test
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """
    def __init__(self,field_dims, embed_dim, type="glu",field_len=10):
        """
        :param field_dims: list, 每个field的
        :param embed_dim:
        """
        super(GFRLFM, self).__init__()

        self.fc = FeaturesLinear(field_dims)
        # 特征
        self.embedding = FeaturesEmbeddingWithGlobalIn(field_dims, embed_dim, type=type, field_len=field_len)
        self.fm = FactorizationMachine(embed_dim,reduce_sum=True)

    def forward(self,x):
        """
        :param x:  Long tensor of size ``(batch_size, num_fields)``
        :return:  Long tensor [batch_size,1]
        """
        emb_x = self.embedding(x) # (B,nf,embed_size)
        x = self.fc(x) + self.fm(emb_x)
        # 直接输出sigmoid
        # return torch.sigmoid(x)
        # x = self.last_fc(torch.cat([self.fc(x),self.fm(emb_x)],dim=1))
        return x

    def get_l2_loss(self, lambdas=1e-5):
        regularization_loss = 0
        for parameter in self.embedding.parameters():
            regularization_loss += torch.sum(parameter.pow(2))
        return lambdas*regularization_loss
#
if __name__ == '__main__':
    import numpy as np


    fd = [3, 4]
    embed_dim = 8
    f_n = np.array([[1, 3], [0, 2], [0, 1], [1, 3]])
    f_n = torch.from_numpy(f_n).long()

    model = FactorizationMachineModel(fd,embed_dim)
    label = torch.randint(0,2,(4,1)).float()
    print(label)
    # loss = nn.BCELoss()
    loss = nn.BCEWithLogitsLoss()

    pred = model(f_n)

    print(pred.size())
    losses = loss(pred,label)
    print(losses.item())
    print(losses)
    _,indexs = torch.max(pred,dim=1)
    print(pred)
    print(indexs)

