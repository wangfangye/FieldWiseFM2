#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Author:wangfy
@project:DLRec
@Time:2020/4/20 7:11 下午
'''



import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import FeaturesLinear,FeaturesEmbedding,FactorizationMachine,MultiLayerPerceptron,FeaturesEmbeddingWithGate,FeaturesEmbeddingWithGlobalIn

"""
A pytorch implementation of DeepFM.

Reference:
    H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
"""

class DeepFactorizationMachineModel(nn.Module):
    """
    DeepFM,主要由三部分组成
    1、线性部分
    2、FM 部分
    3、Deep部分
    """
    def __init__(self,field_dims,embed_dim,mlp_layers=(128,64),dropout=0.5,field_len = 10):
        super(DeepFactorizationMachineModel, self).__init__()
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(embed_dim=embed_dim,reduce_sum=True)

        self.embedding = FeaturesEmbedding(field_dims,embed_dim)
        self.embed_output_size = field_len * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_size,mlp_layers,dropout)
        # self.last_fc = nn.Linear(3,1)

    def forward(self,x):
        """
        :param x:
        :return:
        """
        x_embed = self.embedding(x) # [B,n_f,e]
        x_out = self.linear(x) + self.fm(x_embed) + self.mlp(x_embed.view(x.size(0),-1))
        # x_con = torch.cat([self.linear(x),self.fm(x_embed),self.mlp(x_embed.view(x.size(0),-1))],dim=1)
        # x_out = self.last_fc(x_con)
        return x_out

    def get_l2_loss(self, lambdas=1e-5):
        regularization_loss = 0
        for parameter in self.embedding.parameters():
            regularization_loss += torch.sum(parameter.pow(2))
        for parameter in self.mlp.parameters():
            regularization_loss += torch.sum(parameter.pow(2))
        return lambdas*regularization_loss

class GELDFM(nn.Module):
    """
    DeepFM,主要由三部分组成
    1、线性部分
    2、FM 部分
    3、Deep部分
    """
    def __init__(self,field_dims,embed_dim,mlp_layers=(128,64),dropout=0.5,type="glu"):
        super(GELDFM, self).__init__()
        self.linear = FeaturesLinear(field_dims)

        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbeddingWithGlobalIn(field_dims,embed_dim,type=type)
        self.embed_output_size = len(field_dims) * embed_dim

        self.mlp = MultiLayerPerceptron(self.embed_output_size,mlp_layers,dropout)
        # self.last_fc = nn.Linear(3,1)

    def forward(self,x):
        """
        :param x:
        :return:
        """
        x_embed = self.embedding(x) # [B,n_f,e]
        x_out = self.linear(x) + self.fm(x_embed) + self.mlp(x_embed.view(x.size(0),-1))
        # x_con = torch.cat([self.linear(x),self.fm(x_embed),self.mlp(x_embed.view(x.size(0),-1))],dim=1)
        # x_out = self.last_fc(x_con)
        return x_out

class GFRLDFM(nn.Module):
    """
    DeepFM + gate
    """
    def __init__(self,field_dims,embed_dim,mlp_layers=(128,64),dropout=0.5,type="glu"):
        super(GFRLDFM,self).__init__()

        self.linear = FeaturesLinear(field_dims)

        self.fm = FactorizationMachine(reduce_sum=True)

        # self.embedding = FeaturesEmbedding(field_dims,embed_dim)
        self.embedding_gfrl = FeaturesEmbeddingWithGlobalIn(field_dims,embed_dim,type=type)
        self.embed_output_size = len(field_dims) * embed_dim

        self.mlp = MultiLayerPerceptron(self.embed_output_size,mlp_layers,dropout)

        self.last_fc = nn.Linear(3,1)

    def forward(self,x):
        """
        :param x:
        :return:
        """
        x_embed = self.embedding_gfrl(x) # [B,n_f,e]
        x_out = self.linear(x) + self.fm(x_embed) + self.mlp(x_embed.view(x.size(0),-1))
        # x_con = torch.cat([self.linear(x),self.fm(x_embed),self.mlp(x_embed.view(x.size(0),-1))],dim=1)
        # x_out = self.last_fc(x_con)
        return x_out




if __name__ == '__main__':
    import numpy as np

    fd = [3, 4]
    embed_dim = 8
    f_n = np.array([[1, 3], [0, 2], [0, 1], [1, 3]])
    f_n = torch.from_numpy(f_n).long()

    model = DeepFMGate(fd, embed_dim, glunum=4)

    print(model)

    label = torch.randint(0, 2, (4, 1)).float()
    print(label)
    loss = nn.BCEWithLogitsLoss()

    pred = model(f_n)

    print(pred.size())
    losses = loss(pred, label)
    print(losses)
    # _, indexs = torch.max(pred, dim=1)
    # print(pred)
    # print(indexs)