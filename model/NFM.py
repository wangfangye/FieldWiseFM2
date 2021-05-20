#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Author:wangfy
@project:DLRec
@Time:2020/4/17 4:12 下午
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import FeaturesLinear, FeaturesEmbedding, FactorizationMachine, MultiLayerPerceptron, \
    FeaturesEmbeddingWithGate, FeaturesEmbeddingWithGlobalIn
from model.MyLayers import GLUActivation1D,GenerateConv

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class NeuralFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Neural Factorization Machine.

    将FM的第二部分，
    Reference:
        X He and TS Chua, Neural Factorization Machines for Sparse Predictive Analytics, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_layers=(400, 400, 400), dropouts=(0.5, 0.5)):
        super(NeuralFactorizationMachineModel, self).__init__()

        self.linear = FeaturesLinear(field_dims)  # 线性部分
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)  # 交互部分
        self.fm = FactorizationMachine(embed_dim, reduce_sum=False)
        self.mlp = MultiLayerPerceptron(embed_dim, mlp_layers, dropouts[1])

    def forward(self, x):
        """
        :param x: [B,n_f]
        :return: [B,1]
        """
        x_fm = self.fm(self.embedding(x))  # [B,d]
        x = self.linear(x) + self.mlp(x_fm)
        return x

    def get_l2_loss(self, lambdas=1e-5):
        regularization_loss = 0
        for parameter in self.mlp.parameters():
            regularization_loss += torch.sum(parameter.pow(2))
        return lambdas*regularization_loss



class GFRLNFM(torch.nn.Module):
    """
    在 NFM的embedding层后添加GEL 强化特征
    """

    def __init__(self, field_dims, embed_dim, mlp_layers=(400,400,400),dropouts=(0.5, 0.5),type="glu"):
        super(GFRLNFM, self).__init__()

        self.linear = FeaturesLinear(field_dims)  # 线性部分

        self.embedding_gate = FeaturesEmbeddingWithGlobalIn(field_dims,embed_dim,type=type)

        self.fm = FactorizationMachine(reduce_sum=False)

        self.mlp = MultiLayerPerceptron(embed_dim, mlp_layers, dropouts[1])

    def forward(self, x):
        """
        :param x: [B,n_f]
        :return: [B,1]
        """

        x_fm = self.fm(self.embedding_gate(x))  # [B,k]
        x = self.linear(x) + self.mlp(x_fm)
        return x

class NFMInterGate(torch.nn.Module):
    """
    NFM 的x_inter之后的层改为添加 gate，gate的数量是超参数
    """

    def __init__(self, field_dims, embed_dim, mlp_layers=(400,400,400), glunum=2, dropouts=(0.1, 0.1)):
        super().__init__()

        self.linear = FeaturesLinear(field_dims)  # 线性部分
        # self.embedding_gate = FeaturesEmbeddingWithGate(field_dims,embed_dim,glu_num=glunum)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)  # 交互部分，先
        # 第二层： 门机制，提取信息
        #
        self.gene_inter = GenerateConv()

        glu_list = list()
        for _ in range(glunum):
            glu_list.append(GLUActivation1D(embed_dim, int(embed_dim*2)))
        self.glus = torch.nn.Sequential(*glu_list)
        # self.mlp_input = len(field_dims) * (len(field_dims)-1)/2
        self.mlp = MultiLayerPerceptron(embed_dim, mlp_layers, dropouts[1])


    def forward(self, x):
        """
        :param x: [B,n_f]
        :return: [B,1]
        """
        # x_fm1 = self.fm1(self.embedding(x))
        x_lin = self.linear(x)
        x_emb = self.embedding(x)

        # 第二层
        x_inter = self.gene_inter(x_emb) # 构建一个交互的层
        x_inter = self.glus(x_inter)  # [B,f*(f-1)/2,emb_size]
        # print(x_inter.size())
        x_nfm = torch.sum(x_inter, dim=1)

        x = x_lin + self.mlp(x_nfm)

        return x


def NFM_test():
    import numpy as np

    fd = [3000, 4000,5000]
    embed_dim = 32
    mlp_layers = (128, 16, 8)

    f_n = np.array([[1, 3,4], [0, 2,5], [0, 1,2], [1, 3,2], [1, 3,2], [0, 2,2], [0, 1,2], [1, 3,2], [1, 3,2], [0, 2,2], [0, 1,2], [1, 3,2]])
    f_n = torch.from_numpy(f_n).long()
    # model = NeuralFactorizationMachineModel(fd, embed_dim, mlp_layers)
    model = NeuralFactorizationMachineModel(fd, embed_dim, mlp_layers)
    print(count_params(model))

    label = torch.randint(0, 2, (12, 1)).float()
    print(label)
    loss = torch.nn.BCEWithLogitsLoss()
    pred = model(f_n)
    print(pred.size())
    print("loss:{:.5}".format(loss(pred, label)))


def count_params(model):
    params = sum(param.numel() for param in model.parameters())
    return params


if __name__ == '__main__':
    NFM_test()
