#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Author:wangfy
@project:DLRec
@Time:2020/4/19 2:11 下午
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import FeaturesEmbedding, FeaturesLinear, AttentionalFactorizationMachine,FeaturesEmbeddingWithGate,FeaturesEmbeddingWithGlobalIn
from model.MyLayers import GlobalGluIn,GFRL
from model.AttentionLayer import GeneralAttention
from utils_de import *
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class AttentionalFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Attentional Factorization Machine.

    Reference:
        J Xiao, et al. Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks, 2017.
    """

    def __init__(self, field_dims, embed_dim, attn_size, dropouts=(0.5,0.5)):
        super().__init__()
        self.num_fields = 10
        # 第一层
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)  # 线性部分

        # 第三层
        self.afm = AttentionalFactorizationMachine(
            embed_dim, attn_size, dropouts)
        # self.last_fc = torch.nn.Linear(2,1)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = self.linear(x) + self.afm(self.embedding(x))  # afm的输入时embed之后的，[B,n_f,embed_size]
        # x = self.last_fc(torch.cat([self.linear(x),self.afm(self.embedding(x))],dim=-1))
        return x

    def get_l2_loss(self, lambdas=2):
        regularization_loss = 0
        for parameter in self.afm.parameters():
            regularization_loss += torch.sum(parameter.pow(2))
        for parameter in self.embedding.parameters():
            regularization_loss += torch.sum(parameter.pow(2))
        return lambdas*regularization_loss




class AFMWithGate(torch.nn.Module):
    """
    A pytorch implementation of Attentional Factorization Machine.

    Reference:
        J Xiao, et al. Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks, 2017.
    """

    def __init__(self, field_dims, embed_dim, attn_size, glunum=2,dropouts=(0.1,0.1)):
        super().__init__()
        # self.num_fields = len(field_dims)
        # 第一层
        self.embedding_gate = FeaturesEmbeddingWithGate(field_dims,embed_dim,glu_num=glunum)
        # self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)  # 线性部分

        # 第三层
        self.afm = AttentionalFactorizationMachine(
            embed_dim, attn_size, dropouts)
        self.last_fc = torch.nn.Linear(2,1)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # x = self.linear(x) + self.afm(self.embedding(x))  # afm的输入时embed之后的，[B,n_f,embed_size]
        x = self.last_fc(torch.cat([self.linear(x),self.afm(self.embedding_gate(x))],dim=-1))
        return x

class GFRLAFM(torch.nn.Module):
    """
    A pytorch implementation of Attentional Factorization Machine.

    Reference:
        J Xiao, et al. Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks, 2017.
    """

    def __init__(self, field_dims, embed_dim, attn_size,dropouts=(0.5,0.5),type="glu"):
        super().__init__()
        self.num_fields = len(field_dims)
        # 第一层
        self.embedding = FeaturesEmbeddingWithGlobalIn(field_dims,embed_dim,type=type)
        # self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)  # 线性部分

        # 第三层
        self.afm = AttentionalFactorizationMachine(
            embed_dim, attn_size, dropouts)
        # self.last_fc = torch.nn.Linear(2,1)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = self.linear(x) + self.afm(self.embedding(x))  # afm的输入时embed之后的，[B,n_f,embed_size]
        # x = self.last_fc(torch.cat([self.linear(x),self.afm(self.embedding_gate(x))],dim=-1))
        return x


class HirAFM(torch.nn.Module):
    """
    使用多通道的AFM
    """

    def __init__(self, field_dims, embed_dim, attn_size, glunum=3, dropouts=(0.5,0.5)):
        super().__init__()
        self.num_fields = len(field_dims)
        # 第一层
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)  # 线性部分

        # glus层，强化每一层的
        self.glus = nn.ModuleList(
            [GFRL(len(field_dims), embed_dim) for _ in range(glunum)]
        )

        # off-the-shelf，含有参数
        self.afm = AttentionalFactorizationMachine(
            embed_dim, attn_size, dropouts, reduce=False)
        # self.last_fc = torch.nn.Linear(2,1)
        self.att = GeneralAttention(embed_dim,conv_size=32)
        self.last_fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x_emb = self.embedding(x)
        x_lin = self.linear(x)
          # afm的输入时embed之后的，[B,n_f,embed_size]
        x_afm0 =self.afm(x_emb)

        x_afms = [self.afm(glu_layer(x_emb)) for glu_layer in self.glus]
        x_afms.append(x_afm0)
        x_afms = torch.stack(x_afms,dim=1)

        x_att,x_acores = self.att(x_afms)
        # x = self.last_fc(torch.cat([self.linear(x),self.afm(self.embedding(x))],dim=-1))
        x_out = self.last_fc(x_att) + x_lin
        return x_out


if __name__ == '__main__':
    import numpy as np
    # df = FIELD_NUMS
    # afm_model = AttentionalFactorizationMachineModel(df,10,32,(0.3,0.3))
    #
    # print(afm_model)
    # from_model = torch.load(".././data/wdwd_best_auc_pre_80.pkl",map_location=torch.device('cpu'))
    # print(from_model)
    # afm_model = load_trained_embedding(from_model=from_model,to_model=afm_model)

        # print(name, '      ', param.size())




    fd = [3, 4]
    embed_dim = 32
    f_n = np.array([[1, 3], [0, 2], [0, 1], [1, 3]])
    f_n = torch.from_numpy(f_n).long()

    model = HirAFM(fd, embed_dim, 8)
    label = torch.randint(0, 2, (4, 1)).float()
    print(label)
    loss = torch.nn.BCEWithLogitsLoss()

    pred = model(f_n).view(-1)

    print(pred.size())
    losses = loss(pred, label.view(-1))
    print(losses)
    # _,indexs = torch.max(pred,dim=1)
    print(pred)
    # print(indexs)
