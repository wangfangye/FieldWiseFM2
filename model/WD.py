#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Author:wangfy
@project:DLRec
@Time:2020/4/17 8:58 下午
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import FeaturesLinear, MultiLayerPerceptron, FeaturesEmbedding,FeaturesEmbeddingWithGate,FeaturesEmbeddingWithGlobalIn
from model.MyLayers import GLUActivation1D


class WideAndDeepModel(torch.nn.Module):
    """
    A pytorch implementation of wide and deep learning.
    WD 的 工业功能比较强。 效果不错，因为参数少，运行快速。
    Reference:
        HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.
    """
    def __init__(self,field_dims,embed_dim, mlp_dims,dropout=0.5,field_len=10):
        super(WideAndDeepModel, self).__init__()
        #     wide 部分： linear + FM
        self.linear = FeaturesLinear(field_dims)

        self.embedding = FeaturesEmbedding(field_dims,embed_dim)

        self.embed_output_dim = field_len * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim,mlp_dims,dropout=dropout)


        # self.last_linear = nn.Linear(2,1)


    def forward(self,x):
        """
        :param x: [B,n_f]
        :return: [B,1]
        """
        embed_x = self.embedding(x)
        # wide  +  DEEP, 和DeepFM比，没有FM部分，否则就是相同的
        # TODO 改进，通过一个Linear
        x = self.linear(x) + self.mlp(embed_x.view(x.size(0),-1))
        # x = self.last_linear(torch.cat([self.linear(x),self.mlp(embed_x.view(x.size(0),-1))],dim=-1))
        return x

    def get_l2_loss(self, lambdas=1e-5):
        regularization_loss = 0
        for parameter in self.embedding.parameters():
            regularization_loss += torch.sum(parameter.pow(2))
        for parameter in self.mlp.parameters():
            regularization_loss += torch.sum(parameter.pow(2))
        return lambdas*regularization_loss


class GELWDL(torch.nn.Module):
    """
    A pytorch implementation of wide and deep learning.
    WD 的 工业功能比较强。 效果不错，因为参数少，运行快速。
    Reference:
        HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.
    """
    def __init__(self,field_dims,embed_dim, mlp_dims,dropout=0.5):
        super(GELWDL, self).__init__()
        #     wide 部分： linear + FM
        self.linear = FeaturesLinear(field_dims)

        self.embedding = FeaturesEmbeddingWithGlobalIn(field_dims,embed_dim)

        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim,mlp_dims,dropout=dropout)


        # self.last_linear = nn.Linear(2,1)


    def forward(self,x):
        """
        :param x: [B,n_f]
        :return: [B,1]
        """
        embed_x = self.embedding(x)
        # wide  +  DEEP, 和DeepFM比，没有FM部分，否则就是相同的
        # TODO 改进，通过一个Linear
        x = self.linear(x) + self.mlp(embed_x.view(x.size(0),-1))
        # x = self.last_linear(torch.cat([self.linear(x),self.mlp(embed_x.view(x.size(0),-1))],dim=-1))
        return x

class WDLGate(torch.nn.Module):
    """
    A pytorch implementation of wide and deep learning.
    WD 的 工业功能比较强。 效果不错，因为参数少，运行快速。
    Reference:
        HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.
    """
    def __init__(self,field_dims,embed_dim, mlp_dims=(400,400,400),glunum=1,dropout=0.1):
        super(WDLGate, self).__init__()
        #     wide 部分： linear + FM
        self.linear = FeaturesLinear(field_dims)

        self.embedding_gate = FeaturesEmbeddingWithGate(field_dims,embed_dim,glu_num=glunum)

        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim,mlp_dims,dropout=dropout)


    def forward(self,x):
        """
        :param x: [B,n_f]
        :return: [B,1]
        """
        embed_x = self.embedding_gate(x)
        # wide  +  DEEP, 和DeepFM比，没有FM部分，否则就是相同的
        # TODO 改进，通过一个Linear
        x = self.linear(x) + self.mlp(embed_x.view(x.size(0),-1))
        # x = self.last_linear(torch.cat([self.linear(x),self.mlp(embed_x.view(x.size(0),-1))],dim=-1))
        return x



class WDLInterGate(nn.Module):
    def __init__(self):
        super(WDLInterGate, self).__init__()
        pass

    def forward(self,x):
        pass


if __name__ == '__main__':
    import numpy as np
    fd = [3, 4]
    embed_dim = 8
    f_n = np.array([[1, 3], [0, 2], [0, 1], [1, 3]])
    f_n = torch.from_numpy(f_n).long()
    model = WDLGate(fd,embed_dim,[64,16],3)
    label = torch.randint(0,2,(4,1)).float()
    print(label)
    loss = nn.BCEWithLogitsLoss()
    pred = model(f_n)
    print(pred.size())