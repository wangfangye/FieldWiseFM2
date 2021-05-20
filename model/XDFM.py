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
from layer import CompressedInteractionNetwork, FeaturesEmbedding, FeaturesLinear, \
    MultiLayerPerceptron, FeaturesEmbeddingWithGlobalIn



class ExtremeDeepFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of xDeepFM.

    Reference:
        J Lian, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems, 2018.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims=(400,400,400), dropout=0.5, cross_layer_sizes=(100,100), split_half=False):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)

        self.linear = FeaturesLinear(field_dims)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.cin(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return x
        # return torch.sigmoid(x.squeeze(1))

class GFRLXDFM(torch.nn.Module):
    """
    A pytorch implementation of xDeepFM.

    Reference:
        J Lian, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems, 2018.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims=(400,400,400), dropout=0.5, cross_layer_sizes=(200,200,200), split_half=True):
        super().__init__()

        self.embedding = FeaturesEmbeddingWithGlobalIn(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim

        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

        # 修改cross  CIN模块
        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)

        self.linear = FeaturesLinear(field_dims)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        # 三部分组成
        x = self.linear(x) + self.cin(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return x

class GFRLCIN(torch.nn.Module):
    """
    A pytorch implementation of xDeepFM.

    Reference:
        J Lian, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems, 2018.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims=(400,400,400), dropout=0.5, cross_layer_sizes=(100,100,100), split_half=True):
        super().__init__()
        self.embedding = FeaturesEmbeddingWithGlobalIn(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        # self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)

        self.linear = FeaturesLinear(field_dims)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.cin(embed_x)
        return x

class CIN(torch.nn.Module):
    """
    A pytorch implementation of xDeepFM.

    Reference:
        J Lian, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems, 2018.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims=(400,400,400), dropout=0.5, cross_layer_sizes=(100,100,100), split_half=True):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        # self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)

        self.linear = FeaturesLinear(field_dims)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.cin(embed_x)
        return x




if __name__ == '__main__':
    import numpy as np
    fd = [3, 4]
    embed_dim = 32
    f_n = np.array([[1, 3], [0, 2], [0, 1], [1, 3]])
    f_n = torch.from_numpy(f_n).long()

    model = ExtremeDeepFactorizationMachineModel(fd, embed_dim)
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