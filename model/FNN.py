#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Author:wangfy
@project:DLRec
@Time:2020/4/21 2:44 下午
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import FeaturesEmbedding, MultiLayerPerceptron


class FactorizationSupportedNeuralNetworkModel(torch.nn.Module):
    """
    A pytorch implementation of Neural Factorization Machine.
    就是对embedding之后的数据使用DNN，并没做其他的操作。
    相当于DeepFm的DNN部分

    想只有的NFM，将embed之后的数据，进行interaction pooling操作

    Reference:
        W Zhang, et al. Deep Learning over Multi-field Categorical Data - A Case Study on User Response Prediction, 2016.
    """
    def __init__(self,field_dims, embed_dim, mlp_layers=(256,128,64),dropout=0.5):
        super(FactorizationSupportedNeuralNetworkModel, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims,embed_dim)

        self.mlp_input_dim = len(field_dims)*embed_dim

        self.mlp = MultiLayerPerceptron(self.mlp_input_dim,mlp_layers)

    def forward(self,x):
        """
        :param x: [B,num_fields]
        :return:
        """
        x = self.embedding(x)
        x = self.mlp(x.view(-1,self.mlp_input_dim))  # [B,n_f*embed_dim] => [B,1]
        return x


if __name__ == '__main__':
    import numpy as np

    fd = [3, 4,8]
    embed_dim = 8
    f_n = np.array([[1, 3,4], [0, 2,6], [0, 1,6], [1, 3,2]])
    f_n = torch.from_numpy(f_n).long()

    model = FactorizationSupportedNeuralNetworkModel(fd, embed_dim)
    label = torch.randint(0, 2, (4, 1)).float()
    print(label)
    pred = model(f_n)
    print(pred)



