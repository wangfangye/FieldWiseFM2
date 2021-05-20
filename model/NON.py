#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Author:wangfy
@project:PNNConvModel
@Time:2021/3/26 2:04 下午
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import FeaturesLinear, FeaturesEmbedding, FeaturesEmbeddingWithGlobalIn, \
    GFRL,MultiLayerPerceptron

class NON(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_layers=(400, 400, 400), dropouts=(0.5, 0.5)):
        super(NON, self).__init__()

        # 线性部分
        self.linear = FeaturesLinear(field_dims)
        # embedding
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # 特征转化
        self.non_enhance = NON_inter(len(field_dims), embed_dim)

        self.dnn = MultiLayerPerceptron(embed_dim*len(field_dims), mlp_layers,dropout=dropouts[0], output_layer=False)

        self.atten_embedding = torch.nn.Linear(embed_dim, 32)
        self.atten_output_dim = len(field_dims) * 32

        self.self_attns = torch.nn.ModuleList([
                torch.nn.MultiheadAttention(32, 4, dropout=dropouts[0]) for _ in range(3)
            ])

        self.input_dim = 400 + self.atten_output_dim + 1
        self.mlp = MultiLayerPerceptron(self.input_dim, embed_dims=(64,32), dropout=dropouts[1])

    def forward(self,x):

        x_emb = self.embedding(x)
        # 使用 non中的方法对模型进行强化
        x_emb = self.non_enhance(x_emb)
        x_fc = self.linear(x)
        x_dnn = self.dnn(x_emb.view(x.size(0),-1))
        # print(x_dnn.size())

        atten_x = self.atten_embedding(x_emb)

        cross_term = atten_x.transpose(0, 1)

        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)

        cross_term = cross_term.transpose(0, 1)



        cross_term = F.relu(cross_term).contiguous().view(-1, self.atten_output_dim)
        x_final = torch.cat([x_fc,
                             cross_term.view(x.size(0),-1),
                             x_dnn.view(x.size(0),-1)],
                             dim=1)

        x_out = self.mlp(x_final)
        return x_out



class NON2(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_layers=(400, 400, 400), dropouts=(0.5, 0.5)):
        super(NON2, self).__init__()

        self.linear = FeaturesLinear(field_dims)  # 线性部分
        # self.embedding = FeaturesEmbedding(field_dims, embed_dim)  # embedding
        self.embedding = FeaturesEmbeddingWithGlobalIn(field_dims, embed_dim)  # embedding

        self.dnn = MultiLayerPerceptron(embed_dim*len(field_dims), mlp_layers,dropout=dropouts[0], output_layer=False)

        self.atten_embedding = torch.nn.Linear(embed_dim, 32)
        self.atten_output_dim = len(field_dims) * 32

        self.self_attns = torch.nn.ModuleList([
                torch.nn.MultiheadAttention(32, 4, dropout=dropouts[0]) for _ in range(3)
            ])

        self.input_dim = 400 + self.atten_output_dim + 1
        self.mlp = MultiLayerPerceptron(self.input_dim, embed_dims=(64,32), dropout=dropouts[1])

    def forward(self,x):
        x_emb = self.embedding(x)
        x_fc = self.linear(x)
        x_dnn = self.dnn(x_emb.view(x.size(0),-1))
        # print(x_dnn.size())

        atten_x = self.atten_embedding(x_emb)

        cross_term = atten_x.transpose(0, 1)

        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)

        cross_term = cross_term.transpose(0, 1)



        cross_term = F.relu(cross_term).contiguous().view(-1, self.atten_output_dim)
        x_final = torch.cat([x_fc,
                             cross_term.view(x.size(0),-1),
                             x_dnn.view(x.size(0),-1)],
                             dim=1)

        x_out = self.mlp(x_final)
        return x_out


