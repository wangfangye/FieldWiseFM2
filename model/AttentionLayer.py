#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Author:wangfy
@project:PNNConvModel
@Time:2020/11/13 9:02 下午
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



class Attention(nn.Module):
    """
    several score types like dot,general and concat
    """
    def __init__(self, method='dot', hidden_size=None):
        super(Attention, self).__init__()
        self.method = method
        if self.method != 'dot':
            self.hidden_size = hidden_size
            if self.method == 'general':
                self.W = nn.Linear(hidden_size, hidden_size)
            elif self.method == 'concat':
                self.W = nn.Linear(self.hidden_size * 2, hidden_size)
                # Parameter是可以训练的
                self.v = nn.Parameter(torch.rand(1, hidden_size))  # 此处定义为Linear也可以
                nn.init.xavier_normal_(self.v.data)

    def forward(self, query, key, value, mask=None, dropout=0):
        if self.method == 'general':
            scores = self.general(query, key)
        elif self.method == 'concat':
            scores = self.concat(query, key)
        else:
            scores = self.dot(query, key)

        # normalize
        # scores = scores / math.sqrt(query.size(-1))
        # mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # softmax
        p_attn = F.softmax(scores, dim=-1)
        # dropout
        if not dropout:
            p_attn = F.dropout(p_attn, dropout)

        return torch.matmul(p_attn, value), p_attn

    def dot(self, query, key):
        scores = torch.matmul(query, key.transpose(-2, -1))
        return scores

    def general(self, query, key):
        scores = torch.matmul(self.W(query), key.transpose(-2, -1))
        return scores

    def concat(self, query, key):
        scores = torch.cat((query.expand(-1, key.size(1), -1), key), dim=2)
        scores = self.W(scores)
        scores = F.tanh(scores)
        scores = torch.matmul(scores, self.v.t()).transpose(-2, -1)
        return scores

class GeneralAttention(nn.Module):
    def __init__(self,embed_dim, conv_size = 0):
        """
        :param embed_dim:
        """
        super(GeneralAttention, self).__init__()
        if conv_size == 0:
            # 如果不存在
            conv_size = embed_dim
        # self.attention = torch.nn.Linear(embed_dim, embed_dim)
        self.attention = torch.nn.Linear(embed_dim, conv_size)
        self.projection = torch.nn.Linear(conv_size, 1)
        # self.projection = torch.nn.Linear(embed_dim, 1)

    def forward(self,key,dim=1):

        attn_scores = F.relu(self.attention(key)) # 先改变维度，
        # projecction相当于 新建 一个 nn.paramater(torch.zeros((att_size,)))
        attn_scores = F.softmax(self.projection(attn_scores), dim=dim) # B,n_f-1, 1
        # 和权重相乘
        attn_output = torch.sum(attn_scores * key, dim=dim)  # B,e
        return attn_output,attn_scores


#
class SelfAttention(nn.Module):
    '''
    Scaled_Dot_Product_Attention
    输入q，k，v 完成 self.attention
    '''
    def __init__(self,):
        super(SelfAttention, self).__init__()
        # Do we need to apply transformer

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))

        if scale:
            attention = attention * math.sqrt(Q.size()[2])
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)

        return context, attention



if __name__ == '__main__':
    inp = torch.randn((21,10,16))
    # general_att = GeneralAttention(16)
    # out,attscore = general_att(inp)
    # print(attscore.size())
    # print(out.size())
    print(inp.size()[2])
    # self_att = SelfAttention()
    # out = self_att(inp,inp,inp)
    # print(out.size())


