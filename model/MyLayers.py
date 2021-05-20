#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Author:wangfy
@project:DLRec
@Time:2020/6/18 7:30 下午
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class DINActivation(nn.Module):
    """
    DIN 中的activation weight
    输入两个向量，通过out product，然后经过线性变换得到一个常量，作为权重，功能等价于attention

    """

    def __init__(self, emb_dim):
        super(DINActivation, self).__init__()
        self.linear = torch.nn.Linear(emb_dim * 3, 1)
        self.prelu = torch.nn.PReLU()

    def forward(self, x1, x2):
        """
        :param x1: B,1,e
        :param x2: B,1,e
        :return:
        """
        x_out = torch.einsum("bij,bie->bje", x1, x2)
        x_out = torch.sum(x_out, dim=1, keepdim=True)
        x_out = torch.cat([x1, x_out, x2], dim=2)
        # dice 激活函数
        # x_out = self.prelu(x_out)
        x_out = self.prelu(x_out)
        x_out = self.linear(x_out)
        return x_out


# https://kexue.fm/archives/6671
class glu_activation(nn.Module):
    def __init__(self, embed_dim, conv_size=64):
        """
        :param embed_dim:
        :param conv_size: 需要和 embed_dim 相似
        """
        super(glu_activation, self).__init__()

        # 卷积核(2,embed_dim),步长(2,1)
        self.conv1 = nn.Conv2d(1, conv_size, (2, embed_dim), (2, 1))
        # TODO 这个卷积需不需要？？？？
        self.conv1_11 = nn.Conv2d(conv_size, embed_dim, (1, 1), (1, 1))

        self.conv2 = nn.Conv2d(1, conv_size, (2, embed_dim), (2, 1))
        self.conv2_11 = nn.Conv2d(conv_size, embed_dim, (1, 1), (1, 1))

        self.acti = torch.tanh()

    def forward(self, x, x_inter):
        """
        :param x:  B, 1, f*(f-1)/2, emb_size
            x_inter:  B, 1, f*(f-1), emb_size
        :return: B, f*(f-1)/2, emb_size
        """
        #
        x_inter = x_inter.transpose(1, 3)

        # 信息通过的概率，但是不能使用tanh，因为这不是概率
        # TODO： 修稿sigmoid,通过，不知道选择tanh的效果如何
        # x1 = self.acti(self.conv1_11(self.conv1(x)))
        x1 = torch.sigmoid(self.conv1_11(self.conv1(x)))

        x2 = self.conv2
        _11(self.conv2(x))
        #  [20, 64, 45, 1] 生成一组新的表达式
        x = x_inter * (torch.tensor(1.0) - x1) + x2 * x1

        return x.squeeze().transpose(1, 2)


class GLUActivation1DNo11(nn.Module):
    """
    不含有一维卷积
    """
    def __init__(self, embed_dim, conv_size):
        super().__init__()
        # TODO 卷积核(1,embed_dim),步长(1,1) 信息的处理处是否保证了可以学到概率 然后特征进行强化
        self.conv1 = nn.Conv2d(1, embed_dim, (1, embed_dim), (1, 1))
        # 一维卷积，这里需不需要
        # self.conv1_11 = nn.Conv2d(conv_size, embed_dim, (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(1, embed_dim, (1, embed_dim), (1, 1))
        #  生维度
        self.conv2_11 = nn.Conv2d(conv_size, embed_dim, (1, 1), (1, 1))
        #  选择激活函数，s：sigmoid （0，1) 按照概率的设定而言，应该使用sigmoid

    def forward(self, x_inter):
        """
        :param x:  B, f*(f-1)/2, emb_size
            x_inter:  B, f*(f-1), emb_size
        :return: B, f*(f-1)/2, emb_size
        （1） 只有x_inter
        （2） 同时使用x_inter和x, glu_activation
        """
        x_inter = x_inter.unsqueeze(1)
        x1 = self.conv1(x_inter)
        #  计算概率
        x1 = torch.sigmoid(x1)

        x2 = self.conv2_11(self.conv2(x_inter))
        x_inter = x_inter.transpose(1, 3)
        x_out = x_inter * (torch.tensor(1.0) - x1) + x2 * x1

        return x_out.squeeze().transpose(1, 2)

class GLUActLinear(nn.Module):
    """
    含有一维卷积
    空间复杂度: 1 *（2*embed_dim + 1）* embed_dim * 1 + 2*embed_dim * embed_dim*(1+1)
    """

    def __init__(self, embed_dim, conv_size):
        super(GLUActLinear, self).__init__()
        # 可以使用全连接层
        #
        # conv_size = embed_dim // 2
        self.conv1 = nn.Conv2d(1, conv_size, (1, embed_dim))
        # 一维卷积，这里需不需要
        self.conv1_11 = nn.Conv2d(conv_size, embed_dim, (1, 1))
        self.conv2 = nn.Conv2d(1, conv_size, (1, embed_dim))
        #  生维度
        self.conv2_11 = nn.Conv2d(conv_size, embed_dim, (1, 1))
        #  选择激活函数，s：sigmoid （0，1），t:tanh （-1，1）

    def forward(self, x_inter):
        """
        :param x:  B, f*(f-1)/2, emb_size
            x_inter:  B, f*(f-1), emb_size
        :return: B, f*(f-1)/2, emb_size
        （1） 只有x_inter
        （2） 同时使用x_inter和x, glu_activation
        """
        x_inter = x_inter.unsqueeze(1) # B,1,x_inter,e
        x1 = self.conv1_11(self.conv1(x_inter)) # B,e,x_inter,1
        # print("inner x1_size",x1.size())
        # 概率　只有这里使用激活函数
        x1 = torch.sigmoid(x1) # B,e,x_inter,1

        x2 = self.conv2_11(self.conv2(x_inter)) # B,e,x_inter,1
        # Gate function
        x_inter = x_inter.transpose(1, 3) # B,e,x_inter,1
        x_out = x_inter * (torch.tensor(1.0) - x1) + x2 * x1
        return x_out.squeeze().transpose(1, 2) # B,x_inter,e
        # return x_out


class GLUActivation1D(nn.Module):
    """
    使用线性模型代替这一部分
    """
    def __init__(self,emb_dim,conv_size=10):
        super(GLUActivation1D, self).__init__()
        self.conv_size = emb_dim*2
        self.linear1 = nn.Linear(emb_dim,self.conv_size)
        self.linear2 = nn.Linear(emb_dim,self.conv_size)
        self.linear1_2 = nn.Linear(self.conv_size,emb_dim)
        self.linear2_2 = nn.Linear(self.conv_size,emb_dim)


    def forward(self,x_inter):

        x1 = self.linear1(x_inter)
        x1 = torch.sigmoid(self.linear1_2(x1))
        x2 = self.linear2(x_inter)
        x2 = self.linear2_2(x2)
        x_out = x_inter * (torch.tensor(1.0) - x1) + x2 * x1
        return x_out

class GlobalGluIn2(nn.Module):
    """
    通过全连接的方式转化
    """
    def __init__(self,field_length,embed_dim):
        super(GlobalGluIn2, self).__init__()
        # （1） 将全局的向量的维度变成和
        self.input_dim = field_length * embed_dim
        self.trans = nn.Linear(embed_dim,embed_dim)
        self.trans2 = nn.Linear(embed_dim, embed_dim)
        #
        self.lin_trans = nn.Linear(self.input_dim,embed_dim)
        self.lin_trans2 = nn.Linear(self.input_dim,embed_dim)

        self.__init_weight()

    def __init_weight(self):
        nn.init.xavier_normal_(self.trans.weight)
        nn.init.xavier_normal_(self.trans2.weight)
        nn.init.xavier_normal_(self.lin_trans.weight)
        nn.init.xavier_normal_(self.lin_trans2.weight)

    def forward(self,x):
        # 新加的特征
        # 这里还要用激活函数吗？？？
        # 第一组
        x_fc = x.view(-1, 1, self.input_dim)  # [B,1,f*e]

        x_trans2 = self.trans2(x)
        x_glo2 = self.lin_trans2(x_fc)  # [B,1,e]

        x_out = x_trans2 * x_glo2
        # 通过x_pro 和 x 计算概率，或者通过这种方式强化
        # （1） v_i*(W v_l) 先通过全连接层 将全量特征的维度转化到 v_i 相同的维度，然后通过内积或者哈德玛积的方式
        # 最后需要接上一个sigmoid，表示概率；同时另外一组的特征，不使用sigmoid,通过概率的方式通过。

        # 第二组 学习概率
        x_trans1 = self.trans(x)
        x_pro = torch.sigmoid(x_trans1 * self.lin_trans(x_fc))
        # 没有sigmoid
        # x_out = x * self.lin_trans2(x_fc)
        x_out = x * (torch.tensor(1.0) - x_pro) + x_out * x_pro
        return x_out

    # def forward(self,x):
    #     # 新加的特征
    #     # 这里还要用激活函数吗？？？
    #     # 第一组 只使用一个Global encoding
    #     x_fc = x.view(-1, 1, self.input_dim)  # [B,1,f*e]
    #
    #     x_trans2 = self.trans2(x)
    #     x_pro = self.lin_trans(x_fc)  # [B,1,e]
    #
    #     x_out = x_trans2 * x_pro
    #     # 通过x_pro 和 x 计算概率，或者通过这种方式强化
    #     # （1） v_i*(W v_l) 先通过全连接层 将全量特征的维度转化到 v_i 相同的维度，然后通过内积或者哈德玛积的方式
    #     # 最后需要接上一个sigmoid，表示概率；同时另外一组的特征，不使用sigmoid,通过概率的方式通过。
    #
    #     # 第二组
    #     x_trans = self.trans(x)
    #     # 1.1 学习概率 这里直接相乘
    #     # x_pro = torch.sigmoid(x_pro * x)
    #     # 这里直接使用hadamard 积 pairwise product
    #     x_pro = torch.sigmoid(x_pro * x_trans)
    #     # 没有sigmoid
    #     # x_out = x * self.lin_trans2(x_fc)
    #     x_out = x * (torch.tensor(1.0) - x_pro) + x_out * x_pro
    #     return x_out



class FLB(nn.Module):
    # GFEL的FLB模块
    def __init__(self, field_length,embed_dim):
        super(FLB,self).__init__()

        self.input_dim = field_length * embed_dim
        # local alignment 这里不对 使用向量矩阵相乘的方式
        self.local = nn.Linear(embed_dim, embed_dim)
        # global alignment
        self.glo_trans = nn.Linear(self.input_dim,embed_dim)



    def forward(self,x_emb):
        x_trans = self.local(x_emb)
        x_fc = x_emb.view(-1, 1, self.input_dim)
        x_glo = self.glo_trans(x_fc)
        # hadamard product
        x_out = x_trans * x_glo
        return x_out

class GlobalGluIn(nn.Module):
    """
    这个是1230改正的，最终版本的GFELs
    """

    def __init__(self, field_length, embed_dim, flb="flu"):
        super(GlobalGluIn, self).__init__()
        # （1） 将全局的向量的维度变成和

        self.input_dim = field_length * embed_dim
        if flb == "flu":
            self.flu1 = FLU(field_length, embed_dim)
            self.flu2 = FLU(field_length, embed_dim)

        elif flb == "senet":
            # 这一部分先不测试。
            self.flu1 = SenetLayer(field_length)
            self.flu2 = SenetLayer(field_length)

        # 如果有需要，我们测试不同的sigmoid的影响
        # self.acti = nn.Sigmoid()


    def forward(self, x):
        x_out = self.flu1(x)
        # 第二组学习概率
        #
        x_pro = torch.sigmoid(self.flu2(x))
        # 使用门机制
        x_out = x * (torch.tensor(1.0) - x_pro) + x_out * x_pro
        return x_out

class AttGFRL(nn.Module):
    def __init__(self, field_length, embed_dim,flu="flu"):
        pass



    def forward(self,x):
        return x


class GFRL(nn.Module):
    # Gated feature refine layer
    def __init__(self, field_length, embed_dim, flu="flu"):
        """
        :param field_length: field_length
        :param embed_dim: embedding dimension
        :param flu: choose the type, default "flu"
        """
        super(GFRL, self).__init__()
        # self.input_dim = field_length * embed_dim
        if flu == "flu":
            self.flu1 = FLU(field_length, embed_dim)
            self.flu2 = FLU(field_length, embed_dim)
        elif flu == "senet":
            # fibinet中的senet结构
            self.flu1 = SenetLayer(field_length)
            self.flu2 = SenetLayer(field_length)

    def forward(self,x):
        x_out = self.flu1(x)
        # 第二组学习概率
        x_pro = torch.sigmoid(self.flu2(x))
        # Gate mechanism
        x_out = x * (torch.tensor(1.0) - x_pro) + x_out * x_pro
        return x_out


class FLU(nn.Module):
    # GFRL的FLU模块
    def __init__(self, field_length, embed_dim):
        super(FLU,self).__init__()
        self.input_dim = field_length * embed_dim
        # local alignment 这里不对 使用向量矩阵相乘的方式
        self.local_w = nn.Parameter(torch.randn(field_length, embed_dim, embed_dim))
        # 每个field 提供一个
        self.local_b = nn.Parameter(torch.randn(field_length, 1, embed_dim))

        # 在local encoder中对于每个field的embedding都需要一个linear，所以我们使用上述的方式，通过矩阵乘法的方式加快计算的速度
        # self.local = nn.Linear(embed_dim, embed_dim)
        # global encoder
        self.glo_trans = nn.Linear(self.input_dim, embed_dim)

        nn.init.xavier_uniform_(self.local_w.data)
        nn.init.xavier_uniform_(self.local_b.data)
        nn.init.xavier_uniform_(self.glo_trans.weight)

    def forward(self,x_emb):
        # local 使用矩阵计算
        # x_emb [b,f,d] self.local_w [f,d,d]
        # x_local [f,b,d]
        # 矩阵相乘 具体的计算过程可以看ONN的论文
        x_local = torch.matmul(x_emb.permute(1,0,2),self.local_w) + self.local_b
        x_local = x_local.permute(1,0,2)

        x_fc = x_emb.view(-1, 1, self.input_dim)
        x_glo = self.glo_trans(x_fc)
        # hadamard product
        x_out = x_local * x_glo
        return x_out

class NON_inter(nn.Module):
    # NON的方法
    def __init__(self, field_length, embed_dim):
        super(NON_inter, self).__init__()
        self.input_dim = field_length * embed_dim
        # local alignment 这里不对 使用向量矩阵相乘的方式
        self.local_w = nn.Parameter(torch.randn(field_length, embed_dim, embed_dim))
        # 每个field 提供一个
        self.local_b = nn.Parameter(torch.randn(field_length, 1, embed_dim))

        # 在local encoder中对于每个field的embedding都需要一个linear，所以我们使用上述的方式，通过矩阵乘法的方式加快计算的速度
        # self.local = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.local_w.data)
        nn.init.xavier_uniform_(self.local_b.data)

    def forward(self,x_emb):
        x_local = torch.matmul(x_emb.permute(1, 0, 2), self.local_w) + self.local_b
        x_local = x_local.permute(1, 0, 2)
        # 用什么方法连接
        # 使用element-wise product 和我们一样
        # 这里和FLU不同的是NON和原始的特征进行融合，而FLU和global的信息进行融合。
        x_local = x_local * x
        return x_local


class GlobalGluOut(nn.Module):
    """
    field的embedding与所有的特征进行out product的方式
    """
    def __init__(self,field_length,embed_dim):
        super(GlobalGluOut, self).__init__()
        # （1） 将全局的向量的维度变成和
        self.input_dim = field_length * embed_dim
        kernel_shape = embed_dim, field_length, self.input_dim
        self.kernel1 = torch.nn.Parameter(torch.zeros(kernel_shape))
        self.kernel2 = torch.nn.Parameter(torch.zeros(kernel_shape))

        torch.nn.init.xavier_uniform_(self.kernel1.data)
        torch.nn.init.xavier_uniform_(self.kernel2.data)


    def forward(self,x):
        """
        :param x:
        :return:
        """
        b,n_f,e = x.size()
        x_fc = x.view(b,1,self.input_dim).expand(b,n_f,self.input_dim) #[B,n_f,f*e]
        kp1 = torch.sum(x_fc.unsqueeze(1) * self.kernel1, dim=-1).permute(0, 2, 1)  # b,num_ix,e,e
        # #b,num_ix,e,e
        x_pro = torch.sigmoid(torch.sum(kp1 * x, -1))

        kp2 = torch.sum(x_fc.unsqueeze(1) * self.kernel2, dim=-1).permute(0, 2, 1)  # b,num_ix,e,e
        print(kp2.size())
        # #b,num_ix,e,e
        x_out = kp2 * x
        print("x_out",x_out.size())
        x_out = x * (torch.tensor(1.0) - x_pro) + x_out * x_pro
        return x_out




class GLUActivation1D2(nn.Module):
    """
    含有一维卷积
    参数分析：
    """

    def __init__(self, embed_dim, conv_size, acti="s"):
        super(GLUActivation1D2, self).__init__()
        # 卷积核(2,embed_dim),步长(2,1)
        self.conv1 = nn.Conv2d(1, conv_size, (1, embed_dim), (1, 1))
        # 一维卷积，
        self.conv1_11 = nn.Conv2d(conv_size, embed_dim, (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(1, conv_size, (1, embed_dim), (1, 1))
        #  生维度
        self.conv2_11 = nn.Conv2d(conv_size, embed_dim, (1, 1), (1, 1))
        #  选择激活函数，s：sigmoid （0，1），t:tanh （-1，1）
        if acti == "s":
            self.acti = torch.nn.Sigmoid()


        elif acti == "t":
            self.acti = torch.nn.Tanh()

    def forward(self, x_inter):
        """
        :param x:  B, f*(f-1)/2, emb_size
            x_inter:  B, f*(f-1), emb_size
        :return: B, f*(f-1)/2, emb_size
        （1） 只有x_inter
        （2） 同时使用x_inter和x, glu_activation
        """
        x_inter = x_inter.unsqueeze(1)
        x1 = self.conv1_11(self.conv1(x_inter))
        # x1 = self.conv1(x_inter)  TODO: 需不需要一维的卷积？？？？？？？
        # print("inner x1_size",x1.size())
        # 概率　只有这里使用激活函数

        # x1 = torch.sigmoid(x1)
        x1 = self.acti(x1)
        x2 = self.conv2_11(self.conv2(x_inter))

        # Gate function
        x_inter = x_inter.transpose(1, 3)
        x_out = x_inter * (torch.tensor(1.0) - x1) + x2 * x1

        # print("x_out",x_out.size())
        # x_out = x_out.transpose(1, 3).squeeze(1)
        return x_out.squeeze().transpose(1, 2)
        # return x_out


class GLUActivation(nn.Module):
    """

    """

    def __init__(self, embed_dim, conv_size):
        super(GLUActivation, self).__init__()
        # 卷积核(2,embed_dim),步长(2,1)
        self.conv1 = nn.Conv2d(1, embed_dim, (1, embed_dim), (1, 1))
        # 一维卷积，
        # self.conv1_11 = nn.Conv2d(conv_size, embed_dim, (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(1, embed_dim, (1, embed_dim), (1, 1))
        #  生维度
        # self.conv2_11 = nn.Conv2d(conv_size, embed_dim, (1, 1), (1, 1))

    def forward(self, x_inter):
        """
        :param x:  B, 1, f*(f-1)/2, emb_size
            x_inter:  B, 1, f*(f-1), emb_size
        :return: B, f*(f-1)/2, emb_size
        （1） 只有x_inter
        （2） 同时使用x_inter和x, glu_activation
        """
        x_inter = x_inter.unsqueeze(1)
        x1 = self.conv1(x_inter)
        # 概率
        x1 = torch.sigmoid(x1)

        x2 = self.conv2(x_inter)
        # Gate function
        x_inter = x_inter.transpose(1, 3)
        x_out = x_inter * (torch.tensor(1.0) - x1) + x2 * x1

        x_out = x_out.transpose(1, 3)
        return x_out


class GenerateConv(nn.Module):
    """
    实现模型的conv部分
    """

    def forward(self, x_emb):
        """
        :param x_emb: b,f,e
        :return: x_emb
                 x_inter
        """
        input_size = x_emb.size()
        num_fields = input_size[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x_emb[:, row], x_emb[:, col]
        # TODO 这里有两个
        x_inter = p * q  # B,0.5*(nf-1)*nf, emb
        return x_inter

        # # 重新两两组合
        # rows = list()
        # # 相乘的方式
        # rows2 = list()
        # for i in range(num_fields - 1):
        #     for j in range(i + 1, num_fields):
        #         rows.append(x_emb[:, i])
        #         rows.append(x_emb[:, j])
        #         rows2.append(x_emb[:, i] * x_emb[:, j])

        # 卷积，好处在于可以共享参数，通过共享参数
        # （1）不使用简单的product来进行计算，而是使用卷积的参数共享方式，
        # （使用）
        # x_emb = torch.stack(rows, dim=1).unsqueeze(1)  # B, 1, f*(f-1), emb_size
        # 相乘的方式,相当于dot product 的方式，
        # x_inter = torch.stack(rows2, dim=1).unsqueeze(1)  # B, 1, f*(f-1)/2, emb_size


class GenerateConv2(nn.Module):
    """
    实现模型的conv部分
    """

    def forward(self, x_emb):
        """
        :param x_emb: b,f,e
        :return: x_emb
                 x_inter
        """
        input_size = x_emb.size()
        num_fields = input_size[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x_emb[:, row], x_emb[:, col]
        # TODO 这里有两个
        x_inter = p * q  # B,n_f-1, emb
        return x_inter.unsqueeze(1)


class InterConv(nn.Module):
    """
    实现模型的conv部分，完整的部分
    """

    def __init__(self, embed_dim, conv_size):
        super(InterConv, self).__init__()
        # 二维卷积实现一位卷积操作。 B,1,fs,e
        self.conv_size = conv_size
        self.conv_inter = nn.Conv2d(1, conv_size, (2, embed_dim), (2, 1))

    def forward(self, x_emb):
        """
        :param x_emb: b,f,e
        :return:
        """
        input_size = x_emb.size()
        num_fields = input_size[1]
        rows = list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                rows.append(x_emb[:, i])
                rows.append(x_emb[:, j])

        x_emb = torch.stack(rows, dim=1).unsqueeze(1)  # B, 1, f*(f-1)/2, emb_size

        # 卷积，好处在于可以共享参数，通过共享参数
        x_emb = self.conv_inter(x_emb)  # b, conv_size，f*(f-1)/2, 1

        # 添加了激活函数
        x_emb = F.relu(x_emb)
        # 然后通过
        x_emb = x_emb.view(input_size[0], -1, self.conv_size)  # b,f(f-1)/2,conv_size
        return x_emb

class SenetLayer(nn.Module):
    def __init__(self,num_fields, ratio=3):
        super(SenetLayer, self).__init__()
        self.temp_dim = num_fields // ratio
        self.excitation = nn.Sequential(
            nn.Linear(num_fields,self.temp_dim),
            nn.Sigmoid(),
            nn.Linear(self.temp_dim, num_fields),
            nn.Sigmoid()
        )

    def forward(self,input_x):
        """
        三步：
        (1) squeeze
        (2) Excitation
        (3) Re-weight
        :param input_x:  [B,f,e]
        :return:  [B,f,e]
        """
        # 第一步 squeeze mean的方式
        Z_mean = torch.mean(input_x,dim=2,keepdim=True).transpose(1,2)
        # 第二步 Excitation, 两个全连接层
        A_weight = self.excitation(Z_mean).transpose(1,2)
        # 第三部 multiplication
        V_embed = torch.mul(A_weight,input_x)
        return V_embed


class BiInterActionAllLayer(nn.Module):
    def __init__(self,embed_dim):
        super(BiInterActionAllLayer,self).__init__()
        # 双线性模型 Fibinet中的
        self.W = nn.Linear(embed_dim,embed_dim)

    def forward(self,input_vi,input_vj):
        """
        pij = hadamard(vi*W,vj)
        input: [B,n_f,e],n_f = f(f-1)/2
        :param x: [B,n_f,e],n_f = f(f-1)/2
        :return: [B,n_f,e],n_f = f(f-1)/2
        """
        return torch.mul(self.W(input_vi),input_vj)



def count_params(model):
    params = sum(param.numel() for param in model.parameters())
    return params


if __name__ == '__main__':
    input = torch.randn(30, 39, 10)
    FLU = FLB(39,10)
    x_out = FLU(input)
    print(count_params(FLU))
    print(x_out.size())
    # 32420 8310
    # 16040 4020




    # # 从embedding转换到适合卷积的操作
    # inter_conv = GenerateConv()
    # x_inter = inter_conv(input)
    #
    # # print("out", out.size())
    # print("x_inter 1", x_inter.size())
    #
    # # #
    # dgcnn = GLUActivation1D(32, 16)
    # out2 = dgcnn(x_inter)
    # print("DGCNN", out2.size())
    # out3 = dgcnn(x_inter)
    # print("DGCNN2", out3.size())



    #
    # print(count_params(dgcnn))
    # print("test")
    # x1 = torch.randn(10,1,64)
    # x2 = torch.randn(10,1,64)
    # # x_out = torch.einsum("bij,bie->bje", x1, x2)
    # # print(x_out.size())
    # print(x1.size())
    # act = DINActivation(64)
    # out = act(x1,x2)
    # print(out.size())
    # print(out.size())
