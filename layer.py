import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import os
# os.environ

from model.MyLayers import GLUActivation1D
from model.MyLayers import GlobalGluIn,SenetLayer,GFRL
from model.AttentionLayer import SelfAttention

class BasicLayer(nn.Module):
    def __init__(self):
        super(BasicLayer, self).__init__()

    def forward(self, x):
        raise NotImplemented

class FeaturesLinear(torch.nn.Module):
    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        # 代相当于全连接层的w，所以dim=1，因为输入仅仅为一个下标，所以通过embedding的方式可以获得
        # 下标所对应的参数，理论上是没有连续形的变量的，所以这是可以的
        self.fc = torch.nn.Embedding(field_dims, output_dim)

        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        # 假设输入[u_id,item_id]:[10,12],前面将他们放在了一个embedding中，所以需要通过offset的方式
        # 如果分开embedding是没有也是可以的，就不需要
        # self.offsets = np.array(
        #     (0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        nn.init.uniform_(self.fc.weight,0.0,0.0)

    def forward(self, x, weights = None):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        :return: (B,1)
        """
        if weights is not None:
            return torch.sum(self.fc(x)*weights, dim=1) + self.bias
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesLinear2(torch.nn.Module):
    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fcs = torch.nn.ModuleList([
            torch.nn.Embedding(emb_size, output_dim) for emb_size in field_dims
        ])
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        # 假设输入[u_id,item_id]:[10,12],前面将他们放在了一个embedding中，所以需要通过offset的方式
        # 如果分开embedding是没有也是可以的，就不需要
        # self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = torch.cat([self.fcs[i](x[:, i])
                       for i in range(x.size()[1])], dim=1)
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(x, dim=1) + self.bias


class FeaturesEmbedding(torch.nn.Module):
    # 理解

    def __init__(self, field_dims, embed_dim):
        # 有个限制，每个field只能有一个特征，比如说：电影的类型可能大于1个，这个时候，这样是不行的
        """
        :param field_dims: list, 每个field几个不同的维度
        :param embed_dim: 相当于K，特征维度
        """
        super().__init__()
        self.embedding = torch.nn.Embedding(field_dims, embed_dim)
        # self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        # torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        # nn.init.kaiming_uniform_(self.embedding.weight)
        # nn.init.uniform_(self.embedding.weight)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        :return: (B,nf,embed_size)
        """
        # x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class FeaturesEmbeddingWithGlobalIn(torch.nn.Module):

    """
        使用Gluin/self-attention/senet等来强化embedding
    """

    def __init__(self, field_dims, embed_dim, type="glu",field_len = 10):
        # 有个限制，每个field只能有一个特征，比如说：电影的类型可能大于1个，这个时候，这样是不行的
        """
        :param field_dims: list, 每个field几个不同的维度
        :param embed_dim: 相当于K，特征维度
        """
        super().__init__()
        self.embedding = torch.nn.Embedding(field_dims, embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        if type == "glu":
            self.enhance_layer = GFRL(field_len,embed_dim)
        # elif type == "senet":
        #     self.enhance_layer = SenetLayer(len(field_dims))
        # elif type == "flu":
        #     self.enhance_layer = FLU(len(field_dims), embed_dim)
        # elif type == "gluse":
        #     self.enhance_layer = GFRL(len(field_dims), embed_dim, flu="senet")
        # elif type == "non":
        #     self.enhance_layer = NON_inter(len(field_dims), embed_dim)
        # torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        nn.init.normal_(self.embedding.weight, std=0.01)
        # nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        :return: (B,nf,embed_size)
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.enhance_layer(self.embedding(x))

class FLU(nn.Module):
    # GFEL的FLU模块
    def __init__(self, field_length,embed_dim):
        super(FLU,self).__init__()
        self.input_dim = field_length * embed_dim
        # local alignment 这里不对 使用向量矩阵相乘的方式
        self.local_w = nn.Parameter(torch.randn(field_length,embed_dim,embed_dim))
        self.local_b = nn.Parameter(torch.randn(field_length,1,embed_dim))
        nn.init.xavier_uniform_(self.local_w.data)
        nn.init.xavier_uniform_(self.local_b.data)
        # 只使用一个linear不正确。
        self.local = nn.Linear(embed_dim, embed_dim)
        # global alignment
        self.glo_trans = nn.Linear(self.input_dim,embed_dim)


    def forward(self,x_emb):
        # local 使用矩阵计算
        # x_emb [b,f,d] self.local_w [f,d,d]
        # x_local [f,b,d]

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
        x_local = x_local * x_emb
        return x_local

class FeaturesEmbeddingWithGate(torch.nn.Module):
    """
    在Embedding之后加入了Gate，来丰富Embedding的信息
    """
    def __init__(self, field_dims, embed_dim, glu_num=2):
        # 有个限制，每个field只能有一个特征，比如说：电影的类型可能大于1个，这个时候，这样是不行的
        """
        :param field_dims: list, 每个field几个不同的维度
        :param embed_dim: 相当于K，特征维度
        """
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

        glus = list()
        for _ in range(0,glu_num):
            glus.append(GLUActivation1D(embed_dim,embed_dim*2))

        self.glu = nn.Sequential(*glus)
        # torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        nn.init.normal_(self.embedding.weight, std=0.01)
        # nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        :return: (B,nf,embed_size)
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        x = self.embedding(x)

        x = self.glu(x).contiguous()
        return x



class FieldAwareFactorizationMachine(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])

        ix = torch.stack(ix, dim=1)  ## B,1/2(nf*(nf-1)),k,然后求和
        return ix


class FactorizationMachine(torch.nn.Module):
    """
    分解机操作 fm 1/2 * (和的平方 - 平方的和)
    """

    def __init__(self,embed_dim, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum
        self.drop = nn.Dropout(p=0.3)
        # self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        # 因为x都是1，所以不需要乘以x: 和的平方 - 平方的和
        square_of_sum = torch.sum(x, dim=1) ** 2  # B，embed_dim
        sum_of_square = torch.sum(x ** 2, dim=1)  # B，embed_dim
        ix = square_of_sum - sum_of_square  # B,embed_dim
        # ix = self.bn(ix)
        ix = self.drop(ix)
        # print(ix.size())
        # 如果reduce_sum = False, 则为哪些需要在这之上进行处理的模型，比如说 NFM，
        if self.reduce_sum:
            # ix = self.drop(ix)
            ix = torch.sum(ix, dim=1, keepdim=True)

        return 0.5 * ix


class FwFMInterLayer(nn.Module):
    def __int__(self, num_fields):
        super(FwFMInterLayer, self).__int__()

        self.num_fields = num_fields
        num_inter = (num_fields * (num_fields - 1))//2
        #
        self.lr = nn.Linear(num_inter,1)

    def forward(self,x_embed):
        row, col = list(), list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                row.append(i), col.append(j)

        x_inter = torch.sum(x[:, row] * x[:, col],dim=2,keepdim=False) # [B,n*(n-1)/2]
        inter_sum = self.lr(x_inter)
        return inter_sum



class MultiLayerPerceptron(torch.nn.Module):
    """
    fc 线性层
    输入 连接之后的 [B,num_field*k]  input_dim = num_field*k
    """

    def __init__(self, input_dim, embed_dims, dropout=0.5, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim

        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        # 使用 *，
        self.mlp = torch.nn.Sequential(*layers)
        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size,num_fields*embed_dim)``
        """
        return self.mlp(x)

class GateNetBit(nn.Module):
    """
    GateNet:Gating-Enhanced Deep Network for Click-Through Rate Prediction
    Bit-wise
    """
    def __init__(self,field_length,embed_dim):
        super(GateNet, self).__init__()
        self.input_dim = field_length * embed_dim
        # local alignment 这里不对 使用向量矩阵相乘的方式
        self.local_w = nn.Parameter(torch.randn(field_length, embed_dim, embed_dim))

        nn.init.xavier_uniform_(self.local_w.data)
        nn.init.xavier_uniform_(self.local_b.data)

    def forward(self, x_emb):
        x_local = torch.matmul(x_emb.permute(1, 0, 2), self.local_w)
        x_local = x_local.permute(1, 0, 2)

        # 相当于门
        x_local = x_local * x_emb
        return x_local

class GateNetVec(nn.Module):
    """
    GateNet:Gating-Enhanced Deep Network for Click-Through Rate Prediction
    vec-wise，
    """
    def __init__(self,field_length,embed_dim):
        super(GateNet, self).__init__()
        self.input_dim = field_length * embed_dim
        # local alignment 这里不对 使用向量矩阵相乘的方式
        self.local_w = nn.Parameter(torch.randn(field_length, embed_dim, 1))
        # 每个field 提供一个
        # self.local_b = nn.Parameter(torch.randn(field_length, 1, embed_dim))

        # 在local encoder中对于每个field的embedding都需要一个linear，所以我们使用上述的方式，通过矩阵乘法的方式加快计算的速度
        # self.local = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.local_w.data)
        nn.init.xavier_uniform_(self.local_b.data)

    def forward(self, x_emb):
        """

        :param x_emb:  [B,F,E]
        :return:
        """
        # 这里获得一个 向量  [F,B,E] * [F,E，1] = [F,B,1]
        x_local = torch.matmul(x_emb.permute(1, 0, 2), self.local_w)

        # [B,F,1]
        x_local = x_local.permute(1, 0, 2)
        x_local = x_local * x_emb
        return x_local



class SenetLayerAll(nn.Module):
    def __init__(self, num_fields, ratio=3):
        super(SenetLayerAll, self).__init__()
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

class MyIFMLayer(nn.Module):
    def __init__(self,field_dims,embed_dim,type_c="con"):
        super(MyIFMLayer, self).__init__()
        # （1）The first part， field feature trans，similar with NON
        field_length = len(field_dims)
        self.input_dim = field_length * embed_dim
        # local alignment 这里不对 使用向量矩阵相乘的方式
        self.local_w = nn.Parameter(torch.randn(field_length, embed_dim, embed_dim))
        # 每个field 提供一个
        self.local_b = nn.Parameter(torch.randn(field_length, 1, embed_dim))

        # （2) Applying Attention for each rrefined-features.
        self.self_att = SelfAttention()
        self.type_c = type_c



    def forward(self, x_emb):

        # 1、
        x_local = torch.matmul(x_emb.permute(1, 0, 2), self.local_w) + self.local_b
        x_local = x_local.permute(1, 0, 2)

        # 2、Attention
        # Q，refined Embedding: [B,F,E]
        # K, original Embedding：[B,F,E]
        # V，original Embedding，[B,F,E]
        # TODO: QKV是否需要特征转化，目前没有进行转化
        x_att, att_score = self.self_att(x_local, x_emb, x_emb, scale=True)

        # 3、How to combine x_emb(original emb)、x_local() and x_att(feature affinity)?
        #  1）全连接？ 2）mean pooling 3） max pooling 4) 直接使用 x_att
        #  消融实验，a: 使用x_local, b: 使用 x_att c:使用其他方式

        # 这里处理两组embedding：x_local and x_att , they represent
        if self.type_c == "hard":
            return x_att * x_local

        if self.type_c == "local":
            return x_local

        if self.type_c == "att":
            return x_att

        #     这是针对三组embedding的，
        if self.type_c == "cat":
            x_con = torch.cat([x_emb, x_local, x_att], dim=2)  # [b,e,3*f]
            return x_con

        x_stack = torch.stack([x_emb,x_local,x_att],dim=2) # [b,e,3,f]
        if self.type_c == "mean":
            return torch.mean(x_stack,dim=2)

        if self.type_c == "max":
            return torch.max(x_stack,dim=2)

        # if self.type_c == "cat2":

        return x_att



class IFMLayer(nn.Module):
    """
    IFM
    """
    def __init__(self, field_dims, embed_dim, embed_dims=[256,256,256], field_len=10, h=1):
        super(IFMLayer, self).__init__()
        field_length = field_len
        embed_dims.append(field_length)
        self.h = h

        input_dim = field_length*embed_dim
        self.mlps = MultiLayerPerceptron(input_dim, embed_dims,output_layer=False)

        # 对特征进行转换
        # self.pra = nn.Parameter(torch.rand(embed_dims[-1], field_length))
        # nn.init.kaiming_uniform_(self.pra.data())


    def forward(self,x_emb):
        """
        :param x:
        :return:
        x_emb：already multiple with x_att,
        x_att: the weight of w_i and x_emb.
        """
        # (1) 输入进行concat
        x_con = x_emb.view(x_emb.size(0),-1)


        # （2）使用MLP层，最后一层的维度是num_fields，需要使用softmax计算原始向量的权重
        # [B,256]
        x_att = self.mlps(x_con)

        x_att = self.h * F.softmax(x_att,dim=1)
        x_att = x_att.unsqueeze(-1)
        # 乘以权重
        x_emb = x_emb * x_att
        return x_emb, x_att

class DIFMLayer(nn.Module):
    """
    DIFM
    """
    def __init__(self):
        super(DIFMLayer, self).__init__()
        pass

    def forward(self,x):
        """
        :param x:
        :return:
        """

        return x


# class MultiLayerPerceptronAux(torch.nn.Module):
#     """
#     fc 线性层
#     输入 连接之后的 [B,num_field*k]  input_dim = num_field*k
#     """
#
#     def __init__(self, input_dim, embed_dims, dropout=0.5, output_layer=True):
#         super().__init__()
#         self.mlp1 = nn.Sequential(
#             nn.Linear(input_dim,embed_dims[0]),
#             torch.nn.BatchNorm1d(embed_dims[0]),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(p=dropout)
#         )
#         self.output1 = nn.Linear(embed_dims[0],1)
#
#         self.mlp2 = nn.Sequential(
#             nn.Linear(embed_dims[0],embed_dims[1]),
#             torch.nn.BatchNorm1d(embed_dims[1]),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(p=dropout)
#         )
#         self.output2 = nn.Linear(embed_dims[1], 1)
#
#         self.mlp3 = nn.Sequential(
#             nn.Linear(embed_dims[1], embed_dims[2]),
#             torch.nn.BatchNorm1d(embed_dims[2]),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(p=dropout)
#         )
#         self.output3 = nn.Linear(embed_dims[2], 1)
#         self._init_weight_()
#
#     def _init_weight_(self):
#         """ We leave the weights initialization here. """
#         for m in self.mlp1:
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#
#         for m in self.mlp2:
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#
#         for m in self.mlp3:
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#
#     def forward(self, x):
#         """
#         :param x: Float tensor of size ``(batch_size,num_fields*embed_dim)``
#         """
#         x = self.mlp1(x)
#         output1 = self.output1(x)
#         x = self.mlp2(x)
#         output2 = self.output2(x)
#         x = self.mlp3(x)
#         output3 = self.output3(x)
#         return output3,output2,output1

#
# class MultiLayerPerceptronLayers(torch.nn.Module):
#
#     """
#      和上面不同的是，输入的是layer，
#     """
#     def __init__(self, embed_dim, mlp_layers, dropout=0.1, output_layer=True):
#         super(MultiLayerPerceptronLayers, self).__init__()
#         layers = list()
#
#         # 给定了训练的层数 和 embed_size ,输入为 [B,nf = mbed_size * (2**（mlp_layers-1）)]
#
#         for i in range(mlp_layers):
#             # [N,]
#             input_size = embed_dim * (2**(mlp_layers - i))
#             layers.append(torch.nn.Linear(input_size, input_size // 2))
#             # layers.append(torch.nn.BatchNorm1d(input_size // 2))
#             layers.append(torch.nn.ReLU())
#             layers.append(torch.nn.Dropout(p=dropout))
#         if output_layer:
#             layers.append(torch.nn.Linear(embed_dim, 1))
#
#         self.mlp = torch.nn.Sequential(*layers)
#
#         self._init_weight_()
#
#     def _init_weight_(self):
#         """ We leave the weights initialization here. """
#         for m in self.mlp:
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#
#     def forward(self, x):
#         """
#         :param x: [B,nf * emb_size*(2**(mlp_layers-1))]
#         :return:
#         """
#         return self.mlp(x)


# define the MLP model
# 这里用layer
# class MLP(nn.Module):
#     def __init__(self, user_num, item_num, factor_num, num_layers, dropout):
#         super(MLP, self).__init__()
#
#         self.embed_user_MLP = nn.Embedding(
#             user_num, factor_num * (2 ** (num_layers - 1)))
#         self.embed_item_MLP = nn.Embedding(
#             item_num, factor_num * (2 ** (num_layers - 1)))
#
#         MLP_modules = []
#         for i in range(num_layers):
#             input_size = factor_num * (2 ** (num_layers - i))
#             MLP_modules.append(nn.Dropout(p=dropout))
#             MLP_modules.append(nn.Linear(input_size, input_size // 2))
#             MLP_modules.append(nn.ReLU())
#         self.MLP_layers = nn.Sequential(*MLP_modules)
#
#         self.predict_layer = nn.Linear(factor_num, 1)
#
#         self._init_weight_()
#
#     def _init_weight_(self):
#         nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
#         nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
#
#         for m in self.MLP_layers:
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.kaiming_uniform_(
#                     self.predict_layer.weight, a=1, nonlinearity='sigmoid')
#
#     def forward(self, user, item):
#         #  [B,1],[B,1]
#         embed_user_MLP = self.embed_user_MLP(user)
#         embed_item_MLP = self.embed_item_MLP(item)
#         interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
#         output_MLP = self.MLP_layers(interaction)
#         prediction = self.predict_layer(output_MLP)
#         return prediction.view(-1)


# class InnerProductNetwork(torch.nn.Module):
#
#     def __init__(self,is_sum=True):
#         super(InnerProductNetwork, self).__init__()
#         self.is_sum = is_sum
#
#     def forward(self, x):
#         """
#         :param is_sum:
#         :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
#         """
#         num_fields = x.shape[1]
#         row, col = list(), list()
#
#         for i in range(num_fields - 1):
#             for j in range(i + 1, num_fields):
#                 row.append(i), col.append(j)  #
#         #
#         if self.is_sum == True:
#             # 默认求和最原始的方式
#             return torch.sum(x[:, row] * x[:, col], dim=2)  # B,1/2* nf*(nf-1)
#         else:
#             #  以下： 如果不求和 B,1/2* nf*(nf-1), K
#             return x[:, row] * x[:, col]

# class IPNN(torch.nn.Module):
#
#     def __init__(self):
#         super(IPNN, self).__init__()
#         self.is_sum = is_sum
#
#     def forward(self, x):
#         """
#         :param is_sum:
#         :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
#         """
#         num_fields = x.shape[1]
#         row, col = list(), list()
#
#         for i in range(num_fields - 1):
#             for j in range(i + 1, num_fields):
#                 row.append(i), col.append(j)  #
#         #
#         if self.is_sum == True:
#             # 默认求和最原始的方式
#             return torch.sum(x[:, row] * x[:, col], dim=2)  # B,1/2* nf*(nf-1)
#         else:
#             #  以下： 如果不求和 B,1/2* nf*(nf-1), K
#             return x[:, row] * x[:, col]

# class PNNActivationWeight(torch.nn.Module):
    # def __init__(self,embed_dim=128,dropout=0.1):
    #     super(PNNActivationWeight, self).__init__()
    #
    #     self.fcs = nn.Sequential(
    #         nn.Linear(3*embed_dim,64),
    #         nn.PReLU(),
    #         nn.Dropout(p=dropout),
    #         nn.Linear(64, 1))
    #
    # def forward(self,x):
    #     """
    #     :param x1: B, n_f,e
    #     :param x2:B,n_f ,e
    #     :return:
    #     """
    #     num_fields = x.shape[1]
    #     row, col = list(), list()
    #
    #     for i in range(num_fields - 1):
    #         for j in range(i + 1, num_fields):
    #             row.append(i), col.append(j)  #
    #     a = torch.cat([x[:,row],x[:, row] * x[:, col], x[:, col]],dim=-1)
    #     a = self.fcs(a)
    #     # print(a.size())
    #     return a.squeeze(-1)
    #     # return torch.sum(x[:, row] * x[:, col], dim=2)  # B,1/2* nf*(nf-1)


#
# class OuterProductNetwork(torch.nn.Module):
#
#     def __init__(self, num_fields, embed_dim, kernel_type='mat'):
#         super().__init__()
#         num_ix = num_fields * (num_fields - 1) // 2
#         if kernel_type == 'mat':
#             kernel_shape = embed_dim, num_ix, embed_dim
#         elif kernel_type == 'vec':
#             kernel_shape = num_ix, embed_dim
#         elif kernel_type == 'num':
#             kernel_shape = num_ix, 1
#         else:
#             raise ValueError('unknown kernel type: ' + kernel_type)
#         self.kernel_type = kernel_type
#
#         self.kernel = torch.nn.Parameter(torch.zeros(kernel_shape))
#
#         torch.nn.init.xavier_uniform_(self.kernel.data)
#
#     def forward(self, x):
#         """
#         :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
#         """
#         num_fields = x.shape[1]
#         row, col = list(), list()
#         for i in range(num_fields - 1):
#             for j in range(i + 1, num_fields):
#                 row.append(i), col.append(j)
#
#         p, q = x[:, row], x[:, col]  # B,n,emb
#
#         if self.kernel_type == 'mat':
#             #  p [b,1,num_ix,e]
#             #  kernel [e, num_ix, e]
#             kp = torch.sum(p.unsqueeze(1) * self.kernel,dim=-1).permute(0,2,1)  #b,num_ix,e
#             # #b,num_ix,e
#             return torch.sum(kp * q, -1)
#         else:
#             return torch.sum(p * q * self.kernel.unsqueeze(0), -1)

#
# class CrossNetwork(torch.nn.Module):
#
#     def __init__(self, input_dim, num_layers):
#         super().__init__()
#
#         self.num_layers = num_layers
#
#         self.w = torch.nn.ModuleList([
#             torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
#         ])
#         # bias
#         self.b = torch.nn.ParameterList([torch.nn.Parameter(
#             torch.zeros((input_dim,))) for _ in range(num_layers)])
#
#     def forward(self, x):
#         """
#         :param x: Float tensor of size ``(batch_size, num_fields*embed_dim)``
#         考虑改进DCN
#         """
#         x0 = x
#         for i in range(self.num_layers):
#             xw = self.w[i](x)
#             x = x0 * xw + self.b[i] + x
#         return x


class AttentionalFactorizationMachine(torch.nn.Module):
    """
    实现attention ,
    """

    def __init__(self, embed_dim, attn_size, dropouts, reduce = True):
        super().__init__()

        self.attention = torch.nn.Linear(embed_dim, attn_size)
        self.projection = torch.nn.Linear(attn_size, 1)
        self.fc = torch.nn.Linear(embed_dim, 1)
        self.dropouts = dropouts
        self.reduce = reduce

        # self.v = nn.Parameter(torch.rand(1, hidden_size))  # 此处定义为Linear也可以
        # nn.init.xavier_normal_(self.v.data)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]

        inner_product = p * q  # B,n_f*(n_f-1), emb

        attn_scores = F.relu(self.attention(inner_product)) # 先改变维度，

        # projecction相当于 新建 一个 nn.paramater(torch.zeros((att_size,)))
        attn_scores = F.softmax(self.projection(attn_scores), dim=1) # B,n_f-1, 1
        # attention也需要dropout吗？？
        attn_scores = F.dropout(attn_scores, p=self.dropouts[0])

        # 和权重相乘
        attn_output = torch.sum(attn_scores * inner_product, dim=1) # B,e
        attn_output = F.dropout(attn_output, p=self.dropouts[1]) # B,1
        # 返回最后一层
        if self.reduce == False:
            return attn_output
        # 对最后一层求和
        return self.fc(attn_output)





class CompressedInteractionNetwork(torch.nn.Module):
    # xDeepFM： CIN

    def __init__(self, input_dim, cross_layer_sizes, split_half=True):
        super().__init__()
        self.num_layers = len(cross_layer_sizes)
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            cross_layer_size = cross_layer_sizes[i]
            self.conv_layers.append(torch.nn.Conv1d(input_dim * prev_dim, cross_layer_size,1,stride=1,dilation=1,bias=True))
            if self.split_half and i != self.num_layers - 1:
                cross_layer_size //= 2
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        xs = list()
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = F.relu(self.conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)

        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))

if __name__ == '__main__':

    x = torch.randn((10,10))
    y = torch.softmax(x,dim=1)
    _,pred = torch.max(y,dim=1)
    print(pred)
