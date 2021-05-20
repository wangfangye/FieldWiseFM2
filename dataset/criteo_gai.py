# import math
# import shutil
# import struct
# from collections import defaultdict
# from functools import lru_cache
# from pathlib import Path
#
# import lmdb
# import numpy as np
# import torch.utils.data
# from tqdm import tqdm
# import random
# import pickle
# from utils_de import FIELD_NUMS
#
#
#
#
# class CriteoDataset(torch.utils.data.Dataset):
#     """
#     Criteo Display Advertising Challenge Dataset
#
#     Data prepration:
#         * Remove the infrequent features (appearing in less than threshold instances) and treat them as a single feature
#         * Discretize numerical values by log2 transformation which is proposed by the winner of Criteo Competition
#
#     :param dataset_path: criteo train.txt path.
#     :param cache_path: lmdb cache path.
#     :param rebuild_cache: If True, lmdb cache is refreshed.
#     :param min_threshold: infrequent feature threshold.
#
#     Reference:
#         https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset
#         https://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf
#     """
#
#     def __init__(self, dataset_path=None,is_train=True):
#         self.NUM_FEATS = 39
#         self.NUM_INT_FEATS = 13
#         self.min_threshold = 10
#         self.field_dims = FIELD_NUMS
#         self.prefix = "./data/criteo/"
#         self.feat_mapper, self.defaults = self.__read_train_all_feats()
#         self.is_train = is_train
#         # self.data_all = self.__yield_buffer(path=dataset_path)
#         # 直接从内存中读取
#         self.data_all = self.__read_data()
#
#     def __getitem__(self, index):
#         # 必须要实现的 从缓存中读取数据
#         np_array = self.data_all[index]
#         return np_array[1:], np_array[0]
#
#     def __len__(self):
#         # 必须要实现的
#         if self.is_train:
#             return 40840617
#         else:
#             return 5000000
#
#     def __read_train_all_feats(self):
#         # 读取特征,目前固定
#         # 加载完整的特征字典 feat_cnts,defaults， 返回给criteo_dataset
#         return pickle.load(open("train_all_feat.pkl", "rb"))
#
#     def __read_data(self):
#         if self.is_train:
#             # 如果是训练接
#             data_train = list()
#             for i in range(7):
#                 path = f"train_arrays/train_arrays_0{i}.p"
#                 split_data = pickle.load(open(path,"rb"))
#                 data_train.extend(split_data)
#             print(len(data_train))
#             return data_train
#         else:
#             data_test = pickle.load(open("test_arrays.pkl", "rb"))
#             print(len(data_test))
#             return data_test
#
#
#     def __yield_buffer(self, path):
#         """
#         :param path: 训练数据 或者是测试数据 路径
#         :return:
#         """
#         feat_mapper, defaults = self.feat_mapper,self.defaults
#         buffer = list()
#         with open(path) as f:
#             pbar = tqdm(f, mininterval=1, smoothing=0.1)
#             pbar.set_description("构建大量的数据")
#             split = 0
#             print(path)
#             for line in pbar:
#                 values = line.rstrip('\n').split('\t')
#                 if len(values) != self.NUM_FEATS + 1:
#                     continue
#                 np_array = np.zeros(self.NUM_FEATS + 1, dtype=np.uint32)
#                 np_array[0] = int(values[0])
#                 # 先数值型的变量
#                 for i in range(1, self.NUM_INT_FEATS + 1):
#                     np_array[i] = feat_mapper[i].get(convert_numeric_feature(values[i]), defaults[i])
#
#                 for i in range(self.NUM_INT_FEATS + 1, self.NUM_FEATS + 1):
#                     np_array[i] = feat_mapper[i].get(values[i], defaults[i])
#                 buffer.append(np_array)
#                 if len(buffer) == 6000000:
#                     f = open(f"train_arrays/train_arrays_0{split}.p", "wb")
#                     pickle.dump(buffer, f)
#                     print("保存训练数据成功",split)
#                     split+=1
#                     buffer = list()
#
#             print(len(buffer))
#             if self.is_train:
#                 f = open(f"train_arrays/train_arrays_0{split}.p", "wb")
#                 pickle.dump(buffer, f)
#                 print("保存训练数据成功",split)
#             else:
#                 f = open("test_arrays.p","wb")
#                 pickle.dump(buffer, f)
#                 print("保存测试数据成功")
#             return buffer
#
#
# @lru_cache(maxsize=None)
# def convert_numeric_feature(val: str):
#     if val == '':
#         return 'NULL'
#     v = int(val)
#     # ONN中的处理方式
#     # return math.floor(2*math.log(val))
#     if v > 2:
#         # ln(x)**2
#         # return
#         return str(int(math.log(v) ** 2))
#     else:
#         return str(v - 2)
#
#
# # 这里需要选择究竟用什么方式
# @lru_cache(maxsize=None)
# def convert_numeric_feature2(val: str):
#     if val == '':
#         return 'NULL'
#     val = int(val)
#     # ONN中的处理方式
#     # 0 怎么办 三种处理方式
#     if val == 0:
#         return "ZERO"
#     elif val < 0:
#         return str(math.floor(-2 * math.log(-val)))
#     return str(math.floor(2 * math.log(val)))
#
#
# def get_criteo_dataset_loader_gai(train_path="criteo_train_4000w.txt", test_path="criteo_test_500w.txt", batch_size=256):
#     # the test_path maybe null, if it is, we need to split the train dataset
#     print("start load dataset from valid ache")
#     prefix = "./data/criteo/"
#     train_path = prefix + train_path
#     test_path = prefix + test_path
#
#     train_dataset = CriteoDataset(dataset_path=train_path, is_train=True)
#     field_dims = train_dataset.field_dims
#     print("开始加载测试集合")
#     test_dataset = CriteoDataset(dataset_path=test_path, is_train=False)
#
#
#     train_length = len(train_dataset)
#     valid_length = 5000000
#
#     train_dataset, valid_dataset = torch.utils.data.random_split(
#         train_dataset, [train_length - valid_length, valid_length])
#
#     print("train_dataset length:",len(train_dataset))
#     print("valid_dataset length:",len(valid_dataset))
#     print("test_dataset length:",len(test_dataset))
#
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,drop_last=True,pin_memory=True)
#
#     valid_loader = torch.utils.data.DataLoader(
#         valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4,drop_last=False)
#
#     test_loader = torch.utils.data.DataLoader(
#         test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,drop_last=False)
#
#     # pickle.dump((train_dataset,test_dataset,field_dims),open("all_data.pkl","wb"))
#     # 3500w 500w 500w
#     return field_dims,train_loader,valid_loader,test_loader
#
#
# def get_criteo_dataset_train(train_path, batch_size=128):
#     # the test_path maybe null, if it is, we need to split the train dataset
#     print("Start loading criteo data....")
#
#     prefix = "./data/criteo/"
#     train_path = prefix + train_path
#     # train_sub100w.txt
#     if train_path == "train.txt":
#         dataset = CriteoDataset(dataset_path=train_path, cache_path=prefix + ".criteoall")
#     else:
#         dataset = CriteoDataset(dataset_path=train_path, cache_path=prefix + ".criteo")
#     all_length = len(dataset)
#     print(all_length)
#
#     train_size = int(0.9 * all_length)
#     test_size = all_length - train_size
#
#     train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
#
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#     print("train_loader", train_loader.__len__())
#     print("test_loader", test_loader.__len__())
#
#     field_dims = dataset.field_dims
#     return field_dims, train_loader, test_loader
#
#
# """
# Only the sub_train100w.txt
# field_dims:[  35   82   78   31  209   91   64   36   79    8   27   29   36  321
#   504 6373 7426  104   13 7100  168    4 7371 3663 6495 2800   27 4172
#  6722   11 2162 1130    5 6554   11   15 5506   48 4284]
#
#
#
# the full data: train.txt, which contains about 45,000,000 simple, 13 numerical features and 26 categorical feature
#
# [    49    101    126     45    223    118     84     76     95      9
#      30     40     75   1458    555 193949 138801    306     19  11970
#     634      4  42646   5178 192773   3175     27  11422 181075     11
#    4654   2032      5 189657     18     16  59697     86  45571]
#
# 161159*256 / 0.9  长度
#
#
# [193949, 192773, 189657, 181075, 138801, 59697, 45571, 42646, 11970, 11422,
# 5178, 4654, 3175, 2032, 1458, 634, 555, 306, 223, 126,
# 118, 101, 95, 86, 84, 76, 75, 49, 45, 40,
# 30, 27, 19, 18, 16, 11, 9, 5, 4]
#
# 特征总数： 1086810
#
# """
#
# if __name__ == '__main__':
#     a = [[1],[2]]
#     b = [[3],[5]]
#     a.extend(b)
#     print(a)
#
#     # train_path = "train_sub100w.txt"
#     # train_path2 = "train.txt"
#     # field_dims, train_loader, test_loader = get_criteo_dataset_train(train_path2, batch_size=256)
#     # print(field_dims)
#     #
#     # print(train_loader.__len__())
#
#     # dataset = CriteoDataset(dataset_path=".././data/criteo/train_sub100w.txt")
#     # trainLoader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
#     # one_iter = iter(trainLoader)
#
#     # print(len(dataset))
#     # print(dataset.field_dims)
#     # fields = dataset.field_dims
#     # alls = 0
#     #
#     # from functools import reduce
#     #
#     # def add(x, y):
#     #     return x + y
#     # result = reduce(add, fields)
#     # print(result)
#
#     # for one in one_iter.next():
#     #     print(one)
