import shutil
import struct
from collections import defaultdict
from pathlib import Path

import lmdb
import numpy as np
import torch.utils.data
from tqdm import tqdm
import pickle



class AvazuDataset(torch.utils.data.Dataset):
    """
    Avazu Click-Through Rate Prediction Dataset

    Dataset preparation
        Remove the infrequent features (appearing in less than threshold instances) and treat them as a single feature

    :param dataset_path: avazu train path
    :param cache_path: lmdb cache path
    :param rebuild_cache: If True, lmdb cache is refreshed
    :param min_threshold: infrequent feature threshold

    Reference
        https://www.kaggle.com/c/avazu-ctr-prediction

    统计数量：

    """

    def __init__(self, dataset_path=None, cache_path='.avazu', rebuild_cache=False, min_threshold=5):
        self.NUM_FEATS = 22 # 将hour算作一个特征，但是其实有点问题
        self.min_threshold = min_threshold
        # self.prefix = "./data/avazu/"
        self.prefix = "/home/fywang/project/remote_pro/PNNConvModel/data/avazu/"
        if rebuild_cache or not Path(cache_path).exists():
            shutil.rmtree(cache_path, ignore_errors=True)
            if dataset_path is None:
                raise ValueError('create cache: failed: dataset_path is None')
            self.__build_cache(dataset_path, cache_path)
        # 从缓存中获取数据
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries'] - 1
            self.field_dims = np.frombuffer(txn.get(b'field_dims'), dtype=np.uint32)

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            np_array = np.frombuffer(
                txn.get(struct.pack('>I', index)), dtype=np.uint32).astype(dtype=np.long)
        return np_array[1:], np_array[0]

    def __len__(self):
        return self.length

    def __build_cache(self, path, cache_path):
        # 构建缓存 （1） 先将特征转化为mapper； 然后保存mapper
        temp_path = self.prefix + "train" # 全部的数据集
        print("temp_path",temp_path)
        # 读取
        # feat_mapper, defaults = self.__get_feat_mapper(temp_path)
        feat_mapper, defaults = self.__read_train_all_feats()

        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            field_dims = np.zeros(self.NUM_FEATS, dtype=np.uint32)
            for i, fm in feat_mapper.items():
                # 第1、2个特征都不要
                field_dims[i - 2] = len(fm) + 1
            with env.begin(write=True) as txn:
                txn.put(b'field_dims', field_dims.tobytes())
            for buffer in self.__yield_buffer(path, feat_mapper, defaults):
                with env.begin(write=True) as txn:
                    for key, value in buffer:
                        txn.put(key, value)

    def __read_train_all_feats(self):
        # 读取特征,目前固定
        # 加载完整的特征字典 feat_cnts,defaults， 返回给criteo_dataset

        return pickle.load(open("avazu_all_feats.pkl", "rb"))


    def __get_feat_mapper(self, path):
        feat_cnts = defaultdict(lambda: defaultdict(int))

        with open(path) as f:
            # 第一行特征数据不要
            f.readline()
            pbar = tqdm(f,mininterval=1, smoothing=0.1)
            pbar.set_description('Create avazu dataset cache: counting features')
            for line in pbar:
                values = line.rstrip('\n').split(',')
                if len(values) != self.NUM_FEATS + 2: # 22+2
                    continue
                # 注意 values[1]是click，不使用，统计的时候保留，最后不使用
                for i in range(2, self.NUM_FEATS + 2):
                    feat_cnts[i][values[i]] += 1

        feat_mapper = {i: {feat for feat, c in cnt.items() if c >= self.min_threshold} for i, cnt in feat_cnts.items()}
        feat_mapper = {i: {feat: idx for idx, feat in enumerate(cnt)} for i, cnt in feat_mapper.items()}
        # 如何以前没有出现，则使用默认的
        defaults = {i: len(cnt) for i, cnt in feat_mapper.items()}

        # 写入
        f = open("avazu_all_feats.pkl", "wb")
        pickle.dump((feat_mapper, defaults), f)

        return feat_mapper, defaults

    def __yield_buffer(self, path, feat_mapper, defaults, buffer_size=int(1e5)):
        item_idx = 0
        buffer = list()
        with open(path) as f:
            # 删除第一行
            f.readline()
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create avazu dataset cache: setup lmdb')
            for line in pbar:
                values = line.rstrip('\n').split(',')
                if len(values) != self.NUM_FEATS + 2: # 23+1
                    continue
                np_array = np.zeros(self.NUM_FEATS+1 , dtype=np.uint32)
                # click数据
                np_array[0] = int(values[1])
                # 其他数据
                for i in range(2, self.NUM_FEATS + 2): #22
                    np_array[i-1] = feat_mapper[i].get(values[i], defaults[i])
                buffer.append((struct.pack('>I', item_idx), np_array.tobytes()))
                item_idx += 1
                if item_idx % buffer_size == 0:
                    yield buffer
                    buffer.clear()
            yield buffer

def get_avazu_dataset(train_path="train", batch_size=2048):

    print("Start loading avazu data....")

    # prefix = "./data/avazu/"
    prefix = "/home/fywang/project/remote_pro/PNNConvModel/data/avazu/"
    train_path = prefix + train_path

    print(train_path)
    # train
    train_dataset = AvazuDataset(dataset_path=train_path,cache_path= "/home/fywang/project/remote_pro/PNNConvModel/dataset"+"/.avazu_train",rebuild_cache=False)
    field_dims = train_dataset.field_dims
    all_length = len(train_dataset)
    print(all_length)
    print(field_dims)
    print(sum(field_dims))

    # 划分数据集 8:1:1
    valid_size = int(0.1 * all_length)
    train_size = all_length - valid_size
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size-valid_size, valid_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print("train_loader", train_loader.__len__())
    print("valid_loader", valid_loader.__len__())
    print("test_loader", test_loader.__len__())

    return field_dims, train_loader, valid_loader, test_loader


# 40428967
# 1544488
# [    241       8       8    3564    4325      25    5066     307      31
#   278182 1242892    6592       6       5    2483       9      10     431
#         5      68     169      61]
#
if __name__ == '__main__':
    # pass
    field_dims, train_loader, valid_loader, test_loader = get_avazu_dataset(batch_size=2048)
    # print(field_dims)
    # path = "/home/fywang/project/remote_pro/PNNConvModel/data/avazu/" + "train"
    # counts = defaultdict(int)
    # with open(path) as f:
    #     f.readline()
    #     pbar = tqdm(f, mininterval=1, smoothing=0.1)
    #     for line in pbar:
    #         values = line.rstrip('\n').split(',')
    #         counts[values[0]] += 1
    # # print(counts)
    # print(len(counts))







