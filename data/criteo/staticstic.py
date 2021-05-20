#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Author:wangfy
@project:PNNConvModel
@Time:2020/9/3 5:09 下午
'''

import argparse
import pickle
from functools import lru_cache
from collections import defaultdict
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

def count_feature(path):
#     按行读取文件
    NUM_FEATS =  39
    NUM_INT_FEATS = 13
    min_threshold = 10
# 获取特征的对应关系
    feat_cnts = defaultdict(lambda: defaultdict(int))
    with open(path) as f:
        pbar = tqdm(f, mininterval=1, smoothing=0.1)
        pbar.set_description('Create criteo dataset cache: counting features')
        for line in pbar:
            values = line.rstrip('\n').split('\t')

            #  如果数据长度不够就跳过，不过只要有" " 空格也是可以的
            if len(values) != NUM_FEATS + 1:
                continue

            for i in range(1, NUM_INT_FEATS + 1):
                # 将连续型变量转化为离散型变量
                feat_cnts[i][convert_numeric_feature(values[i])] += 1

            for i in range(NUM_INT_FEATS + 1, NUM_FEATS + 1):
                feat_cnts[i][values[i]] += 1

            feat_cnts[0][values[0]] += 1

    feat_mapper = {i: {feat for feat, c in cnt.items() if c >= min_threshold} for i, cnt in feat_cnts.items()}
    feat_mapper = {i: {feat: idx for idx, feat in enumerate(cnt)} for i, cnt in feat_mapper.items()}
    defaults = {i: len(cnt) for i, cnt in feat_mapper.items()}
    return feat_mapper,defaults

@lru_cache(maxsize=None)
# 这里需要选择究竟用什么方式
def convert_numeric_feature(val: str):
    if val == '':
        return 'NULL'
    v = int(val)
    # ONN中的处理方式
    # return math.floor(2*math.log(val))
    if v > 2:
        # ln(x)**2
        # return
        return str(int(math.log(v) ** 2))
    else:
        return str(v - 2)

@lru_cache(maxsize=None)
# 这里需要选择究竟用什么方式
def convert_numeric_feature2(val: str):
    if val == '':
        return 'NULL'
    val = int(val)
    # ONN中的处理方式
    # 0 怎么办
    if val == 0:
        return "ZERO"
    elif val < 0:
        return str(math.floor(-2*math.log(-val)))
    return str(math.floor(2*math.log(val)))

def main(path):
    f = open("train_all_feat.pkl","wb")
    feat_cnts,defaults = count_feature(path)
    # 保存在里面
    pickle.dump((dict(feat_cnts),defaults),f)
    # 再确认一次
    pickle.dump(defaults,open("field_num.pkl","wb"))

def pick_index():
    feat_cnt = pickle.load(open("feature_cnt_all.pkl","rb"))
    pickle.dump(feat_cnt[0],open("label.pkl","wb"))

def read_train_all_feats():
    # 加载完整的特征字典 feat_cnts,defaults， 返回给criteo_dataset
    return pickle.load(open("train_all_feat.pkl","rb"))
    # return pickle.load(open("label.pkl","rb"))
# 0: 34095179 1:11745438

def count_label():
    label_dict = defaultdict(int)
    with open("criteo_test_500w.txt","r") as f:
        pbar = tqdm(f, mininterval=1, smoothing=0.1)
        pbar.set_description('Create criteo dataset cache: counting features')
        for line in pbar:
            values = line.rstrip('\n').split('\t')
            label_dict[values[0]] += 1

    for k,v in label_dict.items():
        print(k,"=====",v)




if __name__ == '__main__':
    count_label()
    # file_path = r"train_sub100w.txt"
    # feat_cnt = count_feature(file_path)
    # f = open("feature_cnt.pkl","wb")
    # parser = argparse.ArgumentParser()
    # # parser.add_argument("--index",type=int,default=0)
    # parser.add_argument("--path",type=str,default="train.txt")
    # args = parser.parse_args()
    # main(args.path)




