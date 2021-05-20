#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Author:wangfy
@project:DLRec
@Time:2020/4/17 8:22 下午

（1） criteo数据集
（2） 分为train,valid,test
（3）
'''

import sys

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
import math

# sys.path.append("..")


import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.FM import FactorizationMachineModel,GFRLFM
from model.NFM import NeuralFactorizationMachineModel
from model.AFM import AttentionalFactorizationMachineModel
from model.LR import LogisticRegression
from model.FM_Models.MyFM import MyIFM
from model.FM_Models.IFM import IFM
from model.ffm import FieldAwareFactorizationMachineModel
from model.WD import WideAndDeepModel
from model.DFM import DeepFactorizationMachineModel


import numpy as np
import random

import os
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error,mean_absolute_error

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.tensorboard import SummaryWriter

PATH_to_log_dir = "./tb/"
# writer = SummaryWriter(PATH_to_log_dir)

sys.path.append("../..")
# from dataset.criteo_valid import get_criteo_dataset_loader_valid
from dataset.frappe.MyloadData import getdataloader_frappe
from utils_de import *
from utils.earlystoping import EarlyStopping


def get_model(
        name,
        field_dims = 5382,
        embed_dim=256,
        mlp_layers=[256, 256, 256],
        field_len=10):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    #  CUDA_VISIBLE_DEVICES=3 python evalMy.py --model_name convnn --sample 2 --mains 2 --epoch 50 --learning_rate 0.01
    # field_dims = dataset.field_dims
    if name == "nfm":
        to_model = NeuralFactorizationMachineModel(field_dims, embed_dim, mlp_layers=[512], dropouts=(0.5, 0.5))
        return to_model

    elif name == "afm":
        to_model = AttentionalFactorizationMachineModel(field_dims, embed_dim, attn_size=16, dropouts=(0.5, 0.5))
        return to_model

    elif name == "wdl":
        return WideAndDeepModel(field_dims, embed_dim, mlp_dims=[1024,512,256],field_len=10)

    elif name == "dfm":
        return DeepFactorizationMachineModel(field_dims,embed_dim,dropout=0.5,field_len=10)

    elif name == "fm":
        return FactorizationMachineModel(field_dims, embed_dim)

    elif name == "gfrlfm":
        return GFRLFM(field_dims, embed_dim, type="glu",field_len=10)

    elif name == "mifm_hard":
        return MyIFM(field_dims, embed_dim,type_c="hard")

    elif name == "mifm_att":
        return MyIFM(field_dims, embed_dim, type_c="att")

    elif name == "mifm_local":
        return MyIFM(field_dims, embed_dim, type_c="local")

    elif name == "ifm":
        return IFM(field_dims, embed_dim, embed_dims=mlp_layers, field_len=field_len)

    elif name == "ffm":
        return FieldAwareFactorizationMachineModel(field_dims, embed_dim)

    elif name == "lr":
        return LogisticRegression(field_dims)

    else:
        raise ValueError('unknown model name: ' + name)

# CTR 预测的train
def count_params(model):
    params = sum(param.numel() for param in model.parameters())
    return params


# 计算RMSE和MAE，损失函数是MSELoss
def train(model,
          optimizer,
          data_loader,
          criterion,
          weight_decay=0.01):
    model.train()
    total_loss = 0
    # for i, (user_item, label) in enumerate(tqdm.tqdm(data_loader, ncols=80, position=0)):
    for i, (user_item, label) in enumerate(data_loader):
        # print(user_item)
        label = label.float()
        user_item = user_item.long()
        user_item = user_item.cuda()  # [B,n_f]
        label = label.cuda()  # [B]


        pred_y = model(user_item).squeeze(1)
        pred_y = torch.clamp(pred_y, -1.0, 1.0)
        loss_mse = criterion(pred_y, label)
        total_loss += loss_mse.item()
        loss = my_l2_loss(pred_y,label) + model.get_l2_loss(lambdas=weight_decay)
        # loss = loss_mse + model.get_l2_loss(lambdas=weight_decay)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss2 = total_loss / (i + 1)
    return loss2**0.5

def test_roc(model, data_loader):
    num_example = 28842
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        # for fields, target in tqdm.tqdm(data_loader, ncols=80, position=0):
        for fields, target in data_loader:
            fields = fields.long()
            target = target.float()
            fields, target = fields.cuda(), target.cuda()
            # y = torch.sigmoid(model(fields).squeeze(1))
            y = model(fields).squeeze(1)
            # 对值的范围进行裁切, （-1，1）
            y = torch.clamp(y, -1.0, 1.0)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    # predictions_bounded = np.maximum(predicts, np.ones(num_example) * -1.0)  # bound the lower values
    #
    # predictions_bounded = np.minimum(predictions_bounded,
    #                                  np.ones(num_example) * 1.0)  # bound the higher values
    RMSE = math.sqrt(mean_squared_error(targets, predicts))
    MAE = mean_absolute_error(targets, predicts)
    return RMSE,MAE

def adjust_learning_rate(lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def my_l2_loss(pred,label):
    return torch.sum(torch.pow((pred-label),2))

def my_l1_loss(pred,label):
    return torch.sum(torch.abs(pred-label))

def main2(dataset_name,
          model_name,
          epoch,
          learning_rate,
          batch_size,
          weight_decay,
          save_dir,
          path):
    path = "./data/data2/"
    # ml 默认type_y=mse
    trainLoader, validLoader, testLoader = \
        getdataloader_frappe(path=path, batch_size=batch_size)
    field_dims = 5382
    print(field_dims)
    # 路径
    time_fix = time.strftime("%d%H%M%S", time.localtime())
    # wdl pretrain
    # for K in [10,20,30,40,50,64]:  # latent factors
    for K in [256]:  # latent factors
        # fout = open(paths+"/logs.p","a+")
        paths = os.path.join(save_dir, dataset_name, model_name, str(K))
        if not os.path.exists(paths):
            os.makedirs(paths)
        with open(paths + f"/{model_name}logs2_{time_fix}_{learning_rate}_{weight_decay}.p", "a+") as fout:

            # criterion = torch.nn.BCELoss() # 结果是0和1 , 所以是bceloss,
            model = get_model(
                field_dims=field_dims,
                name=model_name,
                embed_dim=K).cuda()
            # 记录配置
            params = count_params(model)
            fout.write("Batch_size:{}\tlearning_rate:{}\tStartTime:{}\tweight_decay:{}\tparams:{}\n"
                       .format(batch_size, learning_rate, time.strftime("%d%H%M%S", time.localtime()), weight_decay,params))
            print("Start train -- K : {}".format(K))
            print(params)

            # 损失函数和优化器
            criterion = torch.nn.MSELoss(reduction="mean")

            optimizer = torch.optim.Adagrad(
                params=model.parameters(),
                lr = learning_rate,
                initial_accumulator_value = 1e-6,
                weight_decay=weight_decay
            )

            # optimizer = torch.optim.Adam(
            #     params=model.parameters(),
            #     lr = learning_rate,
            #     weight_decay=weight_decay,
            # )

            # early_stopping = EarlyStopping(patience=6, verbose=True, prefix=path)

            val_rmse_best = 1000
            rmse_index_record = ""  # 记录当前的值

            val_mae_best = 1000
            mae_index_record = ""  # 记录当前的值

            # scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True, patience=6)
            scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=6)

            for epoch_i in range(epoch):
                # adjust_learning_rate(learning_rate, optimizer, epoch)

                print("frappe", model_name, K, epoch_i, weight_decay, learning_rate)

                start = time.time()

                train_rmse = train(model, optimizer=optimizer, data_loader=trainLoader, criterion=criterion, weight_decay=weight_decay)

                # valid 计算mae，rmse
                val_rmse, val_mae = test_roc(model, validLoader)

                # test
                test_rmse, test_mae = test_roc(model, testLoader)

                # 是否调整学习率
                scheduler.step(val_rmse)
                end = time.time()

                if val_rmse < val_rmse_best:
                    val_rmse_best = val_rmse
                    rmse_index_record = "epoch_i:{}\t{:.6f}\t{:.6f}".format(epoch_i, test_rmse, test_mae)

                if val_mae < val_mae_best:
                    val_mae_best = val_mae
                    mae_index_record = "epoch_i:{}\t{:.6f}\t{:.6f}".format(epoch_i, test_rmse, test_mae)

                print(
                    "Train  K:{}\tEpoch:{}\ttrain_rmse:{:.6f}\tval_rmse:{:.6f}\tval_mae:{:.6f}\ttime:{:.6f}\ttest_rmse:{:.6f}\ttest_mae:{:.6f}\n"
                        .format(K, epoch_i, train_rmse, val_rmse, val_mae, end - start, test_rmse, test_mae))

                fout.write(
                    "Train  K:{}\tEpoch:{}\ttrain_rmse:{:.6f}\tval_rmse:{:.6f}\tval_mae:{:.6f}\ttime:{:.6f}\ttest_rmse:{:.6f}\ttest_mae:{:.6f}\n"
                        .format(K, epoch_i, train_rmse, val_rmse, val_mae, end - start, test_rmse, test_mae))

                # 判断是否早停止
                # early_stopping(val_rmse)

            print("Test:{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n"
                  .format(K, val_rmse, val_rmse_best, val_mae, val_mae_best, test_rmse, test_mae))

            fout.write("Test:{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n"
                       .format(K, val_rmse, val_rmse_best, val_mae, val_mae_best, test_rmse, test_mae))
            # 记录最好的两组结果
            fout.write("Rmse_best:\t{}\nMae_best:\t{}".format(rmse_index_record, mae_index_record))

            # 保存模型参数
            #
            torch.save({"state_dict":model.state_dict()},paths+f"/{model_name}_{K}_{time_fix}_{val_rmse_best}.pt")

            # torch.save({"state_dict": model.state_dict(), "best_auc": val_auc_best},
            #            paths + f"/{model_name}_final_{K}_{time_fix}.pt")
            # writer.close()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子

if __name__ == '__main__':

    setup_seed(2021)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='ml_frappe')
    parser.add_argument('--path', default="train_sub100w.txt", help="")
    parser.add_argument('--model_name', default='fm')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--device', default='cuda:0', help="cuda:0")
    parser.add_argument('--save_dir', default='chkpt_frappe')
    parser.add_argument('--choice', default=0,type=int)
    args = parser.parse_args()

    if args.choice == 0:
        # embedding相同
        model_names = ["fm"] * 3

    elif args.choice == 1:
        # embedding相同
        model_names = ["afm"]*3

    elif args.choice == 2:
        # embedding相同
        model_names = ["nfm"]*3

    elif args.choice == 3:
        model_names = ["lr"] * 3

    elif args.choice == 4:
        # model_names = ["mifm_hard","mifm_att","mifm_local"] * 3
        model_names = ["mifm_local"] * 3

    elif args.choice == 5:
        model_names = ["ifm"] * 3

    elif args.choice == 6:
        model_names = ["wdl"] * 3

    elif args.choice == 7:
        model_names = ["dfm"] * 3

    elif args.choice == 8:
        model_names = ["gfrlfm"] * 3

    print(model_names)

    for name in model_names:
        main2(dataset_name=args.dataset_name,
              model_name=name,
              epoch=args.epoch,
              learning_rate=args.learning_rate,
              batch_size=args.batch_size,
              weight_decay=args.weight_decay,
              save_dir=args.save_dir,
              path=args.path)
