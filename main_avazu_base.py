#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Author:wangfy
@project:DLRec
@Time:2020/4/17 8:22 下午
'''

import sys

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary

# sys.path.append("..")


import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.LR import LogisticRegression
from model.FM import FactorizationMachineModel
from model.AFM import AttentionalFactorizationMachineModel
from model.NFM import NeuralFactorizationMachineModel
from model.PlainDNN import PlainDNN
from model.DFM import DeepFactorizationMachineModel
from model.PNN import ProductNeuralNetworkModel
from model.FibiNet import FiBiNet
from model.AFN import AdaptiveFactorizationNetwork
from model.Autoint import AutomaticFeatureInteractionModel
from model.WD import WideAndDeepModel
from model.XDFM import ExtremeDeepFactorizationMachineModel,CIN
from model.ffm import FieldAwareFactorizationMachineModel
from model.DCN import DeepCrossNetworkModel
import numpy as np
import random

import os
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_log_error, mean_squared_error

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.tensorboard import SummaryWriter

PATH_to_log_dir = "./tb/"
# writer = SummaryWriter(PATH_to_log_dir)

sys.path.append("../..")
from dataset.avazu import get_avazu_dataset
from utils_de import *
from utils.earlystoping import EarlyStopping

# PRE_trained_path = "./chkpt/criteo/wdwd_best_auc_pre_10.pkl"


def get_model(
        name,
        field_dims,
        embed_dim=16,
        conv_size=16,
        mlp_layers=(400,400,400)):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    #  CUDA_VISIBLE_DEVICES=3 python evalMy.py --model_name convnn --sample 2 --mains 2 --epoch 50 --learning_rate 0.01
    if name == "nfm":
        to_model = NeuralFactorizationMachineModel(field_dims, embed_dim, mlp_layers, dropouts=(0.5, 0.5))
        return to_model
    elif name == "lr":
        return LogisticRegression(field_dims)

    elif name == "fm":
        return FactorizationMachineModel(field_dims,embed_dim)
    elif name == "dfm":
        return DeepFactorizationMachineModel(field_dims,embed_dim,dropout=0.5)
    elif name == "xdfm":
        to_model = ExtremeDeepFactorizationMachineModel(field_dims, embed_dim, mlp_dims=mlp_layers, dropout=0.5)
        return to_model
    # elif name == "dnn":
    #     return PlainDNN(field_dims,embed_dim,mlp_layers=mlp_layers,dropout=0.5)
    #
    elif name == "dcn":
        to_model = DeepCrossNetworkModel(field_dims,embed_dim,3,mlp_layers,dropout=0.5)
        return to_model
    elif name == "cin":
        return CIN(field_dims,embed_dim, dropout=0.5, cross_layer_sizes=(100,100,100,100), split_half=True)
    elif name == "afm":
        to_model = AttentionalFactorizationMachineModel(field_dims, embed_dim, attn_size=32, dropouts=(0.5, 0.5))
        return to_model

    elif name == "dnn":
        return PlainDNN(field_dims, embed_dim, mlp_layers=mlp_layers, dropout=0.5)
    #
    elif name == 'ipnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=embed_dim, mlp_layers=mlp_layers, method='inner',
                                         dropout=0.5)
    elif name == 'opnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=embed_dim, mlp_layers=mlp_layers, method='outer',
                                         dropout=0.5)

    elif name == "fibinet":
        return FiBiNet(field_dims, embed_dim=embed_dim,mlp_layers=mlp_layers,dropout=0.5)
    elif name == "afn":
        return AdaptiveFactorizationNetwork(field_dims, embed_dim=embed_dim,LNN_dim=100)

    elif name == "autoint":
        return AutomaticFeatureInteractionModel(field_dims, embed_dim=embed_dim, mlp_dims=mlp_layers, num_heads=4)

    elif name == "wdl":
        return WideAndDeepModel(field_dims, embed_dim, mlp_layers)

    elif name == "xdeepfm":
        return ExtremeDeepFactorizationMachineModel(field_dims, embed_dim, mlp_layers, dropout=0.5)

    elif name == "ffm":
        return FieldAwareFactorizationMachineModel(field_dims, embed_dim)
    else:
        raise ValueError('unknown model name: ' + name)



def metrics(model, test_loader, top_k):
    HR, NDCG = [], []

    for user_item, label in test_loader:
        if torch.cuda.is_available():
            user_item = user_item.to(DEVICE)

        predictions = model(user_item).view(-1)

        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(
            user_item[:, 1], indices).cpu().numpy().tolist()

        gt_item = user_item[0][1].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))
    return np.mean(HR), np.mean(NDCG)


# CTR 预测的train
def count_params(model):
    params = sum(param.numel() for param in model.parameters())
    return params

def train(model,
          optimizer,
          data_loader,
          criterion,
          device="cuda:0",
          log_interval=50000, ):
    model.train()
    pred = list()
    target = list()
    total_loss = 0
    for i, (user_item, label) in enumerate(tqdm.tqdm(data_loader)):
        # print(user_item)
        label = label.float()
        user_item = user_item.long()

        # if torch.cuda.is_available():

        user_item = user_item.cuda()  # [B,n_f]
        label = label.cuda()  # [B]

        model.zero_grad()
        # optimizer.zero_grad()
        # 使用了sigmoid，所以是BCELoss
        pred_y = torch.sigmoid(model(user_item).squeeze(1))
        loss = criterion(pred_y, label)
        loss.backward()
        optimizer.step()

        pred.extend(pred_y.tolist())
        target.extend(label.tolist())
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            print('train_loss:', total_loss / (i+1))
            # print("logloss",log_loss(target,pred))

    #  这里可能需要计算logloss
    # print("end train....")
    loss2 = total_loss / (i+1)
    all_loss = log_loss(target, pred)
    # print("=============",all_loss)
    return loss2


# CTR 预测的评估， 返回NDCG和HG

def test(model, optimizer, data_loader, criterion, device, log_interval=1000):
    # model.eval()
    pass


def test_roc(model, data_loader):
    model.eval()
    targets, predicts = list(), list()
    total_loss = 0
    with torch.no_grad():
        for fields, target in tqdm.tqdm(
                data_loader, smoothing=0, mininterval=1.0):
            fields = fields.long()
            target = target.float()
            fields, target = fields.cuda(), target.cuda()
            y = torch.sigmoid(model(fields).squeeze(1))

            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    #         这里可能有错误，criterion 不知道能不能计算两个list
    # roc, mse
    #
    return roc_auc_score(targets, predicts), log_loss(targets, predicts)


# 跑frappe和ml-tag实验，AUC和RMSE
def main2(dataset_name,
          model_name,
          epoch,
          learning_rate,
          batch_size,
          weight_decay,
          save_dir,
          path):
    field_dims, trainLoader, validLoader, testLoader = get_avazu_dataset(batch_size=batch_size)
    print("trainLoader", len(trainLoader))
    print("validLoader", len(validLoader))
    print("testLoader", len(testLoader))

    print(field_dims)
    print(sum(field_dims))

    # 路径
    time_fix = time.strftime("%d%H%M%S", time.localtime())

    # wdl pretrain
    for K in [10]:  # latent factors
    # fout = open(paths+"/logs.p","a+")
        paths = os.path.join(save_dir, dataset_name, model_name, str(K))
        if not os.path.exists(paths):
            os.makedirs(paths)
        with open(paths + f"/{model_name}logs2_{time_fix}.p", "a+") as fout:
            # 记录配置
            fout.write("Batch_size:{}\tlearning_rate:{}\tStartTime:{}\tweight_decay:{}\n"
                       .format(batch_size, learning_rate, time.strftime("%d%H%M%S", time.localtime()), weight_decay))
            # [256, 128, 64, 32, 16, 8, 4]:
            print("Start train -- K : {}".format(K))
            #
            # criterion = torch.nn.BCEWithLogitsLoss()  # 没有使用sigmoid,所以需要用这个
            criterion = torch.nn.BCELoss()
            # criterion = torch.nn.BCELoss() # 结果是0和1 , 所以是bceloss,
            model = get_model(
                name=model_name,
                field_dims=field_dims,
                embed_dim=K,).cuda()

            print(count_params(model))

            # TODO: (1) load pretrained parameters; (2)

            # if torch.cuda.is_available():
            #     model.to(DEVICE)
            # if torch.cuda.device_count()>1:
            #     print(f"Start train the {model_name} model in several GPUs")
            #     model = nn.DataParallel(model)

            optimizer = torch.optim.Adam(
                params=model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay)

            # scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True, patience=6)

            # 初始化EarlyStopping
            # print("初始化 early_stopping")
            early_stopping = EarlyStopping(patience=8, verbose=True, prefix=path)

            val_auc_best = 0
            auc_index_record = ""  # 记录当前的值

            val_loss_best = 1000
            loss_index_record = ""  # 记录当前的值

            scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=4)
            # scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True, patience=4)
            for epoch_i in range(epoch):
                print(model_name, K, epoch_i)

                start = time.time()

                train_loss = train(model, optimizer, trainLoader, criterion)

                # valid
                val_auc, val_loss = test_roc(model, validLoader)

                # test
                test_auc, test_loss = test_roc(model, testLoader)

                # scheduler.step(val_auc)
                scheduler.step(val_loss)
                # auc, test_mse = test_roc(model, testLoader)
                # 调整学习率
                end = time.time()
                # writer.add_scalar(f"train/loss_{K}", train_loss, epoch_i)
                # writer.add_scalar(f"val/loss-{K}" + str(), val_loss, epoch_i)
                # writer.add_scalar(f"val/auc-{K}", val_auc, epoch_i)
                # if val_auc > val_auc_best:
                #     # 直接保存完整模型，因为我需要embedding，来训练其他模型
                #     # torch.save({"state_dict": model.state_dict(), "best_auc": val_auc_best},
                #     #            paths + f"/{model_name}_final_{K}_{time_fix}.pt")
                #     pass
                    # torch.save(model,paths+f"/{model_name}_best_auc_{K}_{time_fix}.pkl")

                if val_auc > val_auc_best:
                    val_auc_best = val_auc
                    auc_index_record = "epoch_i:{}\t{:.6f}\t{:.6f}".format(epoch_i,test_auc,test_loss)

                if val_loss < val_loss_best:
                    val_loss_best = val_loss
                    loss_index_record = "epoch_i:{}\t{:.6f}\t{:.6f}".format(epoch_i,test_auc,test_loss)

                print(
                    "Train  K:{}\tEpoch:{}\ttrain_loss:{:.6f}\tval_loss:{:.6f}\tval_auc:{:.6f}\ttime:{:.6f}\ttest_loss:{:.6f}\ttest_auc:{:.6f}\n"
                    .format(K, epoch_i, train_loss, val_loss, val_auc, end - start,test_loss,test_auc))


                fout.write(
                    "Train  K:{}\tEpoch:{}\ttrain_loss:{:.6f}\tval_loss:{:.6f}\tval_auc:{:.6f}\ttime:{:.6f}\ttest_loss:{:.6f}\ttest_auc:{:.6f}\n"
                    .format(K, epoch_i, train_loss, val_loss, val_auc, end - start,test_loss,test_auc))

                # 判断是否早停止
                early_stopping(val_auc)
                if early_stopping.early_stop:
                    print("Early stopping")
                    # torch.save({"state_dict": model.state_dict(), "es_auc": val_auc, "es_loss": val_loss},
                    #            paths + f"/{model_name}_es_{K}_{time_fix}.pt")
                    break

            print("Test:{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n"
                  .format(K, val_auc, val_auc_best, val_loss, val_loss_best,test_loss,test_auc))

            fout.write("Test:{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n"
                       .format(K, val_auc, val_auc_best, val_loss, val_loss_best,test_loss,test_auc))

            # 记录最好的两组结果
            fout.write("auc_best:\t{}\nloss_best:\t{}".format(auc_index_record, loss_index_record))

            # 保存模型参数
            #
            # torch.save({"statee_dict":model.state_dict()},paths+f"/{model_name}_{K}.pt")
            # torch.save({"state_dict":model.state_dict(),"best_auc":val_auc_best}, paths + f"/{model_name}_final_{K}_{time_fix}.pt")
            # writer.close()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子

if __name__ == '__main__':

    # setup_seed(20)


    #
    # field_dims, trainLoader, validLoader, testLoader = get_dataloader(dataset="ml-tag")
    # print(field_dims,len(trainLoader))

    #  CUDA_VISIBLE_DEVICES=3 python evalCo.py --sample 2 --model_name coffm --mains 2 --epoch 10
    # CUDA_VISIBLE_DEVICES=1 python eval.py --model_name ncf2
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='avazu_base')
    parser.add_argument('--path', default="avazu", help="")
    parser.add_argument('--model_name', default='fm')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0', help="cuda:0")
    parser.add_argument('--save_dir', default='chkpt_avazu0318')
    parser.add_argument('--choice', default=0,type=int)
    args = parser.parse_args()

    if args.choice == 0:
        model_names = ["lr"] * 3

    elif args.choice == 1:
        model_names = ["nfm","dnn"] * 3

    elif args.choice == 2:
        # 分层
        model_names = ["fm"] * 3

    elif args.choice == 3:
        # 跑一下HirFM的
        model_names = ["afm"] * 3
    elif args.choice == 4:
        #     试一下FM
        # model_names = ["ipnn"] * 4
        model_names = ["opnn"] * 4


    elif args.choice == 5:
        # PlainDNN
        model_names = ["dfm"] * 4
    elif args.choice == 6:
        # hir DNN，对特征进行attention
        model_names = ["dcn"] * 6

    elif args.choice == 7:
        # hir DNN，对特征进行attention
        model_names = ["xdfm"] * 6


    elif args.choice == 8:
        model_names = ["wdl"] * 6

    #     CUDA_VISIBLE_DEVICES=1 python main_valid_base.py --choice 10
    elif args.choice == 9:
        model_names = ["dnn"] * 6

    elif args.choice == 10:
        model_names = ["fibinet"] * 6
    elif args.choice == 11:
        model_names = ["afn"] * 6

    elif args.choice == 12:
        model_names = ["autoint"] * 6

    elif args.choice == 13:
        model_names = ["ffm"] * 6

    elif args.choice == 14:
        model_names = ["wdl"] * 6

    elif args.choice == 15:
        model_names = ["cin"] * 6

    elif args.choice == 16:
        model_names = ["dcn"] * 6

    elif args.choice == 17:
        model_names = ["ffm"] * 6

    elif args.choice == 18:
        model_names = ["xdfm"] * 6







    print(model_names)

    for name in model_names:  # TODO
        main2(dataset_name=args.dataset_name,
              model_name=name,
              epoch=args.epoch,
              learning_rate=args.learning_rate,
              batch_size=args.batch_size,
              weight_decay=args.weight_decay,
              save_dir=args.save_dir,
              path=args.path)

