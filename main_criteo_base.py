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

# sys.path.append("..")


import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.DFM import DeepFactorizationMachineModel,DeepFMGate
from model.LR import LogisticRegression
from model.FM import FactorizationMachineModel
from model.WD import WideAndDeepModel,WDLGate
from model.AFM import AttentionalFactorizationMachineModel,AFMWithGate
from model.GateWithAFM import GateWithAFMModel
from model.NFM import NeuralFactorizationMachineModel,GELNFM,NFMInterGate
from model.PlainDNN import PlainDNN
from model.PNN import ProductNeuralNetworkModel
from model.combine_model.HirAttNFM import HANFM,HANFMSelf
from model.combine_model.WideAttFM import WideAttFM
from model.PlainDNN import PlainDNN,HirPlainDNN
from model.AFM import HirAFM
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
from dataset.criteo import get_criteo_dataset_train, get_criteo_dataset_loader
from dataset.criteo_valid import get_criteo_dataset_loader_valid
from utils_de import *
from utils.earlystoping import EarlyStopping

PRE_trained_path = "./chkpt/criteo/wdwd_best_auc_pre_10.pkl"


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

    elif name[:4] == "gnfm":
        # gate*n + nfm
        # 目前而言，这个效果不好
        glunum = int(name[4:])
        return GELNFM(field_dims, embed_dim, mlp_layers, glunum=glunum, dropouts=(0.5, 0.5))

    elif name[:4] == "nfmg":
        # 这个目前不用了, gate 用在interaction后面。
        glunum = int(name[4:])
        to_model = NFMInterGate(field_dims, embed_dim, glunum=glunum, dropouts=(0.5, 0.5))
        return to_model

    elif name[:7] == "widenfm":
        # 分层，然后进行attention ，最后nfm
        glunum = int(name[7:])
        return HANFM(field_dims, embed_dim, mlp_layers, glunum=glunum, dropouts=(0.5, 0.5))

    elif name[:9] == "HANFMSelf":
        glunum = int(name[9:])
        return HANFMSelf(field_dims, embed_dim=embed_dim, mlp_layers=mlp_layers,
                         dropouts=(0.5, 0.5), glunum=glunum)
    elif name[:6] == "widefm":
        glunum = int(name[6:])
        return WideAttFM(field_dims,embed_dim,glunum=glunum)
    elif name[:2] == "fm":
        return FactorizationMachineModel(field_dims,embed_dim)

    elif name[:3] == "dnn":
        return PlainDNN(field_dims,embed_dim,mlp_layers=mlp_layers,dropout=0.5)

    elif name[:6] == "hirdnn":
        glunum = int(name[6:])
        return HirPlainDNN(field_dims,embed_dim,glunum=glunum,dropout=0.5)
    elif name[:3] == "dfm":
        return DeepFactorizationMachineModel(field_dims,embed_dim,dropout=0.5)
    elif name[:7] == "wideafm":
        glunum = int(name[7:])
        return HirAFM(field_dims,embed_dim,attn_size=32,glunum=glunum)

    else:
        raise ValueError('unknown model name: ' + name)



# evaluate function


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


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
    # device = torch.device(DEVICE)
    # data_path = "./Data/"

    # criteo_path = "train_sub100w.txt"
    # criteo_path = path

    # field_dims, trainLoader, validLoader = get_criteo_dataset_train(
    #     criteo_path, batch_size=batch_size)

    # field_dims, trainLoader, validLoader = get_criteo_dataset_loader(batch_size=batch_size)
    # 这里还应该有一个测试集合
    field_dims, trainLoader, validLoader, testLoader = get_criteo_dataset_loader_valid(batch_size=batch_size)
    # print("trainLoader", len(trainLoader))
    # print("validLoader", len(validLoader))
    # print("testLoader", len(testLoader))

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

            scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True, patience=4)
            for epoch_i in range(epoch):
                print("criteo",model_name, K, epoch_i)

                start = time.time()

                train_loss = train(model, optimizer, trainLoader, criterion)

                # valid
                val_auc, val_loss = test_roc(model, validLoader)

                # test
                test_auc, test_loss = test_roc(model, testLoader)

                scheduler.step(val_auc)
                # scheduler.step(val_loss)
                # auc, test_mse = test_roc(model, testLoader)
                # 调整学习率

                end = time.time()
                # writer.add_scalar(f"train/loss_{K}", train_loss, epoch_i)
                # writer.add_scalar(f"val/loss-{K}" + str(), val_loss, epoch_i)
                # writer.add_scalar(f"val/auc-{K}", val_auc, epoch_i)

                #
                # if val_auc > val_auc_best:
                #     # 直接保存完整模型，因为我需要embedding，来训练其他模型
                #     torch.save(model,paths+f"/{model_name}_best_auc_{K}_{time_fix}.pkl")

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

    setup_seed(20)


    #
    # field_dims, trainLoader, validLoader, testLoader = get_dataloader(dataset="ml-tag")
    # print(field_dims,len(trainLoader))

    #  CUDA_VISIBLE_DEVICES=3 python evalCo.py --sample 2 --model_name coffm --mains 2 --epoch 10
    # CUDA_VISIBLE_DEVICES=1 python eval.py --model_name ncf2
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='criteo_wide_1230')
    parser.add_argument('--path', default="train_sub100w.txt", help="")
    parser.add_argument('--model_name', default='fm')
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0', help="cuda:0")
    parser.add_argument('--save_dir', default='chkpt_cri_es')
    parser.add_argument('--choice', default=0,type=int)
    args = parser.parse_args()
    # lr,afm,dfm,deepfm,wdl,
    # 运行一下AFM，NFM
    # "wdgate"
    # for name in ["gwdl1","gwdl2","gafm1","gafm2","gnfm1","gnfm2","gdfm1"]:
    # for name in ["ipnn","gwdl1","gwdl2", "gdfm1","gdfm2","gafm1","gafm2","fm","afm","wdl","nfm","dfm", "gnfm1","gnfm2", "gwdl1","gwdl2", "gdfm1","gdfm2", "gafm1","gafm2"]:
    # TODO:(1) WDL 3次 对
    # TODO:(2) NFM 3次 对
    # TODO:(3) dfm 3次 对
    # TODO:(4) PNN 3次
    # TODO:(5) AFM 3次 对
    # TODO:(6) FM 3次
    # TODO:(7) LR 3次 对

    #  Gate： 1
    # TODO:(1) gafm 3次
    # TODO:(2) gnfm 3次 完成
    # TODO:(3) gdfm 3次
    # TODO:(4) gwdl 3次 完成

    #  Gate： 2
    #     # TODO:(1) gafm 3次
    #     # TODO:(2) gnfm 3次 完成
    #     # TODO:(3) gdfm 3次
    #     # TODO:(3) gwdl 3次 完成

    #  Gate： 3
    # TODO:(1) gafm 3次
    # TODO:(2) gnfm 3次 对
    # TODO:(3) gdfm 3次
    # TODO:(3) gwdl 3次 对

    # model_names = ["HANFMSelf" + str(glunum) for glunum in range(1,4,1)] * 4
    # model_names = ["hanfm"+str(glunum) for glunum in range(1,5,1)]
    # model_names = ["gnfm1", "gnfm1", "gnfm1", "gnfm2", "gnfm2", "gnfm2", "gnfm3", "gnfm3", "gnfm3","nfm","nfm","nfm",]
    # model_names = ["HANFMSelf" + str(glunum) for i in range(1,4,1)]
    # model_names = ["HANFMSelf" + str(glunum) for i in range(1,4,1)]

    # for name in ["ipnn","gwdl1","gwdl2", "gdfm1","gdfm2","gafm1","gafm2","fm","afm","wdl","nfm","dfm", "gnfm1","gnfm2", "gwdl1","gwdl2", "gdfm1","gdfm2", "gafm1","gafm2"]:
    # for name in ["lr","lr","lr","nfm","nfm","nfm","wdl","wdl","wdl","dfm","dfm","dfm"]:  # 完成10.14
    # for name in ["gwdl1","gwdl1","gwdl1","gwdl2","gwdl2","gwdl2","gnfm1","gnfm1","gnfm1","gnfm2","gnfm2","gnfm2"]: 完成
    if args.choice == 0:
        # embedding相同
        model_names = ["HANFMSelf" + str(glunum) for glunum in range(2, 9, 2)] * 3

    elif args.choice == 1:
        model_names = ["nfm"] * 6

    elif args.choice == 2:
        # 分层
        model_names = ["widenfm" + str(glunum) for glunum in range(2, 9, 2)] * 6

    elif args.choice == 3:
        # 跑一下HirFM的
        model_names = ["widenfm" + str(glunum) for glunum in [1]] * 6
    elif args.choice == 9:
        # 跑一下HirFM的
        model_names = ["widefm" + str(glunum) for glunum in [1,2]] * 6

    elif args.choice == 4:
        model_names = ["fm"] * 6
    elif args.choice == 5:
        # PlainDNN
        model_names = ["dnn"] * 6
    elif args.choice == 6:
        # hir DNN，对特征进行attention
        model_names = ["hirdnn" + str(glunum) for glunum in range(4, 7, 2)] * 3
        # model_names = ["hirdnn" + str(glunum) for glunum in [4]] * 6
    elif args.choice == 7:
        # hir DNN，对特征进行attention
        model_names = ["hirdnn" + str(glunum) for glunum in [2, 8]] * 3
    elif args.choice == 10:
        #     试一下FM
        model_names = ["widefm" + str(glunum) for glunum in [1,2]] * 3

    elif args.choice == 11:
        #     试一下FM
        model_names = ["wideafm" + str(glunum) for glunum in [3]] * 3
    elif args.choice == 12:
        #     试一下FMs
        model_names = ["widefm" + str(glunum) for glunum in [1]] * 2

    elif args.choice == 13:
        #     试一下FM
        model_names = ["widefm" + str(glunum) for glunum in [3]] * 2

    elif args.choice == 14:
        #     试一下FM
        model_names = [model_name + str(2) for model_name in ["widefm","wideafm","widenfm"]] * 3




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

