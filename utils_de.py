#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Author:wangfy
@project:PNNConvModel
@Time:2020/9/8 7:01 下午
'''

#
FIELD_NUMS = [49,101,126,45,223,118,84,76,95,9,
        30,40,75,1458,555,193949,138801,306,19,11970,
        634,4,42646,5178,192773,3175,27,11422,181075,11,
        4654,2032,5,189657,18,16,59697,86,45571]




# 16 32 64 128 256 512 1024 2048 4096 8192 16384

def load_trained_embedding(from_model,to_model):
    """
    :param from_model:
    :param to_model:
    :return: model with trained params
    """
    model_dict = to_model.state_dict()
    state_dict_trained = {name: param for name, param in from_model.named_parameters() if name in model_dict.keys()}
    model_dict.update(state_dict_trained)
    to_model.load_state_dict(model_dict)
    return to_model


def count_params(model):
    params = sum(param.numel() for param in model.parameters())
    return params


if __name__ == '__main__':
    print(sum(FIELD_NUMS))