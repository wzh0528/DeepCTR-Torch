# -*- coding: utf-8 -*-

import sys

sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import argparse
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                                  get_feature_names)
from deepctr_torch.models import DeepFM, DIN, DIEN

import ast


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, choices=["ml-20m", "ml-1m", "books"], default="ml-1m"
    )
    parser.add_argument(
        "--model", type=str, choices=["din", "dien", "deepfm"], default="din"
    )
    args = parser.parse_args()
    return args

def padding(lst, padding_num, max_len):
    result = lst
    result.extend([padding_num] * (max_len - len(lst)))
    return result

def padding_and_increment(lst, padding_num, max_len):
    result = [x + 1 for x in lst]
    result.extend([padding_num] * (max_len - len(lst)))
    return result

def padding_increment_and_flatten(nested_list, padding_num, max_inner_len, max_outer_len):
    # 处理内层列表：每个元素加1并填充0到最大内层长度
    processed_inner_lists = [
        [x + 1 for x in inner] + [padding_num] * (max_inner_len - len(inner))
        for inner in nested_list
    ]
    # 填充外层列表：填充0列表至最大外层长度
    processed_inner_lists.extend([[padding_num] * max_inner_len] * (max_outer_len - len(processed_inner_lists)))
    # 扁平化为一维数组
    flattened = [item for sublist in processed_inner_lists for item in sublist]
    return flattened

def find_global_max_len(nested_lists):
    max_inner_len = 0
    max_outer_len = 0
    for outer_list in nested_lists:
        max_outer_len = max(max_outer_len, len(outer_list))
        for inner_list in outer_list:
            max_inner_len = max(max_inner_len, len(inner_list))
    return max_inner_len, max_outer_len

def feature_process(data):
    # for fea in ['genres', 'uni_seq_item_id', 'uni_seq_genres']:
    print("feature processing...")
    max_len_dict = {}
    for fea in ['genres']:
        data[fea] = data[fea].apply(ast.literal_eval)
        max_len_dict[fea] = data[fea].apply(len).max()
        print(max_len_dict[fea])
        data[fea] = data[fea].apply(lambda lst: padding_and_increment(lst, 0, max_len_dict[fea]))
    
    for fea in ['sequence_item_ids']:
        data[fea] = data[fea].apply(ast.literal_eval)
        data[fea + '_length'] = data[fea].apply(len)
        max_len_dict[fea] = data[fea + '_length'].max()
        print(max_len_dict[fea])
        data[fea] = data[fea].apply(lambda x: padding(x, 0, max_len_dict[fea]))
    
    for fea in ['uni_seq_genres']:
        data[fea] = data[fea].apply(ast.literal_eval)
        max_len_dict[fea + '_inner'], max_len_dict[fea + '_outter'] = find_global_max_len(data[fea])
        print(max_len_dict[fea + '_inner'])
        print(max_len_dict[fea + '_outter'])
        data[fea] = data[fea].apply(lambda x: padding_increment_and_flatten(x, 0, max_len_dict[fea + '_inner'], max_len_dict[fea + '_outter']))

    return max_len_dict

def get_feature_column():
    feature_columns = [SparseFeat('item_id', 3953, embedding_dim=18),
                       SparseFeat('item_genres', 19, embedding_dim=18, use_bag=True, baglen=6)]

    feature_columns += [VarLenSparseFeat(SparseFeat('hist_item_id', 3953, embedding_dim=18), 511, length_name="seq_length"),
                        VarLenSparseFeat(SparseFeat('hist_item_genres', 19, embedding_dim=18, use_bag=True, baglen=6), 6 * 511, length_name="seq_length")]

    return feature_columns

if __name__ == "__main__":
    args = get_args()
    print('data_loading...')
    train_data = pd.read_csv('~/userlm/data/ml1m_test_sample.csv')
    # train_data = train_data.sample(frac=0.5, random_state=42)

    target = ['label']

    feature_process(train_data)

    feature_columns = get_feature_column()
    behavior_feature_list = ["item_id", "item_genres"]
    # behavior_feature_list = ["item_id"]

    feature_dict = { 'item_id': 'item_id', 'item_genres': 'genres', 
                    'hist_item_id': 'sequence_item_ids',  'hist_item_genres': 'uni_seq_genres',
                    'seq_length': 'sequence_item_ids_length'}
    feature_names = get_feature_names(feature_columns)
    train_model_input = {name: np.array(train_data[feature_dict[name]].to_list()) for name in feature_names}  
    train_label = train_data[target].values

    # for name in feature_names:
    #     print(name)
    #     print(train_model_input[name][0]) 
    # print(train_label[0])
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    if args.model == 'deepfm':
        model = DeepFM(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns,
                    task='binary',
                    l2_reg_embedding=1e-5, device=device)
    elif args.model == 'din':
        model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True)
    elif args.model == 'dien':
        model = DIEN(feature_columns, behavior_feature_list,
                    dnn_hidden_units=[32, 64, 32], dnn_dropout=0.9, gru_type="AUGRU", use_negsampling=False, device=device)

    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )
    
    print('training...')
    history = model.fit(train_model_input, train_label, batch_size=32, epochs=10, verbose=2)
