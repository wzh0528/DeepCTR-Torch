# -*- coding: utf-8 -*-

import sys

sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                                  get_feature_names)
from deepctr_torch.models import DeepFM, DIN, DIEN

import ast


def padding(lst, padding_num, max_len):
    paddinged = lst
    paddinged.extend([padding_num] * (max_len - len(lst)))
    return paddinged

def padding_and_increment(lst, padding_num, max_len):
    incremented = [x + 1 for x in lst]
    incremented.extend([padding_num] * (max_len - len(lst)))
    return np.array(incremented)

def feature_process(data):
    # for fea in ['genres', 'uni_seq_item_id', 'uni_seq_genres']:
    max_len_dict = {}
    for fea in ['genres']:
        data[fea] = data[fea].apply(ast.literal_eval)
        max_len_dict[fea] = data[fea].apply(len).max()
        data[fea] = data[fea].apply(lambda lst: padding_and_increment(lst, 0, max_len_dict[fea]))
    
    for fea in ['sequence_item_ids']:
        data[fea] = data[fea].apply(ast.literal_eval)
        data[fea + '_length'] = data[fea].apply(len)
        max_len_dict[fea] = data[fea + '_length'].max()
        data[fea] = data[fea].apply(lambda lst: padding(lst, 0, max_len_dict[fea]))

    return max_len_dict

def get_feature_column():
    feature_columns = [SparseFeat('item_id', 3953, embedding_dim=8),
                       SparseFeat('item_genres', 19, embedding_dim=8, use_bag=True, baglen=6)]

    feature_columns += [VarLenSparseFeat(SparseFeat('hist_item_id', 3953, embedding_dim=8), 511, length_name="seq_length")]
    #                     VarLenSparseFeat(SparseFeat('hist_item_gender', 2 + 1, embedding_dim=8, use_bag=True, baglen=4), 4 * 4, length_name="seq_length")]

    return feature_columns

if __name__ == "__main__":
    train_data = pd.read_csv('~/userlm/data/ml1m_test_sample.csv')
    test_data = pd.read_csv('~/userlm/data/ml1m_test_sample.csv')

    target = ['label']

    _ = feature_process(train_data)
    _ = feature_process(test_data)
    # print(max_len_dict['genres'])

    # print(type(train_data['genres']))
    # print(train_data['genres'])
    # print(type(train_data['genres']))

    # print(train_data['genres'].shape)
    # print(train_data['genres'].to_list())
    # print(train_data['item_id'])

    feature_columns = get_feature_column()
    behavior_feature_list = ["item_id"]
    print(feature_columns)
    print(train_data.keys())

    feature_dict = { 'item_id': 'item_id', 'item_genres': 'genres', 'hist_item_id': 'sequence_item_ids', 'seq_length': 'sequence_item_ids_length'}
    feature_names = get_feature_names(feature_columns)
    train_model_input = {name: np.array(train_data[feature_dict[name]].to_list()) for name in feature_names}  
    train_label = train_data[target].values
    test_model_input = {name: np.array(test_data[feature_dict[name]].to_list()) for name in feature_names} 
    test_label = test_data[target].values

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    # model = DeepFM(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns,
    #                task='binary',
    #                l2_reg_embedding=1e-5, device=device)

    # model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True)
    model = DIEN(feature_columns, behavior_feature_list,
                 dnn_hidden_units=[32, 64, 32], dnn_dropout=0.9, gru_type="AUGRU", use_negsampling=True, device=device)

    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )

    history = model.fit(train_model_input, train_label, batch_size=128, epochs=1, verbose=2,
                        validation_split=0.2)
    pred_ans = model.predict(test_model_input, 256)
    print("")
    print("test LogLoss", round(log_loss(test_label, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test_label, pred_ans), 4))
