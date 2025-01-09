# -*- coding: utf-8 -*-
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models import *

import numpy as np

import ast


def find_min_in_column(column):
    # Flatten the lists and filter out empty lists
    flattened_values = [item for sublist in column for item in sublist if sublist]
    
    # If there are no valid items, return np.nan or your choice to handle such cases
    if not flattened_values:
        return np.nan
    
    # Return the minimum value from the flattened list
    return min(flattened_values)

def flatten(nested_list):
    for item in nested_list:
        if isinstance(item, list):  # 检查元素是否还是一个列表
            yield from flatten(item)
        else:
            yield item

if __name__ == "__main__":
    train_data = pd.read_csv('~/userlm/data/ml1m_test_sample.csv')
    # test_data = pd.read_csv('./ml1m_test_sample.txt')
    # target = ['label']
    # print(ast.literal_eval(train_data['genres'][0])[0])
    # train_data['genres_'] = train_data['genres'].apply(ast.literal_eval)
    print(train_data['item_id'].min())

    for fea in ['uni_seq_genres', 'genres', 'sequence_item_ids']:
        print(fea)
        fea_list = train_data[fea].apply(ast.literal_eval)
        print("min", min(list(flatten(fea_list))))
        print("max", max(list(flatten(fea_list))))





