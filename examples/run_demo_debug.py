import sys

sys.path.insert(0, '..')

import numpy as np
import torch
from deepctr_torch.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                                  get_feature_names)
from deepctr_torch.models import DIN, DeepFM, DIEN, WDL


def get_xy_fd():
    feature_columns = [SparseFeat('user', 3, embedding_dim=4), SparseFeat('gender', 2, embedding_dim=4),
                       SparseFeat('item', 3 + 1, embedding_dim=4), 
                       SparseFeat('item_gender', 2 + 1, embedding_dim=4, use_bag=True, baglen=2),
                       DenseFeat('score', 1)]
    
    feature_columns += [VarLenSparseFeat(SparseFeat('hist_item', 3 + 1, embedding_dim=4), 2, length_name="seq_length"),
                        # VarLenSparseFeat(SparseFeat('item_gender', 2 + 1, embedding_dim=4), 4, length_name="seq_length"),               
                        VarLenSparseFeat(SparseFeat('hist_item_gender', 1 + 1, embedding_dim=4, use_bag=True, baglen=2), 2 * 2, length_name="seq_length")]
    behavior_feature_list = ["item", "item_gender"]
    uid = np.array([0, 1])
    ugender = np.array([0, 1])
    iid = np.array([1, 2])  # 0 is mask value
    # igender =  np.array([2,1,0])  # 0 is mask value
    igender = np.array([[1,1],[1,0]])
    score = np.array([0.1, 0.2])

    

    hist_iid = np.array([[1, 0], [2, 1]])
    # hist_igender = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0]])
    hist_igender = np.array([[1, 0, 0, 0], [1, 1, 1, 1]])
    behavior_length = np.array([1, 2])

    feature_dict = {'user': uid, 'gender': ugender, 'item': iid, 'item_gender': igender,
                    'hist_item': hist_iid, 'hist_item_gender': hist_igender, 'score': score,
                    "seq_length": behavior_length}
    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    y = np.array([1, 0])

    return x, y, feature_columns, behavior_feature_list


if __name__ == "__main__":
    x, y, feature_columns, behavior_feature_list = get_xy_fd()
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    # model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True)
    # model = DeepFM(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns,
    #                 task='binary',
    #                 l2_reg_embedding=1e-5, device=device)
    model = WDL(linear_feature_columns=behavior_feature_list, dnn_feature_columns = feature_columns, device=device)

    model.compile('adagrad', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    history = model.fit(x, y, batch_size=2, epochs=10, verbose=2, validation_split=0.0, shuffle=False)
