from typing import Any
import torch
import numpy as np
import tools
import compress_pickle as pickle
from os.path import exists, join as osjoin
from abc import abstractmethod

class Normalization():
    def __init__(self, norm_dict:dict, total_keys:list) -> None:
        self.means = np.asarray([norm_dict[key]['mean'] for key in total_keys])
        self.stds = np.asarray([norm_dict[key]['std'] for key in total_keys])

    def restore(self, in_data, selected_idx):
        # restore de-norm data
        # in_data: (..., n_selected_fea, seqs_len)
        means, stds = self.means[selected_idx], self.stds[selected_idx] + 1e-4
        out = in_data * stds + means
        return out

    def __call__(self, in_data, selected_idx) -> Any:
        # in_data: (..., n_selected_fea, seqs_len)
        means, stds = self.means[selected_idx], self.stds[selected_idx] + 1e-4
        out = (in_data - means[:, None]) / (stds[:, None])
        return out
    
def Collect_Fn(data_list:list):
    result = {}
    result['data'] = np.stack([d['data'] for d in data_list], axis=0)
    result['length'] = np.asarray([d['length'] for d in data_list], dtype=np.int32)
    return result

def unroll(x:np.ndarray, mask:np.ndarray):
    # x: (batch, n_fea, seqs_len) or (batch, seqs_len) or (batch, seqs_len, n_cls)
    # mask: (batch, seqs_len)
    assert(len(x.shape) <= 3 and len(mask.shape) == 2)
    if len(x.shape) == 2:
        return x.flatten()[mask.flatten()]
    elif x.shape[2] == mask.shape[1]:
        batch, n_fea, seqs_len = x.shape
        x = np.transpose(x, (0, 2, 1)).reshape((batch*seqs_len, n_fea))
        return x[mask.flatten(), :]
    elif x.shape[1] == mask.shape[1]:
        batch, seqs_len, n_cls = x.shape
        x = x.reshape((batch*seqs_len, n_cls))
        return x[mask.flatten(), :]
    else:
        assert(0)

def map_func(a:np.ndarray, mapping:dict):
    '''
    mapping: key=origin_idx, value=target_idx
    '''
    a_shape = list(a.shape)
    n_targets = len(np.unique(list(mapping.values())))
    a_shape[-1] = n_targets
    result = np.zeros(tuple(a_shape))
    for k, v in mapping.items():
        result[..., v] += a[..., k]
    return result

def cal_label_weight(n_cls, label:np.ndarray):
    '''
    Get the weight of n_cls inversely proportional to the number.
    label: (..., n_cls)
    return: (n_cls,)
    '''
    hard_label = np.argmax(label, axis=-1).flatten()
    weight = np.asarray([np.mean(hard_label == c) for c in range(n_cls)])
    weight = 1 / weight
    weight = weight / np.sum(weight)
    return weight

