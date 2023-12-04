from typing import Any
import torch
import numpy as np
import tools
import compress_pickle as pickle
from os.path import exists, join as osjoin
from abc import abstractmethod
from .label_generator import LabelGenerator
from .utils import Normalization, unroll


def label_func_min(pred_window:np.ndarray, pred_window_mask:np.ndarray):
    assert(pred_window.shape == pred_window_mask.shape)
    invalid_flag = pred_window.max() + 1
    pred_window = pred_window * pred_window_mask + invalid_flag * np.logical_not(pred_window_mask)
    label = np.min(pred_window, axis=1)
    sequence_mask = label != invalid_flag
    return sequence_mask, label

def label_func_max(pred_window:np.ndarray, pred_window_mask:np.ndarray):
    assert(pred_window.shape == pred_window_mask.shape)
    invalid_flag = pred_window.min() - 1
    pred_window = pred_window * pred_window_mask + invalid_flag * np.logical_not(pred_window_mask)
    label = np.max(pred_window, axis=1)
    sequence_mask = label != invalid_flag
    return sequence_mask, label

def vent_label_func(pred_window:np.ndarray, pred_window_mask:np.ndarray):
    sequence_mask, label = label_func_max(pred_window, pred_window_mask)
    label = np.clip(label, -1, 1) # 1,2 -> 1
    return sequence_mask, label

class DataGenerator():
    def __init__(self, n_fea, limit_idx=[], forbidden_idx=[]) -> None:
        if len(limit_idx) == 0:
            self._avail_idx = [idx for idx in range(n_fea) if idx not in forbidden_idx]
        else:
            self._avail_idx = [idx for idx in range(n_fea) if (idx in limit_idx) and (idx not in forbidden_idx)]

    @property
    def avail_idx(self):
        return self._avail_idx

class DynamicDataGenerator(DataGenerator):
    def __init__(self, window_points, 
                 n_fea, 
                 label_generator: LabelGenerator, 
                 label_func,
                 target_idx, 
                 limit_idx=[], 
                 forbidden_idx=[], 
                 norm:Normalization=None
    ) -> None:
        super().__init__(n_fea, limit_idx, forbidden_idx)
        self.norm = norm
        self.label_func = label_func
        self.target_idx = target_idx
        self.window = window_points # how many points we should look forward
        self.label_gen = label_generator
    
    def __call__(self, _data:np.ndarray, seq_lens:np.ndarray) -> dict:
        '''
        data: (batch, n_fea, seq_lens)
        mask: (batch, seq_lens)
        return: 
            mask(batch, seq_lens), label(batch, seq_lens, n_cls)
        '''

        mask = tools.make_mask((_data.shape[0], _data.shape[2]), seq_lens) # (batch, seq_lens)
        data = _data.copy()
        target = data[:, self.target_idx, :]
        data = data[:, self.avail_idx, :]
        # 将target按照时间顺序平移
        for idx in range(target.shape[1]-1): # 最后一个格子预测一格
            stop = min(data.shape[2], idx+self.window)
            pred_window = target[:, idx+1:stop] # seq_len的最后一个格子是无效的
            pred_window_mask = mask[:, idx+1:stop]
            sequence_mask, label = self.label_func(pred_window, pred_window_mask) # (batch, window) -> (batch, )
            target[:, idx] = label
            mask[:, idx] = sequence_mask
        # 将target转化为标签
        label = self.label_gen(target) * mask[..., None]
        if self.norm is not None:
            data = self.norm(data, self.avail_idx)
        
        result =  {'data': data, 'mask': mask, 'label': label}
        return result
    
class SliceDataGenerator(DataGenerator):
    '''
    生成每个时间点和预测窗口的标签, 并进行展开
    '''
    def __init__(self, 
                 window_points, 
                 n_fea, 
                 label_generator: LabelGenerator, 
                 label_func,
                 target_idx, 
                 limit_idx=[], 
                 forbidden_idx=[], 
                 norm:Normalization=None
    ) -> None:
        super().__init__(n_fea, limit_idx, forbidden_idx)
        self.norm = norm
        self.label_func = label_func
        self.target_idx = target_idx
        self.window = window_points # 向前预测多少个点内的ARDS
        self.label_gen = label_generator
        self.slice_len = None
    
    def __call__(self, _data:np.ndarray, seq_lens:np.ndarray) -> dict:
        '''
        data: (batch, n_fea, seq_lens)
        mask: (batch, seq_lens)
        return: 
            label(n_slice, dim_target)
        '''

        mask = tools.make_mask((_data.shape[0], _data.shape[2]), seq_lens) # (batch, seq_lens)
        data = _data.copy()
        target = data[:, self.target_idx, :]
        data = data[:, self.avail_idx, :]
        self.slice_len = data.shape[0]
        # 将target按照时间顺序平移
        for idx in range(target.shape[1]-1): # 最后一个格子预测一格
            stop = min(data.shape[2], idx+self.window)
            pred_window = target[:, idx+1:stop] # seq_len的最后一个格子是无效的
            pred_window_mask = mask[:, idx+1:stop]
            sequence_mask, label = self.label_func(pred_window, pred_window_mask) # (batch, window) -> (batch, )
            target[:, idx] = label
            mask[:, idx] = np.logical_and(mask[:, idx], sequence_mask)
        # 将target转化为标签
        label = self.label_gen(target) * mask[..., None]
        if self.norm is not None:
            data = self.norm(data, self.avail_idx)

        # 转化为slice
        data, label = unroll(data, mask), unroll(label, mask)
        result = {'data': data, 'mask': mask, 'label': label, 'slice_len': self.slice_len}
        return result

    def restore_from_slice(self, x:np.ndarray):
        '''make_slice的反向操作, 保证顺序不会更改'''
        pass

class StaticDataGenerator(DataGenerator):
    '''
    生成每个时间点和预测窗口的标签，但是不展开时间轴
    '''
    def __init__(self, 
                 start_point, 
                 window_points, 
                 n_fea, 
                 label_generator: LabelGenerator, 
                 label_func,
                 target_idx, 
                 limit_idx=[], 
                 forbidden_idx=[], 
                 norm:Normalization=None
    ) -> None:
        super().__init__(n_fea, limit_idx, forbidden_idx)
        self.norm = norm
        self.label_func = label_func
        self.target_idx = target_idx
        self.window = window_points
        self.start_point = start_point
        self.label_gen = label_generator
    
    def __call__(self, _data:np.ndarray, seq_lens:np.ndarray) -> dict:
        '''
        data: (batch, n_fea, seq_lens)
        mask: (batch, seq_lens)
        return: 
            mask(new_batch, window), label(new_batch, n_cls), data(new_batch, avail_idx, window)
        '''

        mask = tools.make_mask(_data.shape[[0,2]], seq_lens) # (batch, seq_lens)
        mask = mask[:, self.start_point:min(target.size(1), self.start_point+self.window)] # (batch, window)
        seq_mask = np.max(mask, axis=1) # (batch, ) throw away all zero sequences
        mask = mask[seq_mask, :]

        data = _data.copy()
        target = data[seq_mask, self.target_idx, :]
        data = data[seq_mask, self.avail_idx, :]
        # calculate target in prediction window
        target = target * mask
        pred_window = target[:, min(target.size(1), self.window)]
        pred_window_mask = mask[:, min(target.size(1), self.window)]
        sequence_mask, target = self.label_func(pred_window, pred_window_mask) # (batch, window) -> (batch, )
        mask[np.logical_not(sequence_mask), :] = False
        # target -> label
        label = self.label_gen(target) * mask[: None]
        if self.norm is not None:
            data = self.norm(data, self.avail_idx)

        result = {'data': data, 'mask': mask, 'label': label}
        return result