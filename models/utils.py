from typing import Any
import torch
import numpy as np
import tools
import compress_pickle as pickle
from os.path import exists
from abc import abstractmethod

def Collect_Fn(data_list:list):
    result = {}
    result['data'] = torch.as_tensor(np.stack([d['data'] for d in data_list], axis=0), dtype=torch.float32)
    result['length'] = np.asarray([d['length'] for d in data_list], dtype=np.int32)
    return result

class Normalization():
    def __init__(self, norm_dict:dict, total_keys:list) -> None:
        self.means = np.asarray([norm_dict[key]['mean'] for key in total_keys])
        self.stds = np.asarray([norm_dict[key]['std'] for key in total_keys])

    def restore(self, in_data, selected_idx):
        # restore de-norm data
        # in_data: (..., n_selected_fea, seqs_len)
        means, stds = self.means[selected_idx], self.stds[selected_idx] + 1e-4
        out = in_data * self.stds + self.means
        return out

    def __call__(self, in_data, selected_idx) -> Any:
        # in_data: (..., n_selected_fea, seqs_len)
        means, stds = self.means[selected_idx], self.stds[selected_idx] + 1e-4
        out = (in_data - means[:, None]) / (stds[:, None])
        return out

class DataGenerator():
    def __init__(self, n_fea, limit_idx=[], forbidden_idx=[], p_cache_file=None) -> None:
        self._avail_idx = [idx for idx in range(n_fea) if idx in limit_idx and idx not in forbidden_idx]
        self.p_cache_file = p_cache_file

    @property
    def avail_idx(self):
        return self._avail_idx
    
    def load_cache(self):
        if self.p_cache_file is not None and exists(self.p_cache_file):
            return pickle.load(self.p_cache_file)
        else:
            return None
    
    def save_cache(self, obj, force=False):
        if self.p_cache_file is not None:
            if exists(self.p_cache_file) and not force:
                return
            else:
                pickle.dump(obj, self.p_cache_file)
    
    @abstractmethod
    def __call__(self, _data:np.ndarray, seq_lens:list) -> Any:
        pass

class LabelGenerator():
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def __call__(self, slice) -> Any:
        pass

class LabelGenerator_4cls(LabelGenerator):
    def __init__(self, centers:list, soft_label=False, smoothing_band=50) -> None:
        super().__init__()
        assert(len(centers)==4)
        self.centers = centers
        self.soft_label = soft_label
        self.smoothing_band = smoothing_band

    def __call__(self, target:np.ndarray) -> Any:
        if self.soft_label:
            return self.label_4cls(self.centers, target, smooth_band=50)
        else:
            return self.label_4cls(self.centers, target, smooth_band=0)

    def label_4cls(self, centers:list, nums:np.ndarray, smooth_band=50):
        '''
        centers: 每个class的中心点, 需要是递增的, n_cls = len(centers)
        nums: 输入(in_shape,) 可以是任意的
        band: 在两个class之间进行线性平滑, band是需要平滑的总宽度
            当输入在band外时(靠近各个中心或者超过两侧), 是硬标签, 只有在band内才是软标签
        return: (..., len(centers)) 其中 (...) = nums.shape
        '''
        num_classes = len(centers)
        smoothed_labels = np.zeros((nums.shape + (num_classes,)))
        for i in range(num_classes-1):
            center_i = centers[i]
            center_j = centers[i+1]
            lower = 0.5*(center_i + center_j) - smooth_band/2
            upper = 0.5*(center_i + center_j) + smooth_band/2
            hard_i = np.logical_and(nums >= center_i, nums <= lower)
            hard_j = np.logical_and(nums < center_j, nums > upper)
            mask = np.logical_and(nums > lower, nums <= upper)
            if smooth_band > 0 and mask.any():
                diff = (nums - center_i) / (center_j - center_i)
                smooth_i = 1 - diff
                smooth_j = diff
                smoothed_labels[..., i][mask] = smooth_i[mask]
                smoothed_labels[..., i+1][mask] = smooth_j[mask]
            smoothed_labels[..., i][hard_i] = 1
            smoothed_labels[..., i+1][hard_j] = 1
        smoothed_labels[..., 0][nums <= centers[0]] = 1
        smoothed_labels[..., -1][nums > centers[-1]] = 1
        return smoothed_labels

class LabelGenerator_regression(LabelGenerator):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, target) -> Any:
        return target[..., None]

class DynamicDataGenerator(DataGenerator):
    '''
    生成每个时间点和预测窗口的标签，但是不展开时间轴
    '''
    def __init__(self, window_points, n_fea, label_generator: LabelGenerator, target_idx, limit_idx=[], forbidden_idx=[], norm:Normalization=None, p_cache_file=None) -> None:
        super().__init__(n_fea, limit_idx, forbidden_idx, p_cache_file)
        self.norm = norm
        self.target_idx = target_idx
        self.window = window_points # 向前预测多少个点内的ARDS
        self.label_gen = label_generator
    
    def __call__(self, _data:np.ndarray, seq_lens:np.ndarray) -> np.ndarray:
        '''
        data: (batch, n_fea, seq_lens)
        mask: (batch, seq_lens)
        return: 
            mask(batch, seq_lens), label(batch, seq_lens, n_cls)
        '''
        cache = self.load_cache()
        if cache is not None:
            return cache

        mask = tools.make_mask(_data.shape[[0,2]], seq_lens) # (batch, seq_lens)
        data = _data.copy()
        target = data[:, self.target_idx, :]
        data = data[:, self.avail_idx, :]
        # 将target按照时间顺序平移
        invalid_flag = target.max()
        for idx in range(target.shape[1]-1): # 最后一个格子预测一格
            stop = min(data.shape[1], idx+self.window)
            mat = mask[:, idx+1:stop] * target[:, idx+1:stop] + np.logical_not(mask[:, idx+1:stop]) * (invalid_flag+1) # seq_len的最后一个格子是无效的
            mat_min = np.min(mat, axis=1) # (batch,)
            mask[mat_min > invalid_flag+0.5, idx] = False # mask=False的部分会在mat被赋值data_max+1, 如果一个时序全都是无效部分, 就要被去掉
        # 将target转化为标签
        label = self.label_gen(target) * mask[..., None]
        if self.norm is not None:
            data = self.norm(data, self.avail_idx)

        result =  {'data': data, 'mask': mask, 'label': label}
        # save cache
        self.save_cache(result, force=False)
        return result
    
class SliceDataGenerator(DataGenerator):
    '''
    生成每个时间点和预测窗口的标签，但是不展开时间轴
    '''
    def __init__(self, window_points, n_fea, label_generator: LabelGenerator, target_idx, limit_idx=[], forbidden_idx=[], norm:Normalization=None, p_cache_file=None) -> None:
        super().__init__(n_fea, limit_idx, forbidden_idx, p_cache_file)
        self.norm = norm
        self.target_idx = target_idx
        self.window = window_points # 向前预测多少个点内的ARDS
        self.label_gen = label_generator
        self.slice_len = None
    
    def __call__(self, _data:np.ndarray, seq_lens:np.ndarray) -> np.ndarray:
        '''
        data: (batch, n_fea, seq_lens)
        mask: (batch, seq_lens)
        return: 
            label(n_slice, dim_target)
        '''
        cache = self.load_cache()
        if cache is not None:
            self.slice_len = cache['slice_len']
            return cache

        mask = tools.make_mask(_data.shape[[0,2]], seq_lens) # (batch, seq_lens)
        data = _data.copy()
        target = data[:, self.target_idx, :]
        data = data[:, self.avail_idx, :]
        self.slice_len = data.shape[0]
        # 将target按照时间顺序平移
        invalid_flag = target.max()
        for idx in range(target.shape[1]-1): # 最后一个格子预测一格
            stop = min(data.shape[1], idx+self.window)
            mat = mask[:, idx+1:stop] * target[:, idx+1:stop] + np.logical_not(mask[:, idx+1:stop]) * (invalid_flag+1) # seq_len的最后一个格子是无效的
            mat_min = np.min(mat, axis=1) # (batch,)
            mask[mat_min > invalid_flag+0.5, idx] = False # mask=False的部分会在mat被赋值data_max+1, 如果一个时序全都是无效部分, 就要被去掉
        # 将target转化为标签
        label = self.label_gen(target) * mask[..., None]
        if self.norm is not None:
            data = self.norm(data, self.avail_idx)

        # 转化为slice
        data, mask, label = self.make_slice(data), self.make_slice(mask), self.make_slice(np.transpose(label, (0, 2, 1))).transpose((0, 2, 1))
        result = {'data': data, 'mask': mask, 'label': label, 'slice_len': self.slice_len}
        # save cache
        self.save_cache(result, force=False)
        return result
    
    def make_slice(self, x:np.ndarray):
        '''
        input: (batch, n_fea, seq_len) or (batch, seq_len)
        output: (batch*slice_len, n_fea) or (batch*slice_len,)
        '''
        x_dim = len(x.shape)
        if x_dim == 2: # for mask (batch, seq_len)
            x = x[:, None, :]
        x = np.transpose(x, (0, 2, 1)) # (batch, seq_len, n_fea)
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2])) # 这样reshape不会破坏最后一维, 还原后等于原来的矩阵
        if x_dim == 2:
            return x[:, 0]
        else:
            return x

    def restore_from_slice(self, x:np.ndarray):
        '''make_slice的反向操作, 保证顺序不会更改'''
        if isinstance(x, list):
            return tuple([self.restore_from_slice(data) for data in x])
        else:
            batch = x.shape[0] // self.slice_len
            x = x.reshape((batch, self.slice_len, x.shape[1])) # 这样reshape不会破坏最后一维, 还原后等于原来的矩阵
            return x

class StaticDataGenerator(DataGenerator):
    '''
    生成每个时间点和预测窗口的标签，但是不展开时间轴
    '''
    def __init__(self, start_point, window_points, n_fea, label_generator: LabelGenerator, target_idx, limit_idx=[], forbidden_idx=[], norm:Normalization=None, p_cache_file=None) -> None:
        super().__init__(n_fea, limit_idx, forbidden_idx, p_cache_file)
        self.norm = norm
        self.target_idx = target_idx
        self.window = window_points
        self.start_point = start_point
        self.label_gen = label_generator
    
    def __call__(self, _data:np.ndarray, seq_lens:np.ndarray) -> np.ndarray:
        '''
        data: (batch, n_fea, seq_lens)
        mask: (batch, seq_lens)
        return: 
            mask(new_batch, window), label(new_batch, n_cls), data(new_batch, avail_idx, window)
        '''
        cache = self.load_cache()
        if cache is not None:
            return cache

        mask = tools.make_mask(_data.shape[[0,2]], seq_lens) # (batch, seq_lens)
        mask = mask[:, self.start_point:min(target.size(1), self.start_point+self.window)] # (batch, window)
        seq_mask = np.max(mask, axis=1) # (batch, ) 一些全都是0的序列需要丢弃
        mask = mask[seq_mask, :]

        data = _data.copy()
        target = data[seq_mask, self.target_idx, :]
        data = data[seq_mask, self.avail_idx, :]
        # 获取预测窗口内的target
        invalid_flag = target.max()
        target = target * mask + invalid_flag * np.logical_not(mask)
        target = np.min(target[:, min(target.size(1), self.window)], axis=1) # (batch,)

        # 将target转化为标签
        label = self.label_gen(target) * mask[: None]
        if self.norm is not None:
            data = self.norm(data, self.avail_idx)

        result =  {'data': data, 'mask': mask, 'label': label}
        # save cache
        self.save_cache(result, force=False)
        return result