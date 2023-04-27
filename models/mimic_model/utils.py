import torch
import numpy as np
import tools


def Collect_Fn(data_list:list):
    result = {}
    result['data'] = torch.as_tensor(np.stack([d['data'] for d in data_list], axis=0), dtype=torch.float32)
    result['length'] = np.asarray([d['length'] for d in data_list], dtype=np.int32)
    return result


class DynamicLabelGenerator():
    '''
    生成每个时间点在预测窗口内的四分类标签
    soft_label: 是否开启软标签
    '''
    def __init__(self, soft_label=False, window=16, centers=list(), smoothing_band=50, limit_idx=[]) -> None:
        assert(len(centers) == 4)
        self.window = window # 向前预测多少个点内的ARDS
        self.centers = centers # 判断ARDS的界限
        self.band = smoothing_band # 平滑程度, 不能超过两个center之间的距离
        self.soft_label = soft_label # 是否开启软标签
        self.forbidden_idx = []
        self.limit_idx = limit_idx
        # generate idx
        self.used_idx = None

    def available_idx(self, n_fea=None):
        '''
        生成可用的特征序号
        '''
        if self.used_idx is not None:
            return self.used_idx
        else:
            assert(n_fea is not None)
            self.used_idx = []
            if len(self.limit_idx) == 0:
                for idx in range(n_fea):
                    if idx not in self.forbidden_idx:
                        self.used_idx.append(idx)
            else:
                for idx in range(n_fea):
                    if idx not in self.forbidden_idx and idx in self.limit_idx:
                        self.used_idx.append(idx)
            return self.used_idx
    
    def __call__(self, _data:np.ndarray, mask:np.ndarray) -> np.ndarray:
        '''
        data: (batch, n_fea, seq_lens)
        mask: (batch, seq_lens)
        return: mask(batch, seq_lens), label(batch, seq_lens, n_cls)
        '''
        data = _data[:, -1, :]
        assert(len(data.shape) == 2 and len(mask.shape) == 2)
        label = np.zeros(data.shape + (len(self.centers),), dtype=np.float32)
        data_max = data.max()
        for idx in range(data.shape[1]-1): # 最后一个格子预测一格
            stop = min(data.shape[1], idx+self.window)
            mat = mask[:, idx+1:stop] * data[:, idx+1:stop] + np.logical_not(mask[:, idx+1:stop]) * (data_max+1) # seq_len的最后一个格子是无效的
            mat_min = np.min(mat, axis=1) # (batch,)
            mask[mat_min > data_max+0.5, idx] = False # mask=False的部分会在mat被赋值data_max+1, 如果一个时序全都是无效部分, 就要被去掉
            if self.soft_label:
                label[:, idx, :] = tools.label_smoothing(self.centers, mat_min, band=50)
            else:
                for c_idx in range(len(self.centers)):
                    if c_idx == 0:
                        label[:, idx, c_idx] = (mat_min <= 0.5*(self.centers[0]+self.centers[1]))
                    elif c_idx == len(self.centers)-1:
                        label[:, idx, c_idx] = (mat_min > 0.5*(self.centers[c_idx-1]+self.centers[c_idx]))
                    else:
                        label[:, idx, c_idx] = \
                            np.logical_and(mat_min > 0.5*(self.centers[c_idx-1]+self.centers[c_idx]), mat_min <= 0.5*(self.centers[c_idx+1]+self.centers[c_idx]))
        return mask, (label * mask[..., None])

class SliceLabelGenerator:
    '''将动态模型生成结果去掉沿时间的拆分'''
    def __init__(self, slice_len, **kwargs):
        '''
        slice_len: 每个样本沿时间轴截取的点数, 
        **kwargs: 给dynamic generator的参数
        '''
        self.dyn_generator = DynamicLabelGenerator(**kwargs)
        self.slice_len = slice_len

    def __call__(self, _data:np.ndarray, _mask:np.ndarray) -> tuple:
        data, mask = _data.copy(), _mask.copy()
        mask, label = self.dyn_generator(data, mask) # (batch, n_fea, seq_len)
        data, mask, label = self.make_slice(data), self.make_slice(mask), self.make_slice(label)
        return mask, {'X': data, 'Y': label}

    def make_slice(self, x):
        '''(batch, n_fea, seq_len)->(batch*slice_len, n_fea)'''
        x = np.transpose(x[:,:, :self.slice_len], (0, 2, 1)) # (batch, seq_len, n_fea)
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2])) # 这样reshape不会破坏最后一维, 还原后等于原来的矩阵
        return x

    def restore_from_slice(self, x):
        '''make_slice的反向操作, 保证顺序不会更改'''
        if isinstance(x, list):
            return tuple([self.restore_from_slice(data) for data in x])
        else:
            batch = x.shape[0] // self.slice_len
            x = x.reshape((batch, self.slice_len, x.shape[1])) # 这样reshape不会破坏最后一维, 还原后等于原来的矩阵
            return x
        

class StaticLabelGenerator():
    '''生成一个大窗口内的最低ARDS四分类标签和训练数据'''
    def __init__(self, window, centers, target_idx, forbidden_idx, limit_idx) -> None:
        '''
        window: 考虑多少时长内的ARDS(point)
        centers: 各类中心
        target_idx: PF_ratio位置
        forbidden_idx: 为了避免static model受到影响, 需要屏蔽一些特征
        limit_idx: 如果为[], 则没有任何影响, 否则选择的特征只可能是其中的特征减去forbidden_idx的特征
        '''
        self.window = window # 静态模型cover多少点数
        self.centers = centers
        self.target_idx = target_idx
        self.forbidden_idx = forbidden_idx
        self.limit_idx = limit_idx
        # generate idx
        self.used_idx = None

    def available_idx(self, n_fea=None):
        '''
        生成可用的特征序号
        '''
        if self.used_idx is not None:
            return self.used_idx
        else:
            assert(n_fea is not None)
            self.used_idx = []
            if len(self.limit_idx) == 0:
                for idx in range(n_fea):
                    if idx not in self.forbidden_idx:
                        self.used_idx.append(idx)
            else:
                for idx in range(n_fea):
                    if idx not in self.forbidden_idx and idx in self.limit_idx:
                        self.used_idx.append(idx)
            return self.used_idx
            
    def __call__(self, data:np.ndarray, mask:np.ndarray) -> np.ndarray:
        '''
        data: (batch, n_fea, seq_lens)
        mask: (batch, seq_lens)
        return: (X, Y)
            X: (batch, new_n_fea)
            Y: (batch, n_cls)
            mask: (batch,)
        '''
        n_fea = data.shape[1]
        # seq_lens = mask.sum(axis=1)
        label = np.zeros((data.shape[0],len(self.centers)))
        data_max = data.max()
        stop = min(data.shape[1], self.window)
        mat = mask[:, 1:stop] * data[:, self.target_idx, 1:stop] + np.logical_not(mask[:, 1:stop]) * (data_max+1) # seq_len的最后一个格子是无效的
        mat_min = np.min(mat, axis=1) # (batch,)
        for c_idx in range(len(self.centers)):
            if c_idx == 0:
                label[:, c_idx] = (mat_min <= 0.5*(self.centers[c_idx]+self.centers[c_idx+1]))
            elif c_idx == len(self.centers)-1:
                label[:, c_idx] = (mat_min > 0.5*(self.centers[c_idx-1]+self.centers[c_idx]))
            else:
                label[:, c_idx] = \
                    np.logical_and(mat_min > 0.5*(self.centers[c_idx-1]+self.centers[c_idx]), mat_min <= 0.5*(self.centers[c_idx+1]+self.centers[c_idx]))
        return mask[:, 0], {'X': data[:, self.available_idx(n_fea), 0], 'Y': label}



class DropoutLabelGenerator:
    '''按照给定的缺失率, 将输入的数据进行-1填充'''
    def __init__(self, dropout, miss_table, value=-1) -> None:
        self.dropout = dropout
        self.value = value
        self.miss_table = np.asarray(miss_table)
        # norm
        self.miss_table = self.dropout * (self.miss_table / np.max(self.miss_table))
    
    def __call__(self, data:np.ndarray) -> np.ndarray:
        '''
        按照给定的缺失权重处理各个特征, 将data内的数据点变为-1, 返回mask和data
        data: (batch, n_fea) or (batch, n_fea, seq_len)
        '''
        dim = len(data.shape)
        if dim == 2:
            data = data[:, :, np.newaxis] # ->(batch, n_fea, seq_len)
        else:
            assert(dim == 3)
        mask = np.ones(data.shape, dtype=bool)
        if self.dropout == 0:
            return mask, data
        for idx in len(data.shape[1]):
            mat = np.ones((mask.shape[0]*mask.shape[2],), dtype=bool)
            k = round(min(1, self.miss_table[idx])*mask.shape[0]*mask.shape[2])
            mat[np.random.permutation(mask.shape[0]*mask.shape[2])[:k]] = False
            mask[:, idx, :] = np.reshape(mat, (mask.shape[0],mask.shape[2]))
        return mask, data * mask + self.value * np.logical_not(mask)

