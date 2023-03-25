import numpy as np
import torch
import tools
from tools import logger as logger
import os
from tqdm import tqdm
import pandas as pd

from catboost import CatBoostClassifier, Pool



def Collect_Fn(data_list:list):
    result = {}
    result['data'] = torch.as_tensor(np.stack([d['data'] for d in data_list], axis=0), dtype=torch.float32)
    result['length'] = np.asarray([d['length'] for d in data_list], dtype=np.int32)
    return result


class CatboostClsTrainer():
    def __init__(self, params:dict, dataset) -> None:
        self.params = params
        self.paths = params['paths']
        self.cache_path = self.paths['catboost_cls_cache']
        tools.reinit_dir(self.cache_path, build=True)
        # self.model = LSTMClsModel(params['device'], params['in_channels'])
        self.dataset = dataset
        self.target_idx = dataset.target_idx
        forbidden_idx = {self.dataset.idx_dict[name] for name in self.params['forbidden_feas']}
        self.generator = Cls2LabelGenerator(
            window=self.params['window'], ards_threshold=self.params['ards_threshold'],
            target_idx=self.target_idx, sepsis_time_idx=dataset.idx_dict['sepsis_time'],
            forbidden_idx=forbidden_idx, post_sepsis_time=self.params['max_post_sepsis_hour']
        ) # 生成标签
        # self.register_vals = {'shap_value':[]}
        # NOTICE: 为了确保out loss一样长, 不加overfit detector, 时间的损耗其实也很少
        self.model = CatBoostClassifier(
            train_dir=self.cache_path, # 读取数据
            iterations=params['iterations'],
            depth=params['depth'],
            loss_function=params['loss_function'],
            learning_rate=params['learning_rate'],
            verbose=0,
        )
        self.data_dict = None
        
    def _extract_data(self):
        result = {}
        for phase in ['train', 'valid', 'test']:
            self.dataset.mode(phase)
            data = []
            seq_lens = []
            for _, batch in tqdm(enumerate(self.dataset), desc=phase):
                data.append(batch['data'])
                seq_lens.append(batch['length'])
            data = np.stack(data, axis=0)
            mask = tools.make_mask((data.shape[0], data.shape[-1]), seq_lens)
            mask, out_dict = self.generator(data, mask) # e.g. [phase]['X']
            out_dict.update({'mask':mask})
            result[phase] = out_dict
        self.dataset.mode('all')
        return result

    def get_loss(self):
        vaild_df = pd.read_csv(os.path.join(self.cache_path, 'test_error.tsv'), sep='\t')
        train_df = pd.read_csv(os.path.join(self.cache_path, 'learn_error.tsv'), sep='\t')
        data = {
            'train': np.asarray(train_df[self.params['loss_function']])[:],
            'valid': np.asarray(vaild_df[self.params['loss_function']])[:]
        }
        return data

    def train(self):
        tools.reinit_dir(self.cache_path, build=True) # 这是catboost输出loss的文件夹
        self.data_dict = self._extract_data()
        train_X = self.data_dict['train']['X'][self.data_dict['train']['mask']]
        train_Y = self.data_dict['train']['Y'][self.data_dict['train']['mask']]
        valid_X = self.data_dict['valid']['X'][self.data_dict['valid']['mask']]
        valid_Y = self.data_dict['valid']['Y'][self.data_dict['valid']['mask']]
        pool_train = Pool(train_X, train_Y)
        pool_valid = Pool(valid_X, valid_Y)
        self.model.fit(pool_train, eval_set=pool_valid, use_best_model=True)
        
    def predict(self, mode):
        test_X = self.data_dict[mode]['X'][self.data_dict[mode]['mask']]
        pool_test = Pool(data=test_X)
        return self.model.predict(pool_test, prediction_type='Probability')[:,1]


class Cls2LabelGenerator():
    '''给出静态模型可用的特征'''
    def __init__(self, window, ards_threshold, target_idx, sepsis_time_idx, post_sepsis_time, forbidden_idx=None) -> None:
        '''
        window: 静态模型考虑多少时长内的ARDS
        ards_threshold: ARDS的PF_ratio阈值
        target_idx: PF_ratio位置
        sepsis_time_idx: sepsis_time位置
        post_sepsis_time: 最长能容忍距离发生sepsis多晚(小时)
        forbidden_idx: 为了避免static model受到影响, 需要屏蔽一些特征
        '''
        self.window = window # 静态模型cover多少点数
        self.ards_threshold = ards_threshold
        self.target_idx = target_idx
        self.sepsis_time_idx = sepsis_time_idx
        self.post_sepsis_time = post_sepsis_time
        self.forbidden_idx = forbidden_idx
        # generate idx
        self.used_idx = None

    def available_idx(self, n_fea):
        if self.used_idx is not None:
            return self.used_idx
        else:
            self.used_idx = []
            for idx in range(n_fea):
                if idx not in self.forbidden_idx:
                    self.used_idx.append(idx)
            return self.used_idx
            
    def __call__(self, data:np.ndarray, mask:np.ndarray) -> np.ndarray:
        '''
        data: (batch, n_fea, seq_lens)
        mask: (batch, seq_lens)
        return: (X, Y)
            X: (batch, new_n_fea)
            Y: (batch,)
            mask: (batch,)
        '''
        n_fea = data.shape[1]
        seq_lens = mask.sum(axis=1)
        sepsis_time = data[:, self.sepsis_time_idx, 0]
        mask = (sepsis_time > -self.post_sepsis_time)
        Y_label = np.zeros((data.shape[0],))
        for idx in range(data.shape[0]):
            Y_label[idx] = np.max(data[idx, -1, :min(seq_lens[idx], self.window)] < self.ards_threshold)
        return mask, {'X': data[:, self.available_idx(n_fea), 0], 'Y': Y_label}
  