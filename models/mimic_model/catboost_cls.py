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
        self.generator = Cls2LabelGenerator(window=self.params['window'], ards_threshold=self.params['ards_threshold']) # 生成标签
        # self.register_vals = {'shap_value':[]}
        # model
        self.model = CatBoostClassifier(
            train_dir=self.cache_path, # 读取数据
            iterations=params['iterations'],
            depth=params['depth'],
            loss_function=params['loss_function'],
            learning_rate=params['learning_rate'],
            od_type = "Iter",
            od_wait = 50
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
            _, result[phase] = self.generator(data, mask) # e.g. [phase]['X']
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
        pool_train = Pool(self.data_dict['train']['X'], self.data_dict['train']['Y'])
        pool_valid = Pool(self.data_dict['valid']['X'], self.data_dict['valid']['Y'])
        
        self.model.fit(pool_train, eval_set=pool_valid, use_best_model=True)
        
    def predict(self, mode):
        pool_test = Pool(data=self.data_dict[mode]['X'])
        return self.model.predict(pool_test, prediction_type='Probability')[:,1]

class Cls2LabelGenerator():
    '''给出未来长期是否发生ARDS和静态模型可用的特征(t=0)'''
    def __init__(self, window=144, ards_threshold=300) -> None:
        self.window = window # 静态模型cover多少点数
        self.ards_threshold = ards_threshold
    
    def __call__(self, data:np.ndarray, mask:np.ndarray) -> np.ndarray:
        '''
        data: (batch, n_fea, seq_lens)
        mask: (batch, seq_lens)
        return: (X, Y)
            X: (batch, n_fea-1) without first PF_ratio
            Y: (batch,)
        '''
        seq_lens = mask.sum(axis=1)
        Y = np.zeros((data.shape[0],))
        for idx in range(data.shape[0]):
            Y[idx] = np.max(data[idx, -1, :min(seq_lens[idx], self.window)] < self.ards_threshold)
        return mask, {'X': data[:, :-1, 0], 'Y': Y}
  