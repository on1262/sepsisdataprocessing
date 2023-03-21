import numpy as np
import torch
import tools
from tools import logger as logger
import os
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostClassifier, Pool



def Collect_Fn(data_list:list):
    result = {}
    result['data'] = torch.as_tensor(np.stack([d['data'] for d in data_list], axis=0), dtype=torch.float32)
    result['length'] = np.asarray([d['length'] for d in data_list], dtype=np.int32)
    return result


class CatboostClsTrainer():
    def __init__(self, params:dict, dataset) -> None:
        self.params = params
        self.cache_path = params['cache_path']
        tools.reinit_dir(self.cache_path, build=True)
        # self.model = LSTMClsModel(params['device'], params['in_channels'])
        self.dataset = dataset
        self.target_idx = dataset.target_idx
        self.generator = Cls2LabelGenerator(window=self.params['window'], ards_threshold=self.params['ards_threshold']) # 生成标签
        self.register_vals = {'train_loss':[], 'valid_loss':[]}
        # model
        self.model = CatBoostClassifier(
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
        for phase in ['train', 'test', 'valid']:
            self.dataset.mode(phase)
            data = []
            seq_lens = []
            for _, batch in tqdm(enumerate(self.dataset), desc=phase):
                data.append(batch['data'])
                seq_lens.append(batch['length'])
            data = np.stack(data, axis=0)
            result[phase] = self.generator(data, seq_lens) # e.g. [phase]['X']
        self.dataset.mode('all')
        return result

    def get_loss(self):
        data = {
            'train': np.asarray(self.register_vals['train_loss']),
            'valid': np.asarray(self.register_vals['valid_loss'])
        }
        return data
    

    def train(self):
        self.data_dict = self._extract_data()
        pool_train = Pool(self.data_dict['train']['X'], self.data_dict['train']['Y'])
        pool_valid = Pool(self.data_dict['valid']['X'], self.data_dict['valid']['Y'])
        
        self.model.fit(pool_train, eval_set=pool_valid, use_best_model=True)
        # TODO get loss
        
    def predict(self, mode):
        pool_test = Pool(data=self.data_dict[mode]['X'])
        return self.model.predict(pool_test, prediction_type='Probability')[:,1]


class Cls2LabelGenerator():
    '''给出未来长期是否发生ARDS和静态模型可用的特征(t=0)'''
    def __init__(self, window=144, ards_threshold=300) -> None:
        self.window = window # 静态模型cover多少点数
        self.ards_threshold = ards_threshold
    
    def __call__(self, data:np.ndarray, seq_lens) -> np.ndarray:
        '''
        data: (batch, n_fea, seq_lens)
        mask: (batch,)
        return: (X, Y)
            X: (batch, n_fea-1)
            Y: (batch,)
        '''
        Y = np.zeros((data.shape[0],))
        for idx in range(data.shape[0]):
            Y[idx] = np.max(data[idx, -1, :min(seq_lens[idx], self.window)] < self.ards_threshold)
        return {'X': data[:, :-1, 0], 'Y': Y}
  