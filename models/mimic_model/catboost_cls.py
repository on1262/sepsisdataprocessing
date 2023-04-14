import numpy as np
import torch
import tools
from tools import logger as logger
import os
from tqdm import tqdm
import pandas as pd
from .utils import Collect_Fn, StaticLabelGenerator
from catboost import CatBoostClassifier, Pool



class CatboostTrainer():
    def __init__(self, params:dict, dataset) -> None:
        self.params = params
        self.paths = params['paths']
        self.cache_path = self.paths['catboost_cls_cache']
        tools.reinit_dir(self.cache_path, build=True)
        # self.model = LSTMClsModel(params['device'], params['in_channels'])
        self.dataset = dataset
        self.target_idx = dataset.target_idx
        self.generator = StaticLabelGenerator(
            window=self.params['window'], ards_threshold=self.params['ards_threshold'],
            target_idx=self.target_idx, sepsis_time_idx=dataset.idx_dict['sepsis_time'],
            forbidden_idx=self.params['forbidden_idx'], post_sepsis_time=self.params['max_post_sepsis_hour'], 
            limit_idx=self.params['limit_idx']
        ) # 生成标签
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


