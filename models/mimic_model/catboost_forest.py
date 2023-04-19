import numpy as np
import torch
import tools
from tools import logger as logger
import os
from tqdm import tqdm
import pandas as pd
from .utils import StaticLabelGenerator, DropoutLabelGenerator
from itertools import combinations
from catboost import CatBoostClassifier, Pool


class CatboostForestTrainer():
    '''集成GBDT森林, 提高模型鲁棒性'''
    def __init__(self, params:dict, dataset) -> None:
        self.params = params
        self.paths = params['paths']
        self.dataset = dataset
        self.target_idx = dataset.target_idx
        self.n_trees = params['n_trees']
        self.n_fea = len(params['centers'])
        self.generator = StaticLabelGenerator(
            window=self.params['window'], centers=self.params['centers'],
            target_idx=self.target_idx, forbidden_idx=self.params['forbidden_idx'],
            limit_idx=self.params['limit_idx']
        )
        self.models = None
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

    def train(self):
        # prepare data and class weight
        self.data_dict = self._extract_data()
        cls_weight = self.cal_label_weight(
            n_cls=len(self.params['centers']), mask=self.data_dict['train']['mask'], label=self.data_dict['train']['Y'])
        train_X = self.data_dict['train']['X'][self.data_dict['train']['mask']]
        train_Y = self.data_dict['train']['Y'][self.data_dict['train']['mask']]
        valid_X = self.data_dict['valid']['X'][self.data_dict['valid']['mask']]
        valid_Y = self.data_dict['valid']['Y'][self.data_dict['valid']['mask']]
        # 生成重要特征的序列
        combination_list = []
        imp_idx = self.params['importance_idx']
        other_idx = list(set(self.generator.available_idx(self.n_fea)) - set(imp_idx))
        for n in range(len(imp_idx)):
            combination_list += (list(combinations(imp_idx, n+1))) # 添加重要特征1->n的全组合
        for idx in range(len(combination_list)):
            combination_list[idx] = sorted(list(combination_list[idx]) + other_idx)
        # 生成其余特征的序列
        other_list = []
        for n in range(round(len(other_idx)*0.6), len(other_idx), 1):
            other_list += list(combinations(other_idx, n+1))
        # 随机挑选序列
        
        self.model = CatBoostClassifier(
            iterations=self.params['iterations'],
            depth=self.params['depth'],
            loss_function=self.params['loss_function'],
            learning_rate=self.params['learning_rate'],
            verbose=0,
            class_weights=cls_weight,
            use_best_model=True
        )
        
        
        
        

        # pool_train = Pool(train_X, np.argmax(train_Y, axis=-1))
        # pool_valid = Pool(valid_X, np.argmax(valid_Y, axis=-1))
        # if 'dropout' in self.params.keys():
        #     dropout_generator = DropoutLabelGenerator(missrate=self.params['dropout'])
        #     _, train_X = dropout_generator(train_X)
        #     _, valid_X = dropout_generator(valid_X)
        # self.model.fit(pool_train, eval_set=pool_valid, use_best_model=True)
        
    def predict(self, mode):
        assert(self.model is not None)
        test_X = self.data_dict[mode]['X'][self.data_dict[mode]['mask']]
        pool_test = Pool(data=test_X)
        return self.model.predict(pool_test, prediction_type='Probability')

    def cal_label_weight(self, n_cls, mask, label):
        '''
        获取n_cls反比于数量的权重
        label: (batch, n_cls)
        mask: (batch,)
        return: (n_cls,)
        '''
        hard_label = np.argmax(label, axis=-1)
        hard_label = hard_label[:][mask[:]]
        weight = np.asarray([np.mean(hard_label == c) for c in range(n_cls)])
        logger.info(f'4cls Label proportion: {weight}')
        weight = 1 / weight
        weight = weight / np.sum(weight)
        logger.info(f'4cls weight: {weight}')
        return weight
