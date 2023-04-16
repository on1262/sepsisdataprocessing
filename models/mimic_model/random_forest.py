import numpy as np
import torch
import tools
from tools import logger as logger
import os
from tqdm import tqdm
import pandas as pd
from .utils import StaticLabelGenerator, DropoutLabelGenerator
from sklearn.ensemble import RandomForestClassifier


class RandomForestTrainer():
    def __init__(self, params:dict, dataset) -> None:
        self.params = params
        self.paths = params['paths']
        self.cache_path = self.paths['random_forest_cache']
        tools.reinit_dir(self.cache_path, build=True)
        self.dataset = dataset
        self.target_idx = dataset.target_idx
        self.generator = StaticLabelGenerator(
            window=self.params['window'], centers=self.params['centers'],
            target_idx=self.target_idx, forbidden_idx=self.params['forbidden_idx'],
            limit_idx=self.params['limit_idx']
        )
        # self.robust = self.params['robust']
        self.model = None
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


    def train(self, addi_params:dict=None):
        tools.reinit_dir(self.cache_path, build=True) # 这是catboost输出loss的文件夹
        self.data_dict = self._extract_data()
        #cls_weight = self.cal_label_weight(
        #    n_cls=len(self.params['centers']), mask=self.data_dict['train']['mask'], label=self.data_dict['train']['Y'])
        self.model = RandomForestClassifier(
            class_weight='balanced',
            max_depth=self.params['max_depth'],
            n_jobs=-1
        )
        train_X = self.data_dict['train']['X'][self.data_dict['train']['mask']]
        train_Y = self.data_dict['train']['Y'][self.data_dict['train']['mask']]
        valid_X = self.data_dict['valid']['X'][self.data_dict['valid']['mask']]
        valid_Y = self.data_dict['valid']['Y'][self.data_dict['valid']['mask']]
        if addi_params is not None:
            if 'dropout' in addi_params.keys():
                dropout_generator = DropoutLabelGenerator(missrate=addi_params['dropout'])
                _, train_X = dropout_generator(train_X)
                _, valid_X = dropout_generator(valid_X)
        tX = np.concatenate([train_X, valid_X], axis=0)
        tY = np.concatenate([np.argmax(train_Y, axis=-1), np.argmax(valid_Y, axis=-1)], axis=0)
        self.model.fit(tX, tY)
        
    def predict(self, mode, addi_params:dict=None):
        '''
        addi_params: dict | None 如果输入为dict, 则会监测是否存在key
            key=dropout:val=missrate, 开启testX的随机置为-1
        '''
        assert(self.model is not None)
        test_X = self.data_dict[mode]['X'][self.data_dict[mode]['mask']]
        if addi_params is not None:
            if 'dropout' in addi_params.keys():
                dropout_generator = DropoutLabelGenerator(missrate=addi_params['dropout'])
                _, test_X = dropout_generator(test_X)
        return self.model.predict_proba(test_X)

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