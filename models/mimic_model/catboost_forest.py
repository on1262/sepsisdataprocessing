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
import random


class CatboostForestTrainer():
    '''集成GBDT森林, 提高模型鲁棒性'''
    def __init__(self, params:dict, dataset) -> None:
        self.params = params
        self.paths = params['paths']
        self.dataset = dataset
        self.target_idx = dataset.target_idx
        self.n_trees = params['n_trees']
        self.n_cls = len(params['centers'])
        self.n_fea = None
        self.generator = StaticLabelGenerator(
            window=self.params['window'], centers=self.params['centers'],
            target_idx=self.target_idx, forbidden_idx=self.params['forbidden_idx'],
            limit_idx=self.params['limit_idx']
        )
        self.models = None
        self.data_dict = None
        self.selects = {}
        self.important_idx = None

    def update_hash(self, combination_list, important_idx:list):
        self.important_idx = important_idx
        for idx, seq in enumerate(combination_list):
            arr = np.zeros((self.n_fea,), dtype=np.int32)
            arr[seq] = 1
            imp_code = ''.join([str(a) for a in arr[important_idx]])
            if self.selects.get(imp_code) is None:
                self.selects[imp_code] = [(arr, idx)]
            else:
                self.selects[imp_code].append((arr,idx))


    def find_nearest_model(self, arr:np.ndarray):
        '''arr: 元素只能是0/1, 0代表该元素缺失'''
        arr = arr.astype(np.int32)
        imp_code = ''.join([str(a) for a in arr[self.important_idx]])
        # 查找imp_code
        nearest = []
        min_dist = self.n_fea
        for key, idx in self.selects[imp_code]:
            dist = np.sum(np.logical_not(key[arr>0]))
            if dist < min_dist:
                min_dist = dist
                nearest = [[idx, np.sum(key)]]
            elif dist == min_dist:
                nearest.append([idx, np.sum(key)])
        if len(nearest) > 1:
            nearest = sorted(nearest, key=lambda x:x[1])[0][0]
        elif len(nearest) == 1:
            nearest = nearest[0][0]
        else:
            assert(0)
        return self.models[nearest]


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
            if self.n_fea is None:
                self.n_fea = out_dict['X'].shape[1]
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
        avail_idx = self.generator.available_idx(self.n_fea)
        imp_idx = []
        for pos, idx in enumerate(avail_idx):
            if idx in self.params['importance_idx']:
                imp_idx.append(pos)
        other_idx = sorted(list(set(range(self.n_fea)) - set(imp_idx)))
        for n in range(len(imp_idx)):
            combination_list += (list(combinations(imp_idx, n+1))) # 添加重要特征1->n的全组合
        for idx in range(len(combination_list)):
            combination_list[idx] = sorted(list(combination_list[idx]) + other_idx)
        combination_list.append(other_idx)
        # 生成其余特征的序列
        all_idx = list(range(self.n_fea))
        other_list = []
        for n in range(round(len(all_idx)*self.params['min_feature_coeff']), len(all_idx), 1):
            other_list += list(combinations(all_idx, n+1))
        # 随机挑选序列
        other_list = random.choices(other_list, k=self.n_trees)
        self.n_trees += len(combination_list)
        combination_list += [sorted(list(seq)) for seq in other_list]
        self.update_hash(combination_list, imp_idx)
        # 训练模型
        self.models = [CatBoostClassifier(
            iterations=self.params['iterations'],
            depth=self.params['depth'],
            loss_function=self.params['loss_function'],
            learning_rate=self.params['learning_rate'],
            verbose=0,
            class_weights=cls_weight,
            use_best_model=True
        ) for _ in range(len(combination_list))]
        # 开始训练
        dropout_generator = DropoutLabelGenerator(dropout=self.params['dropout'], miss_table=tools.generate_miss_table(self.dataset.idx_dict))
        for idx, model in enumerate(self.models):
            logger.info(f'Catboost forest: Train{idx}/{len(self.models)} models')
            _, new_train_X = dropout_generator(train_X.copy())
            _, new_valid_X = dropout_generator(valid_X.copy())
            avoid_list = []
            for i in range(self.n_fea):
                if i not in combination_list[idx]:
                    avoid_list.append(i)
            new_train_X[:, avoid_list] = -1
            new_valid_X[:, avoid_list] = -1
            pool_train = Pool(new_train_X, np.argmax(train_Y, axis=-1))
            pool_valid = Pool(new_valid_X, np.argmax(valid_Y, axis=-1))
            model.fit(pool_train, eval_set=pool_valid, use_best_model=True)
        
        
    def predict(self, mode, missrate=None):
        if missrate is None:
            test_X = self.data_dict[mode]['X'][self.data_dict[mode]['mask']]
            pool_test = Pool(data=test_X)
            return self.find_nearest_model(np.ones((self.n_fea,))).predict(pool_test, prediction_type='Probability')
        else:
            dropout_generator = DropoutLabelGenerator(dropout=missrate, miss_table=tools.generate_miss_table(self.dataset.idx_dict))
            test_X = self.data_dict[mode]['X'][self.data_dict[mode]['mask']] # (batch, n_fea)
            mask, new_test_X = dropout_generator(test_X.copy())
            result = np.zeros((new_test_X.shape[0], 4))
            for idx in range(mask.shape[0]):
                arr = mask[idx, :]
                model = self.find_nearest_model(arr)
                pool_test = Pool(data=new_test_X[[idx], :])
                result[idx, :] = model.predict(pool_test, prediction_type='Probability')[0, :]
            return result
            

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
