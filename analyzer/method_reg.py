import torch
import torchinfo
import numpy as np
from datasets.mimic_dataset import MIMICDataset, Subject, Admission, Config # 这个未使用的import是pickle的bug
from sklearn.model_selection import KFold
import tools
import os, pickle
from tqdm import tqdm
from tools import logger as logger
from .container import DataContainer
from .utils import generate_labels


class LSTM4RegAnalyzer:
    '''动态模型, 四分类预测'''
    def __init__(self, params:dict, container:DataContainer) -> None:
        self.params = params
        self.paths = params['paths']
        self.container = container
        self.model_name = 'LSTM_reg'
        self.loss_logger = tools.LossLogger()
        # copy attribute from container
        self.target_idx = self.dataset.target_idx
        self.dataset = container.dataset
        self.data = self.dataset.data
        # initialize
        self.out_dir = os.path.join(self.paths['out_dir'], self.model_name)
        tools.reinit_dir(self.out_dir, build=True)


    def label_explore(self, labels):
        pass

    def run(self):
        '''预测窗口内是否发生ARDS的分类器'''
        if self.dataset.name() == 'mimic-iv':
            import models.mimic_model as mlib
        # step 1: append additional params
        self.params['in_channels'] = self.dataset.data.shape[1]
        # step 2: init variables
        kf = KFold(n_splits=self.container.n_fold, shuffle=True, random_state=self.container.seed)
        out_dir = os.path.join(self.paths['out_dir'], self.model_name)
        tools.reinit_dir(out_dir, build=True)
        metric = tools.RegressionMetric(target_name=self.container.target_name, out_dir=self.out_dir)
        # step 3: generate labels
        generator = mlib.ClsLabelGenerator(window=self.params['window'], threshold=self.params['centers'], smoothing_band=self.params['smoothing_band'])
        mask, label = generate_labels(self.dataset, self.data, self.target_idx, generator, self.out_dir)
        # step 4: train and predict
        for idx, (data_index, test_index) in enumerate(kf.split(X=self.dataset)): 
            valid_num = round(len(data_index)*0.15)
            train_index, valid_index = data_index[valid_num:], data_index[:valid_num]
            self.dataset.register_split(train_index, valid_index, test_index)
            trainer = mlib.LSTMRegTrainer(self.params, self.dataset)
            if idx == 0:
                trainer.summary()
            trainer.train()
            self.loss_logger.add_loss(trainer.get_loss())
            Y_mask = mask[test_index, ...]
            Y_gt = label[test_index, ...]
            Y_pred = trainer.predict(mode='test')
            Y_pred = np.asarray(Y_pred)
            metric.add_prediction(Y_pred, Y_gt, Y_mask) # 去掉mask外的数据
            self.dataset.mode('all') # 恢复原本状态
        # step 5: result explore
        self.loss_logger.plot(std_bar=False, log_loss=False, title='Loss for LSTM cls Model', 
            out_path=os.path.join(out_dir, 'loss.png'))


class BaselineNearestRegAnalyzer:
    def __init__(self, params, dataset) -> None:
        self.params = params
        self.dataset = dataset
        self.target_idx = self.dataset.target_idx
        self.model_name = 'nearest_reg'

    def predict(self, mode:str):
        '''
        input: mode: ['test']
        output: (test_batch, seq_len)
        '''
        self.dataset.mode(mode)
        pred = np.zeros((len(self.dataset), self.dataset.data.shape[1]))
        for idx, data in tqdm(enumerate(self.dataset), desc='testing', total=len(self.dataset)):
            np_data = data['data']
            pred[idx, :] = np_data[self.target_idx, :]
        return pred

    def run(self):
        if self.dataset.name() == 'mimic-iv':
            import models.mimic_model as mlib
        # step 1: append additional params
        self.params['in_channels'] = self.dataset.data.shape[1]
        # step 2: init variables
        kf = KFold(n_splits=self.container.n_fold, shuffle=True, random_state=self.container.seed)
        out_dir = os.path.join(self.paths['out_dir'], self.model_name)
        tools.reinit_dir(out_dir, build=True)
        metric = tools.RegressionMetric(target_name=self.container.target_name, out_dir=self.out_dir)
        # step 3: generate labels
        generator = mlib.ClsLabelGenerator(window=self.params['window'], threshold=self.params['centers'], smoothing_band=self.params['smoothing_band'])
        mask, label = generate_labels(self.dataset, self.data, self.target_idx, generator, self.out_dir)
        # step 4: train and predict
        for _, (data_index, test_index) in enumerate(kf.split(X=self.dataset)): 
            valid_num = round(len(data_index)*0.15)
            train_index, valid_index = data_index[valid_num:], data_index[:valid_num]
            self.dataset.register_split(train_index, valid_index, test_index)
            Y_mask = mask[test_index, ...]
            Y_gt = label[test_index, ...]
            Y_pred = self.predict(mode='test')
            Y_pred = np.asarray(Y_pred)
            metric.add_prediction(Y_pred, Y_gt, Y_mask) # 去掉mask外的数据
            self.dataset.mode('all') # 恢复原本状态

