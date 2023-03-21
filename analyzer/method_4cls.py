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


class LSTM4ClsAnalyzer:
    '''动态模型, 四分类预测'''
    def __init__(self, params:dict, container:DataContainer) -> None:
        self.params = params
        self.paths = params['paths']
        self.container = container
        self.model_name = 'LSTM_4cls'
        self.loss_logger = tools.LossLogger()
        # copy attribute from container
        self.target_idx = self.dataset.target_idx
        self.dataset = container.dataset
        self.data = self.dataset.data
        # initialize
        self.out_dir = os.path.join(self.paths['out_dir'], self.model_name)
        tools.reinit_dir(self.out_dir, build=True)

    def generate_labels(self, generator):
        '''生成4class标签'''
        self.dataset.mode('all')
        pkl_path = os.path.join(self.out_dir, 'dataset_derived.pkl')
        if os.path.exists(pkl_path):
            logger.info(f'Load derived data set from {pkl_path}')
            with open(pkl_path, 'rb') as fp:
                mask, label = pickle.load(fp)
        else:
            logger.info('Generating label')
            mask = tools.make_mask((self.data.shape[0], self.data.shape[2]), self.dataset.seqs_len) # -> (batch, seq_lens)
            mask[:, -1] = False # 最后一格无法预测
            label = generator(self.data[:, self.target_idx, :], mask)
        return mask, label

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
        metric_2cls = tools.DichotomyMetric()
        metric_4cls = tools.MultiClassMetric(class_names=self.params['class_names'], out_dir=out_dir)
        # step 3: generate labels
        generator = mlib.LabelGenerator(window=self.params['window'], threshold=self.params['centers'], smoothing_band=self.params['smoothing_band'])
        mask, label = self.generate_labels(generator)
        # step 4: train and predict
        for idx, (data_index, test_index) in enumerate(kf.split(X=self.dataset)): 
            valid_num = round(len(data_index)*0.15)
            train_index, valid_index = data_index[valid_num:], data_index[:valid_num]
            self.dataset.register_split(train_index, valid_index, test_index)
            trainer = mlib.LSTMClsTrainer(self.params, self.dataset)
            if idx == 0:
                trainer.summary()
            trainer.train()
            self.loss_logger.add_loss(trainer.get_loss())
            Y_mask = mask[test_index, ...]
            Y_gt = label[test_index, ...]
            Y_pred = trainer.predict(mode='test')
            Y_pred = np.asarray(Y_pred)
            metric_4cls.add_prediction(Y_pred, Y_gt, Y_mask) # 去掉mask外的数据
            metric_2cls.add_prediction(map_func(Y_pred)[..., 1][Y_mask][:], map_func(Y_gt)[..., 1][Y_mask][:])
            self.dataset.mode('all') # 恢复原本状态
        # step 5: result explore
        self.loss_logger.plot(std_bar=False, log_loss=False, title='Loss for LSTM cls Model', 
            out_path=os.path.join(out_dir, 'loss.png'))
        metric_4cls.confusion_matrix(comment=self.model_name)
        metric_4cls.write_result()

        metric_2cls.plot_roc(title=f'{self.model_name} model ROC (4->2 cls)', save_path=os.path.join(out_dir, f'{self.model_name}_ROC.png'))
        print('Metric 2 classes:')
        print(metric_2cls.generate_info())

class BaselineNearestClsAnalyzer:
    def __init__(self, params, dataset) -> None:
        self.params = params
        self.dataset = dataset

    def run(self):
        '''预测窗口内是否发生ARDS的分类器'''
        if self.dataset.name() == 'mimic-iv':
            import datasets.mimic_dataset as dlib
            import models.mimic_model as mlib
        params = self.loc_conf['baseline_nearest_cls']
        kf = KFold(n_splits=self.n_fold, shuffle=True, random_state=100)
        self.register_values.clear()
        model_name = 'nearest_cls'
        out_dir = os.path.join(self.gbl_conf['paths']['out_dir'], model_name)
        tools.reinit_dir(out_dir, build=True)
        metric = tools.DichotomyMetric()
        # 制作ARDS标签
        self.dataset.mode('all') # 恢复原本状态
        logger.info('Generating ARDS label')
        generator = mlib.LabelGenerator(window=params['window'], threshold=self.ards_threshold)
        mask = mlib.make_mask((self.data.shape[0], self.data.shape[2]), self.dataset.seqs_len) # -> (batch, seq_lens)
        # 手动去掉最后一格
        mask[:, -1] = False
        labels = generator(self.data[:, self.target_idx, :], mask)
        positive_proportion = np.sum(labels[:]) / np.sum(mask)
        logger.info(f'Positive label={positive_proportion:.2f}')
        # 训练集划分
        for idx, (data_index, test_index) in enumerate(kf.split(X=self.dataset)):
            valid_num = round(len(data_index)*0.15)
            train_index, valid_index = data_index[valid_num:], data_index[:valid_num]
            self.dataset.register_split(train_index, valid_index, test_index)
            Y_mask = mask[test_index, :][:]
            Y_gt = labels[test_index, :][:][Y_mask]
            # prediction
            self.dataset.mode('test')
            Y_pred = np.zeros((len(test_index), self.data.shape[-1]), dtype=np.float32)
            for idx, data in tqdm(enumerate(self.dataset), desc='Testing', total=len(self.dataset)):
                np_data = data['data']
                seq_lens = data['length']
                for t_idx in range(seq_lens):
                    Y_pred[idx, t_idx] = np.mean(np_data[self.target_idx, :(t_idx+1)] < self.ards_threshold)
            Y_pred = np.asarray(Y_pred)[:][Y_mask]
            metric.add_prediction(Y_pred, Y_gt) # 去掉mask外的数据
            self.dataset.mode('all') # 恢复原本状态
        metric.plot_roc(title='Baseline cls model ROC', save_path=os.path.join(out_dir, 'baseline_nearest_cls_ROC.png'))
        print(metric.generate_info())

def map_func(a:np.ndarray):
    '''
    将4分类的结果map到2分类的结果
    默认是[0,1,2,3]对应[重度,中度,轻度,无]
    映射是ARDS=[0,1,2], No ARDS=[3]
    a: (..., n_cls) 可以是软标签
    return (..., 2) 其中[...,0]代表无ARDS, [...,1]代表有ARDS, 可以是软标签
    '''
    a_shape = a.shape
    a_shape[-1] = 2
    result = np.zeros(a_shape)
    result[..., 0] = a[..., 3]
    result[..., 1] = a[..., 0] + a[..., 1] + a[..., 2]
    return result


def explore_result(ards_threshold, Y_pred, Y_gt, mask, out_dir, cmt):
        '''
        输出二分类误差和flips的统计关系, 观察误差大的样本是否存在特殊的分布
        Y_pred, Y_gt: (batch, seq_lens), 值域只能是[0,1]
        '''
        delta = np.abs(Y_pred - Y_gt)
        cover = (Y_gt > 0) * (Y_gt < ards_threshold) * mask
        diffs = np.diff(cover.astype(int), axis=1)
        # count the number of flips
        num_flips = np.count_nonzero(diffs, axis=1)
        num_flips = np.repeat(num_flips[:, None], Y_pred.shape[1], axis=1)
        mask = mask[:]
        num_flips = num_flips[:][mask]
        delta = delta[:][mask][:, None]
        tools.plot_reg_correlation(
            X=delta, fea_names=['Prediction Abs Error'], Y=num_flips, target_name='Num flips', adapt=True, write_dir_path=out_dir, plot_dash=False, comment=cmt)