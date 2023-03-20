import torch
import torchinfo
import numpy as np
from datasets.mimic_dataset import MIMICDataset, Subject, Admission, Config # 这个未使用的import是pickle的bug
import models.mimic_model as mimic_model
from sklearn.model_selection import KFold

import tools
import os
from tqdm import tqdm
from tools import logger as logger


def lstm_cls(self):
    '''预测窗口内是否发生ARDS的分类器'''
    params = self.loc_conf['lstm_cls']
    params['cache_path'] = self.gbl_conf['paths']['lstm_cls_cache']
    params['in_channels'] = self.data.shape[1]
    kf = KFold(n_splits=self.n_fold, shuffle=True, random_state=100)
    self.register_values.clear()
    model_name = 'LSTM_cls'
    out_dir = os.path.join(self.gbl_conf['paths']['out_dir'], model_name)
    tools.reinit_dir(out_dir, build=True)
    metric = tools.DichotomyMetric()
    # loss logger
    loss_logger = tools.LossLogger()
    # 制作ARDS标签
    self.dataset.mode('all') # 恢复原本状态
    logger.info('Generating ARDS label')
    generator = lstmcls.LabelGenerator(window=params['window'], threshold=self.ards_threshold)
    mask = lstmcls.make_mask((self.data.shape[0], self.data.shape[2]), self.dataset.seqs_len) # -> (batch, seq_lens)
    # 手动去掉最后一格
    mask[:, -1] = False
    labels = generator(self.data[:, self.target_idx, :], mask)
    self.label_explore(labels, mask, out_dir)
    positive_proportion = np.sum(labels[:]) / np.sum(mask)
    logger.info(f'Positive label={positive_proportion:.2f}')
    # 训练集划分
    for idx, (data_index, test_index) in enumerate(kf.split(X=self.dataset)):
        valid_num = round(len(data_index)*0.15)
        train_index, valid_index = data_index[valid_num:], data_index[:valid_num]
        self.dataset.register_split(train_index, valid_index, test_index)
        trainer = lstmcls.Trainer(params, self.dataset)
        if idx == 0:
            trainer.summary()
        trainer.train()
        loss_logger.add_loss(trainer.get_loss())
        Y_mask = mask[test_index, :]
        Y_gt = labels[test_index, :]
        Y_pred = trainer.predict(mode='test')
        Y_pred = np.asarray(Y_pred)
        Y_pred = (Y_pred - Y_pred.min()) / (Y_pred.max() - Y_pred.min()) # 使得K-fold每个model输出的分布相近, 避免average性能下降
        if idx == 0: # 探查prediction和gt差异大的点
            logger.info('Explore result')
            self.explore_cls_result(Y_pred, Y_gt, Y_mask, out_dir, f'idx={idx}')
        Y_mask = Y_mask[:]
        metric.add_prediction(Y_pred[:][Y_mask], Y_gt[:][Y_mask]) # 去掉mask外的数据
        self.dataset.mode('all') # 恢复原本状态
    loss_logger.plot(std_bar=False, log_loss=False, title='Loss for LSTM cls Model', 
        out_path=os.path.join(out_dir, 'loss.png'))
    metric.plot_roc(title='LSTM cls model ROC', save_path=os.path.join(out_dir, 'lstm_cls_ROC.png'))
    print(metric.generate_info())


def nearest_cls(self):
        '''预测窗口内是否发生ARDS的分类器'''
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
        generator = lstmcls.LabelGenerator(window=params['window'], threshold=self.ards_threshold)
        mask = lstmcls.make_mask((self.data.shape[0], self.data.shape[2]), self.dataset.seqs_len) # -> (batch, seq_lens)
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
            X=delta, fea_names=['Prediction Abs Error'], Y=num_flips, target_name='Num flips', restrict_area=True, write_dir_path=out_dir, plot_dash=False, comment=cmt)