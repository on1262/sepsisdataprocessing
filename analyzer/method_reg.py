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



def lstm_reg(self):
    '''回归模型'''
    params = self.loc_conf['lstm_autoreg']
    kf = KFold(n_splits=self.n_fold, shuffle=True)
    self.register_values.clear()
    model_name = 'LSTM_autoreg'
    tools.reinit_dir(os.path.join(self.gbl_conf['paths']['out_dir'], model_name), build=True)
    start_points = params['start_points']
    log_paths = [os.path.join(self.gbl_conf['paths']['out_dir'], model_name, f'result_{sp}.log') for sp in start_points]
    pred_point = params['pred_point']
    # datasets and params
    metrics = {key:tools.DynamicPredictionMetric(target_name=self.target_name, out_dir=self.gbl_conf['paths']['out_dir']) for key in start_points}
    params['in_channels'] = self.data.shape[1]
    params['target_idx'] = self.target_idx
    params['target_std'] = self.dataset.norm_dict[self.target_name]['std']
    params['fea_names'] = self.dataset.total_keys
    params['cache_path'] = self.gbl_conf['paths']['lstm_autoreg_cache']
    params['norm_arr'] = self.dataset.get_norm_array()
    logger.info(f'lstm_model: in channels=' + str(params['in_channels']))
    assert(np.min(params['norm_arr'][:, 1]) > 1e-4) # std can not be zero
    # loss logger
    loss_logger = tools.LossLogger()
    # 训练集划分
    for idx, (data_index, test_index) in enumerate(kf.split(X=self.dataset)):
        valid_num = round(len(data_index)*0.15)
        train_index, valid_index = data_index[valid_num:], data_index[:valid_num]
        self.dataset.register_split(train_index, valid_index, test_index)
        trainer = lstmreg.LSTMAutoRegTrainer(params, self.dataset)
        if idx == 0:
            trainer.summary()
        trainer.train()
        loss_logger.add_loss(trainer.get_loss())
        Y_gt = self.data[test_index, self.target_idx, :]
        Y_pred = trainer.predict(mode='test', start_points=start_points, pred_point=pred_point)
        Y_pred = np.asarray(Y_pred)
        # 生成对齐后的start_idx, 并不是原来的start_idx
        start_idx = np.zeros((len(test_index),), dtype=np.int32)
        duration = pred_point*np.ones((len(test_index),), dtype=np.int32)
        for idx, key in enumerate(start_points):
            yp, gt = Y_pred[idx, :, :], Y_gt[:, key+1:key+pred_point+1]
            # yp = self.dataset.restore_norm(self.target_name, Y_pred[idx, :, :]) # 恢复norm
            # gt = self.dataset.restore_norm(self.target_name, Y_gt[:, key+1:key+pred_point+1])
            metrics[key].add_prediction(prediction=yp , gt=gt, start_idx=start_idx, duration=duration)
        self.dataset.mode('all') # 恢复原本状态
    loss_logger.plot(std_bar=False, log_loss=False, title='Loss for LSTM autoreg Model', 
        out_path=os.path.join(self.gbl_conf['paths']['out_dir'], tools.remove_slash(model_name), 'loss.png'))
    for idx, key in enumerate(start_points):
        metrics[key].write_result(model_name+'_sp_'+str(key), log_path=log_paths[idx])
        metrics[key].plot(model_name+'_sp_'+str(key))
    # self.create_final_result()


def nearest_reg(self):
        '''查看自相关性'''
        params = self.loc_conf['baseline_nearest_reg']
        kf = KFold(n_splits=self.n_fold, shuffle=True)
        self.register_values.clear()
        model_name = 'baseline_nearest'
        start_points = params['start_points']
        metrics = {key:tools.DynamicPredictionMetric(target_name=self.target_name, out_dir=self.gbl_conf['paths']['out_dir']) for key in start_points}
        tools.reinit_dir(os.path.join(self.gbl_conf['paths']['out_dir'], model_name), build=True)
        params['target_idx'] = self.target_idx
        params['target_std'] = self.dataset.norm_dict[self.target_name]['std']
        
        pred_point = params['pred_point']
        log_paths = [os.path.join(self.gbl_conf['paths']['out_dir'], model_name, f'result_{sp}.log') for sp in start_points]
        for idx, (data_index, test_index) in enumerate(kf.split(X=self.dataset)):
            valid_num = round(len(data_index)*0.15)
            train_index, valid_index = data_index[valid_num:], data_index[:valid_num]
            self.dataset.register_split(train_index, valid_index, test_index)
            model = BaselineNearest(params, self.dataset)
            Y_gt = self.data[test_index, self.target_idx, :]
            Y_pred = model.predict(mode='test', start_points=start_points, pred_point=pred_point)
            # 生成对齐后的start_idx, 并不是原来的start_idx
            start_idx = np.zeros((len(test_index),), dtype=np.int32)
            duration = pred_point*np.ones((len(test_index),), dtype=np.int32)
            for idx, key in enumerate(start_points):
                yp = Y_pred[idx, :, :]
                gt = Y_gt[:, key+1:key+pred_point+1]
                # yp = self.dataset.restore_norm(self.target_name, Y_pred[idx, :, :]) # 恢复norm
                # gt = self.dataset.restore_norm(self.target_name, Y_gt[:, key+1:key+pred_point+1])
                metrics[key].add_prediction(prediction=yp , gt=gt, start_idx=start_idx, duration=duration)
            self.dataset.mode('all') # 恢复原本状态
        for idx, key in enumerate(start_points):
            metrics[key].write_result(model_name+'_sp_'+str(key), log_path=log_paths[idx])
            metrics[key].plot(model_name+'_sp_'+str(key))