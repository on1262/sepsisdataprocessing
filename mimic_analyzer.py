import torch
import torchinfo
import numpy as np
from mimic_dataset import MIMICDataset, Subject, Admission,Config # 这个未使用的import是pickle的bug
from mimic_model.lstm import LSTMModel, LSTMTrainer
from mimic_model.baseline import BaselineNearest
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import tools
import os
from tools import logger as logger


# 鉴于mimic数据集提取后的大小和数量都和之前的数据集在同一规模, 所以这里决定写一个接口(继承)
# 直接复用dynamic_analyzer的代码

class MIMICAnalyzer():
    def __init__(self, dataset:MIMICDataset):
        self.dataset = dataset
        self.data = dataset.data
        self.gbl_conf = tools.GLOBAL_CONF_LOADER['mimic_analyzer']
        self.loc_conf = Config(cache_path=self.gbl_conf['paths']['conf_cache_path'], manual_path=self.gbl_conf['paths']['conf_manual_path'])
        self.target_name = dataset.target_name # only one name
        self.target_idx = dataset.target_idx
        self.n_fold = 4
        # for feature importance
        self.register_values = {}

    def feature_explore(self):
        '''输出mimic-iv数据集的统计特征'''
        logger.info('Analyzer: Feature explore')
        out_dir = self.gbl_conf['paths']['out_dir']
        
        # random plot sample time series
        self._plot_time_series_samples(self.target_name, n_sample=400, n_per_plots=40, write_dir=os.path.join(out_dir, "target_plot"))
        self._plot_time_series_samples("220224", n_sample=400, n_per_plots=40, write_dir=os.path.join(out_dir, "pao2_plot"))
        self._plot_time_series_samples("223835", n_sample=400, n_per_plots=40, write_dir=os.path.join(out_dir, "fio2_plot"))
        # plot fourier tranform result
        # 不等长的话结果的合并很麻烦
        # self._plot_fourier_transform(self.target_name, self.dataset.target_time_arr[:,1], save_dir=out_dir)

    def _detect_adm_data(self, id:int):
        for s_id, s in self.dataset.subjects.items():
            for adm in s.admissions:
                logger.info(adm[int(id)][:,0])
    
    def _plot_time_series_samples(self, fea_name:str, n_sample:int=100, n_per_plots:int=10, write_dir=None):
        '''
        fea_name: total_keys中的项, 例如"220224"
        '''
        if write_dir is not None:
            tools.reinit_dir(write_dir)
        n_sample = min(n_sample, self.data.shape[0])
        n_plot = int(np.ceil(n_sample / n_per_plots))
        fea_idx = self.dataset.idx_dict[fea_name]
        start_idx = 0
        label = self.dataset.get_fea_label(fea_name)
        for p_idx in range(n_plot):
            stop_idx = min(start_idx + n_per_plots, n_sample)
            mat = self.dataset.restore_norm(fea_idx, self.data[start_idx:stop_idx, fea_idx, :])
            for idx in range(stop_idx-start_idx):
                series = mat[idx, :]
                plt.plot(series[series > 0], alpha=0.3)
            plt.title(f"Time series sample of {label}")
            plt.xlabel("time tick=0.5 hour")
            plt.xlim(left=0, right=72)
            plt.ylabel(label)
            start_idx = stop_idx
            if write_dir is None:
                plt.show()
            else:
                plt.savefig(os.path.join(write_dir, f"plot_{p_idx}.png"))
            plt.close()

    def lstm_model(self):
        params = self.loc_conf['lstm_model']
        kf = KFold(n_splits=self.n_fold, shuffle=True)
        self.register_values.clear()
        model_name = 'LSTM_model'
        tools.reinit_dir(os.path.join(self.gbl_conf['paths']['out_dir'], model_name), build=True)
        start_points = params['start_points']
        log_paths = [os.path.join(self.gbl_conf['paths']['out_dir'], model_name, f'result_{sp}.log') for sp in start_points]
        pred_point = params['pred_point']
        # datasets and params
        metrics = {key:tools.DynamicPredictionMetric(target_name=self.target_name, out_dir=self.gbl_conf['paths']['out_dir']) for key in start_points}
        params['in_channels'] = self.data.shape[1]
        logger.info(f'lstm_model: in channels=' + str(params['in_channels']))
        params['target_idx'] = self.target_idx
        params['target_std'] = self.dataset.norm_dict[self.target_name]['std']
        params['fea_names'] = self.dataset.total_keys
        params['cache_path'] = self.gbl_conf['paths']['lstm_cache']
        # loss logger
        loss_logger = tools.LossLogger()
        # 训练集划分
        for idx, (data_index, test_index) in enumerate(kf.split(X=self.dataset)):
            valid_num = round(len(data_index)*0.15)
            train_index, valid_index = data_index[valid_num:], data_index[:valid_num]
            self.dataset.register_split(train_index, valid_index, test_index)
            trainer = LSTMTrainer(params, self.dataset)
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
                yp = self.dataset.restore_norm(self.target_name, Y_pred[idx, :, :]) # 恢复norm
                gt = self.dataset.restore_norm(self.target_name, Y_gt[:, key+1:key+pred_point+1])
                metrics[key].add_prediction(prediction=yp , gt=gt, start_idx=start_idx, duration=duration)
            self.dataset.mode('all') # 恢复原本状态
        loss_logger.plot(std_bar=False, log_loss=False, title='Loss for LSTM Model', 
            out_path=os.path.join(self.gbl_conf['paths']['out_dir'], tools.remove_slash(model_name), 'loss.png'))
        for idx, key in enumerate(start_points):
            metrics[key].write_result(model_name+'_sp_'+str(key), log_path=log_paths[idx])
            metrics[key].plot(model_name+'_sp_'+str(key))
        # self.create_final_result()


    def nearest_method(self):
        params = self.loc_conf['baseline_nearest']
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
                yp = self.dataset.restore_norm(self.target_name, Y_pred[idx, :, :]) # 恢复norm
                gt = self.dataset.restore_norm(self.target_name, Y_gt[:, key+1:key+pred_point+1])
                metrics[key].add_prediction(prediction=yp , gt=gt, start_idx=start_idx, duration=duration)
            self.dataset.mode('all') # 恢复原本状态
        for idx, key in enumerate(start_points):
            metrics[key].write_result(model_name+'_sp_'+str(key), log_path=log_paths[idx])
            metrics[key].plot(model_name+'_sp_'+str(key))
        

    # 收集各个文件夹里面的result.log, 合并为final result.log
    def create_final_result(self):
        logger.info('Creating final result')
        out_dir = self.gbl_conf['paths']['out_dir']
        with open(os.path.join(out_dir, 'final_result.log'), 'w') as final_f:
            for dir in os.listdir(out_dir):
                p = os.path.join(out_dir, dir)
                if os.path.isdir(p):
                    if 'result.log' in os.listdir(p):
                        rp = os.path.join(p, 'result.log')
                        logger.info(f'Find: {rp}')
                        with open(rp, 'r') as f:
                            final_f.write(f.read())
                            final_f.write('\n')
        logger.info(f'Final result saved at ' + os.path.join(out_dir, 'final_result.log'))

if __name__ == '__main__':
    dataset = MIMICDataset()
    analyzer = MIMICAnalyzer(dataset)
    analyzer._detect_adm_data(220224)
    # analyzer.feature_explore()
    # analyzer.lstm_model()
    # analyzer.nearest_method()