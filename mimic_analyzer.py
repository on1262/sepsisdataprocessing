import torch
import torchinfo
import numpy as np
from mimic_dataset import MIMICDataset, Subject, Admission # 这个未使用的import是pickle的bug
from mimic_model.lstm import LSTMModel, LSTMTrainer
from dynamic_model import baseline as Baseline
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
        self.conf = tools.GLOBAL_CONF_LOADER['mimic_analyzer']
        self.target_name = dataset.target_name # only one name
        self.target_idx = dataset.target_idx
        self.n_fold = 5
        # for feature importance
        self.register_values = {}

    def feature_explore(self):
        '''输出mimic-iv数据集的统计特征'''
        logger.info('Analyzer: Feature explore')
        out_dir = self.conf['paths']['out_dir']
        # random plot sample time series
        self._plot_time_series_samples(self.target_name, n_sample=400, n_per_plots=40, write_dir=os.path.join(out_dir, "target_plot"))
        self._plot_time_series_samples("220224", n_sample=400, n_per_plots=40, write_dir=os.path.join(out_dir, "pao2_plot"))
        self._plot_time_series_samples("223835", n_sample=400, n_per_plots=40, write_dir=os.path.join(out_dir, "fio2_plot"))
        # plot fourier tranform result
        # 不等长的话结果的合并很麻烦
        # self._plot_fourier_transform(self.target_name, self.dataset.target_time_arr[:,1], save_dir=out_dir)

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
        for p_idx in range(n_plot):
            stop_idx = min(start_idx + n_per_plots, n_sample)
            mat = self.dataset.restore_norm(fea_idx, self.data[start_idx:stop_idx, fea_idx, :]) * 100
            for idx in range(stop_idx-start_idx):
                series = mat[idx, :]
                plt.plot(series[series > 0], alpha=0.3)
            plt.title(f"Time series sample of {fea_name}")
            plt.xlabel("time tick=0.5 hour")
            plt.ylabel(self.dataset.get_fea_label(fea_name))
            start_idx = stop_idx
            if write_dir is None:
                plt.show()
            else:
                plt.savefig(os.path.join(write_dir, f"plot_{p_idx}.png"))
            plt.close()

    def lstm_model(self, params):
        kf = KFold(n_splits=self.n_fold, shuffle=True)
        self.register_values.clear()
        model_name = 'LSTM_model'
        tools.reinit_dir(os.path.join(self.conf['paths']['out_dir'], model_name), build=True)
        log_path = os.path.join(self.conf['paths']['out_dir'], model_name, 'result.log')
        metrics_list = params['metrics_list']
        # datasets and params
        metrics = {key:tools.DynamicPredictionMetric(target_name=self.target_name, out_dir=self.conf['paths']['out_dir']) for key in metrics_list}
        # if 'quantile' in params.keys():
        #     for mode in metrics_list:
        #         metrics[mode].set_quantile(params['quantile'], round((len(params['quantile']) - 1) / 2))
        #     logger.info(f'Enable quantile in model {model_name}')
        params['in_channels'] = self.data.shape[1]
        logger.info(f'lstm_model: in channels=' + str(params['in_channels']))
        params['target_idx'] = self.target_idx
        params['fea_names'] = self.dataset.total_keys
        params['cache_path'] = self.conf['paths']['lstm_cache']
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
            for mode in metrics_list:
                index = None
                if mode == 'test':
                    index = test_index
                elif mode == 'valid':
                    index = valid_index
                else:
                    index = train_index
                Y_gt = self.data[index, self.target_idx, 1:]
                Y_pred = trainer.predict(mode=mode)
                Y_pred = np.asarray(Y_pred)
                # if mode == 'test': # 生成输出采样图
                #     if 'quantile' in params.keys():
                #         quantile_idx = round(len(params['quantile'][:-1]) / 2)
                #         tools.simple_plot(data=Y_pred[quantile_idx, np.random.randint(low=0, high=Y_pred.shape[1]-1, size=(10,)), :], 
                #             title=f'Random Output Plot idx={idx}', out_path=os.path.join(self.conf['paths']['out_dir'], model_name, f'out_plot_idx={idx}.png'))
                # 生成对齐后的start_idx, 并不是原来的start_idx
                start_idx = np.zeros((Y_pred.shape[0]), dtype=np.int32)
                duration = self.dataset.seqs_len
                metrics[mode].add_prediction(prediction=Y_pred, gt=Y_gt, start_idx=start_idx, duration=duration[index])
        loss_logger.plot(std_bar=False, log_loss=False, title='Loss for LSTM Model', 
            out_path=os.path.join(self.conf['paths']['out_dir'], tools.remove_slash(model_name), 'loss.png'))
        # if metrics['test'].quantile_flag:
        #     self._plot_quantile_association(model_name, metrics['test'].get_record())
        metrics['test'].write_result(model_name, log_path=log_path)
        metrics['test'].plot(model_name)
        # for key in metrics_list:
        #     if key != 'test':
        #         corr_dir = os.path.join(self.conf['paths']['out_dir'], tools.remove_slash(model_name), f'correlation_{key}')
        #         tools.reinit_dir(write_dir_path=corr_dir)
        #         metrics[key].plot_corr(corr_dir=corr_dir, comment=key)
        self.create_final_result()


    def baseline_methods(self, models:set, params=None):
        kf = KFold(n_splits=self.n_fold, shuffle=True)
        for model_name in models:
            self.register_values.clear()
            tools.reinit_dir(os.path.join(self.conf['paths']['out_dir'], model_name), build=True)
            log_path = os.path.join(self.conf['paths']['out_dir'], model_name, 'result.log')
            tools.clear_file(log_path)
            logger.info(f'Evaluating baseline methods:{model_name}')
            metric = None
            
            if 'simple' in model_name: # simple_nearest simple_average simple_holt
                dataset = self.data[:, self.target_idx, :].copy()
                metric = tools.DynamicPredictionMetric(target_name=self.target_name, out_dir=self.conf['paths']['out_dir'])
                predictor = Baseline.SimpleTimeSeriesPredictor()
                for _, (_, test_index) in enumerate(kf.split(X=dataset)):
                    X_test = dataset[test_index, :] # (sample, tick)
                    start_idx_test = np.zeros((len(test_index)), dtype=np.int32)
                    duration_test = self.dataset.seqs_len[test_index]
                    result = predictor.predict(X_test, start_idx=start_idx_test, duration=duration_test, mode=model_name.split('simple_')[1], params=params)
                    metric.add_prediction(prediction=result, gt=X_test, start_idx=start_idx_test, duration=duration_test)


    # 收集各个文件夹里面的result.log, 合并为final result.log
    def create_final_result(self):
        logger.info('Creating final result')
        out_dir = self.conf['paths']['out_dir']
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
    analyzer.feature_explore()