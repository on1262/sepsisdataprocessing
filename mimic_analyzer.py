import torch
import torchinfo
import numpy as np
from mimic_dataset import MIMICDataset, Subject, Admission # 这个未使用的import是pickle的bug
from mimic_model.lstm import LSTMModel
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
        self._plot_time_series_samples(n_sample=400, n_per_plots=40, write_dir=os.path.join(out_dir, "sample_plot"))
        # plot fourier tranform result
        # 不等长的话结果的合并很麻烦
        # self._plot_fourier_transform(self.target_name, self.dataset.target_time_arr[:,1], save_dir=out_dir)

    def _plot_time_series_samples(self, n_sample:int=100, n_per_plots:int=10, write_dir=None):
        if write_dir is not None:
            tools.reinit_dir(write_dir)
        n_sample = min(n_sample, self.data.shape[0])
        n_plot = int(np.ceil(n_sample / n_per_plots))
        start_idx = 0
        for p_idx in range(n_plot):
            stop_idx = min(start_idx + n_per_plots, n_sample)
            mat = self.dataset.restore_norm(self.target_idx, self.data[start_idx:stop_idx, self.target_idx, :]) * 100
            for idx in range(stop_idx-start_idx):
                series = mat[idx, :]
                plt.plot(series[series > 0], alpha=0.3)
            plt.title(f"Time series sample of {self.target_name}")
            plt.xlabel("time tick=0.5 hour")
            plt.ylabel(self.target_name)
            start_idx = stop_idx
            if write_dir is None:
                plt.show()
            else:
                plt.savefig(os.path.join(write_dir, f"plot_{p_idx}.png"))
            plt.close()
    
    def lstm_train(self, epochs):
        '''训练LSTM模型'''
        pass

    def lstm_test(self, test_config):
        '''测试LSTM模型, 自回归'''
        pass


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
                    start_idx_test = np.zeros((len(test_index)), dtype=np.int)
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