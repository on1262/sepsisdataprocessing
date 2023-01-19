import tools
from dynamic_sepsis_dataset import DynamicSepsisDataset
import os
import numpy as np
import pandas as pd
from tools import logger as logger
import matplotlib.pyplot as plt


class DynamicAnalyzer:
    def __init__(self, dataset:DynamicSepsisDataset):
        self.dataset = dataset
        self.conf = tools.GLOBAL_CONF_LOADER['dynamic_analyzer']
        self.data_pd = dataset.data_pd
        self.type_dict = dataset.get_type_dict()
        self.target_fea = dataset.target_fea
        self.fea_manager = dataset.fea_manager
        # init time arr
        self.target_time_arr = tools.cal_available_time(
            data=self.data_pd,
            expanded_target=self.fea_manager.get_expanded_fea(self.target_fea)
        ) # [:,0]=start_time, [:,1]=duration

    def feature_explore(self):
        logger.info('Analyzer: Feature explore')
        out_dir = self.conf['paths']['out_dir']
        # target feature avaliable days
        tools.plot_single_dist(self.target_time_arr[:,1] / 24, f"Available Days: {self.target_fea}", \
            save_path=os.path.join(out_dir, 'target_available_days.png'))
        # first ards days
        first_ards_days = self._cal_first_ards_days(self.data_pd, self.fea_manager.get_expanded_fea(self.target_fea))
        tools.plot_single_dist(first_ards_days, f"First ARDS Days", \
            save_path=os.path.join(out_dir, 'first_ards_days.png'))
        # random plot sample time series
        self._plot_time_series_samples(self.target_fea, n_sample=400, n_per_plots=40, \
            write_dir=os.path.join(out_dir, "samples_plot"))
        # plot fourier tranform result
        self._plot_fourier_transform(self.target_fea, self.target_time_arr[:,1], save_dir=out_dir)

    
    '''
    生成动态模型所需的时间切片
    mode: 
        'target_time' 适用于只看目标历史数据的方法
        'k_slice' 适用于只用T-k天预测第T天数据的方法
    '''
    def make_slice(self, mode:str='target_time', k=None)->dict:
        # 将开始和持续时间变成整数
        step = times[1] - times[0]
        start_idx = np.round(self.target_time_arr[:,0] / step)
        dur_len = np.round(self.target_time_arr[:,1] / step)
        if mode == 'target_time':
            expanded = self.fea_manager.get_expanded_fea(self.target_fea)
            names = [val[1] for val in expanded]
            times = np.asarray([val[0] for val in expanded])
            result = self.data_pd[names].to_numpy(dtype=float)
            assert(start_idx.shape[0] == dur_len.shape[0] and start_idx.shape[0] == result.shape[0])
            return {'data':result, 'start_idx': start_idx, 'dur_len':dur_len} # dict{key:ndarray}
        elif mode == 'k_slice':
            assert(k is not None)
            data = self.data_pd[dur_len > k, :]
            dur_len = dur_len[dur_len > k]
            start_idx = start_idx[dur_len > k]
            data.reset_index(drop=True, inplace=True)
            sta_names = set(self.fea_manager.get_names(sta=True))
            dyn_names = set(self.fea_manager.get_names(dyn=True))
            dyn_dict = {key:[val[1] for val in self.fea_manager.get_expanded_fea(key)] for key in dyn_names} # old name
            dyn_names.remove(self.target_fea)
            result = pd.DataFrame(columns=sta_names + dyn_names)
            # 0 1 2 3 4
            # 0 1 2
            for r_idx in range(len(data)):
                for delta in range(dur_len[r_idx] - k): # duration=N, k=1, delta=0,1,2,...,N-2
                    new_row = {}
                    for name in sta_names:
                        new_row[name] = data[r_idx, name]
                    for name in dyn_names:
                        new_row[name] = data[r_idx, dyn_dict[name][start_idx[r_idx] + delta]]
                    new_row[self.target_fea] = data[r_idx, dyn_dict[self.target_fea][start_idx[r_idx] + delta + k]]
                    result.loc[len(result)] = new_row
            logger.info(f'Extended Datasets size={len(result)}, with {len(result.columns)} features')
            # generate new type dict
            new_type_dict = self.generate_dyn_type_dict()
            for name in result.columns:
                if name not in dyn_dict.keys():
                    new_type_dict[name] = self.dataset.type_dict[name]
            return {'data':result, 'type_dict':new_type_dict}


    def generate_dyn_type_dict(self) -> dict:
        dyn_names = set(self.fea_manager.get_names(dyn=True))
        result = {}
        for name in dyn_names:
            result[name] = self.dataset.type_dict[self.fea_manager.get_expanded_fea(name)[0,1]]
        return result

    def _cal_first_ards_days(self, data:pd.DataFrame, expanded_target:list):
        assert(isinstance(expanded_target[0], tuple))
        # expanded_target = self.fea_manager.get_expanded_fea(dyn_fea)
        offset = expanded_target[1][0]
        names = [val[1] for val in expanded_target]
        valid_mat = (1 - data[names].isna()).astype(bool)
        result = -np.ones((len(data))) # start_time, duration
        for r_idx in range(len(data)):
            for time, name in expanded_target:
                if bool(valid_mat.at[r_idx, name]) is True and data.at[r_idx, name] < 300:
                    result[r_idx] = (time + offset)/24
                    break
        return result[result > -0.5]
    
    # 打印若干个sample的时间趋势
    def _plot_time_series_samples(self, dyn_name, n_sample:int=100, n_per_plots:int=10, write_dir=None):
        if write_dir is not None:
            tools.reinit_dir(write_dir)
        expanded = self.fea_manager.get_expanded_fea(dyn_name=dyn_name)
        names = [val[1] for val in expanded]
        times = np.asarray([val[0] for val in expanded])
        data_arr = self.data_pd[names].to_numpy(dtype=float)
        n_plot = int(np.ceil(n_sample / n_per_plots))
        idx = 0
        for p_idx in range(n_plot):
            stop_idx = min(idx + n_per_plots, n_sample)
            plt.plot(times, data_arr[idx:stop_idx, :].T, alpha=0.3)
            plt.title(f"Time series sample for {dyn_name}")
            plt.xlabel("time/hour")
            plt.ylabel(dyn_name)
            idx = stop_idx
            if write_dir is None:
                plt.show()
            else:
                plt.savefig(os.path.join(write_dir, f"plot_{p_idx}.png"))
            plt.close()

    def _plot_fourier_transform(self, fea_name, target_time_arr, save_dir=None):
        # 只拿天数=7的样本作傅里叶变换
        exp_fea = [val[1] for val in self.fea_manager.get_expanded_fea(self.target_fea)]

        data = self.data_pd.loc[target_time_arr > 24*6.5, exp_fea].to_numpy(dtype=float)
        result = np.log10(np.abs(np.fft.fft(data, axis=1))) # log amp
        result_mean = np.mean(result, axis=0)
        result_std = np.std(result, axis=0)
        freq = np.asarray([0.5*val/len(exp_fea) for val in range(len(exp_fea))])
        fig, ax = plt.subplots()
        ax.plot(freq, result_mean, 'b+-')
        ax.errorbar(freq, result_mean, result_std, capsize=4, ecolor='C0')
        plt.title(f"Frequent amplitude for target feature with std")
        plt.xlabel("frequency/day^-1")
        plt.ylabel('Log amplitude')
        if save_dir is None:
            plt.show()
        else:
            plt.savefig(os.path.join(save_dir, 'frequency_energy.png'))
        plt.close()







if __name__ == "__main__":
    dataset = DynamicSepsisDataset(from_pkl=True)
    analyzer = DynamicAnalyzer(dataset=dataset)
    analyzer.feature_explore()