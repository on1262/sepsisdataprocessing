import tools
from dynamic_sepsis_dataset import DynamicSepsisDataset
import os
import numpy as np
import pandas as pd
from tools import logger as logger
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from dynamic_model import baseline as Baseline


class DynamicAnalyzer:
    def __init__(self, dataset:DynamicSepsisDataset):
        self.dataset = dataset
        self.conf = tools.GLOBAL_CONF_LOADER['dynamic_analyzer']
        self.data_pd = dataset.data_pd
        self.type_dict = dataset.get_type_dict()
        self.target_fea = dataset.target_fea # only one name
        self.dynamic_target_name = self.conf['dynamic_target_name'] # 这个是动态模型的k+1天特征
        self.fea_manager = dataset.fea_manager
        self.n_fold = 5

        # for feature importance
        self.register_values = {}
        self.shap_names = self.conf['shap_names'] # value是显示范围, 手动设置要注意左边-1项不需要, 右边太大的项要限

    '''
        生成数据集的总体特征, 这个方法不是必须运行的
    '''
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

    def baseline_methods(self, models:set, params=None):
        kf = KFold(n_splits=self.n_fold, shuffle=True)
        for model_name in models:
            self.register_values.clear()
            tools.reinit_dir(os.path.join(self.conf['paths']['out_dir'], model_name), build=True)
            log_path = os.path.join(self.conf['paths']['out_dir'], model_name, 'result.log')
            tools.clear_file(log_path)
            logger.info(f'Evaluating baseline methods:{model_name}')
            metric = tools.DynamicPredictionMetric(target_name=self.target_fea,
                expanded_fea=self.fea_manager.get_expanded_fea(self.target_fea), out_dir=self.conf['paths']['out_dir'])
            if 'simple' in model_name: # simple_nearest simple_average simple_holt
                predictor = Baseline.SimpleTimeSeriesPredictor()
                slice_dict = self.dataset.target_time_dict
                dataset = slice_dict['data'].copy()
                dataset = tools.convert_type({'_default': -1}, dataset, {k:float for k in dataset.columns}).to_numpy()
                for idx, (train_index, test_index) in enumerate(kf.split(X=dataset)):
                    X_test = dataset[test_index, :]
                    start_idx_test = slice_dict['start_idx'][test_index]
                    duration_test = slice_dict['dur_len'][test_index]
                    result = predictor.predict(X_test, start_idx=start_idx_test, duration=duration_test, mode=model_name.split('simple_')[1], params=params)
                    metric.add_prediction(prediction=result, gt=X_test, start_idx=start_idx_test, duration=duration_test)

            elif model_name == 'slice_linear_reg':
                slice_dict = self.dataset.slice_dict
                dataset = slice_dict['data'].copy()
                dataset = tools.convert_type({'_default': -1}, dataset, slice_dict['type_dict'])
                tools.assert_no_na(dataset)
                Y_pred = -np.ones(slice_dict['gt_table'].shape)
                train_cols = list(dataset.columns)
                train_cols.remove(self.dynamic_target_name)
                for idx, (train_index, test_index) in enumerate(kf.split(X=dataset)):
                    model = Baseline.SliceLinearRegression(slice_dict['type_dict'], params=params['slice_linear_reg'])
                    X_train = dataset.loc[train_index, train_cols]
                    X_test = dataset.loc[test_index, train_cols]
                    Y_train = dataset.loc[train_index, self.dynamic_target_name]
                    model.train(X_train, Y_train)
                    result = model.predict(X_test=X_test)
                    Y_pred = model.map_result(Y_pred=Y_pred, result=result, map_table=slice_dict['map_table'], index=test_index)
                metric.add_prediction(
                    prediction=Y_pred, gt=slice_dict['gt_table'].to_numpy(), start_idx=slice_dict['start_idx'], duration=slice_dict['dur_len'])

            elif model_name == 'slice_catboost_reg':
                slice_dict = self.dataset.slice_dict
                dataset = slice_dict['data'].copy()
                dataset = tools.convert_type({'_default': -1}, dataset, slice_dict['type_dict'])
                tools.assert_no_na(dataset)
                Y_pred = -np.ones(slice_dict['gt_table'].shape)
                train_cols = list(dataset.columns)
                train_cols.remove(self.dynamic_target_name)
                for idx, (train_index, test_index) in enumerate(kf.split(X=dataset)):
                    model = Baseline.SliceCatboostRegression(slice_dict['type_dict'], params=params['slice_catboost_reg'])
                    X_train = dataset.loc[train_index, train_cols]
                    X_test = dataset.loc[test_index, train_cols]
                    Y_train = dataset.loc[train_index, self.dynamic_target_name]
                    model.train(X_train, Y_train)
                    result = model.predict(X_test=X_test)
                    Y_pred = model.map_result(Y_pred=Y_pred, result=result, map_table=slice_dict['map_table'], index=test_index)
                    shap_array, shap, sorted_names = model.model_explanation()
                    # 注册每个样本的每个特征对应的shap值
                    if idx == 0:
                        self.register_values['shap'] = {}
                        self.register_values['shap_arr'] = {}
                    for fea_idx, name in enumerate(train_cols):
                        if name not in self.shap_names.keys():
                            continue
                        pairs = np.concatenate((shap_array[:, [fea_idx]], model.X_valid[:, [fea_idx]].astype(np.float32)), axis=1)
                        if self.register_values['shap_arr'].get(name) is None:
                            self.register_values['shap_arr'][name] = pairs
                        else:
                            self.register_values['shap_arr'][name] = np.concatenate((self.register_values['shap_arr'][name], pairs), axis=0)
                    # 将每次得到的shap值相加, 便于计算k_fold的平均重要性
                    for i in range(len(shap)):
                        if self.register_values['shap'].get(sorted_names[i]) is None:
                            self.register_values['shap'][sorted_names[i]] = shap[i]
                        else:
                            self.register_values['shap'][sorted_names[i]] += shap[i]
                    self._plot_fea_importance(model_name)
                metric.add_prediction(
                    prediction=Y_pred, gt=slice_dict['gt_table'].to_numpy(), start_idx=slice_dict['start_idx'], duration=slice_dict['dur_len'])
            metric.write_result(model_name, log_path=log_path)
            metric.plot(model_name)
        self.create_final_result()

    # 收集各个文件夹里面的result.log, 合并为final result
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
                



    def _plot_fea_importance(self, model_name):
        # 得到整体的特征重要性表
        items = list(self.register_values['shap'].items())
        items = np.asarray(sorted(items, key= lambda x:x[1]))
        shap_vals = np.asarray(items[:, 1], dtype=np.float32) / self.n_fold
        with open(os.path.join(self.conf['paths']['out_dir'], model_name,'shap.log'), 'w') as fp:
            fp.write(f'Slice CatBoost Regressor feature importance \n')
            for i in reversed(range(shap_vals.shape[0])):
                fp.write(f"{shap_vals[i]},{str(items[i,0])}\n")
        tools.plot_fea_importance(shap_vals[-10:], items[-10:, 0], save_path=os.path.join(self.conf['paths']['out_dir'], model_name,'shap.png'))
        # 给出单个特征的重要性图
        if 'shap_arr' in self.register_values.keys():
            for fea_name, vals in self.register_values['shap_arr'].items():
                tools.plot_shap_scatter(fea_name, vals[:, 0], vals[:, 1], self.shap_names[fea_name], 
                    os.path.join(self.conf['paths']['out_dir'], model_name))
    
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
    baseline_models = {
        'simple_nearest',
        'simple_average',
        'simple_holt',
        'slice_linear_reg',
        'slice_catboost_reg'
    }
    params = {
        'holt':{
            'alpha': 0.5,
            'beta':0.05,
            'step':1
        },
        'slice_linear_reg':{
            'ts_mode':'greedy'
        },
        'slice_catboost_reg':{
            'ts_mode':'greedy',
            'loss_function': 'RMSE',
            'iterations': 400,
            'depth': 5,
            'learning_rate': 0.04,
            'od_type' : "Iter",
            'od_wait' : 100,
            'verbose': 1
        }
    }
    # module test
    # analyzer.feature_explore()
    analyzer.baseline_methods(models=baseline_models, params=params)
    # analyzer.baseline_methods(models={'slice_linear_reg'}, params=params)
    # analyzer.baseline_methods(models={}, params=params)

