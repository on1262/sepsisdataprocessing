'''
动态模型baseline
输入dict见analyzer::make_slice
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
import tools

'''
    简单的时序预测
    需要目标量的历史值, 有三种预测方式: nearest, average, holt
    对于每个可以预测的时间节点都会预测, 而不是只预测最后一个点
    输入数据需要每个duration大于等于2
'''
class SimpleTimeSeriesPredictor:
    def predict(self, data:np.ndarray, start_idx:np.ndarray, duration:np.ndarray, mode:str=['nearest', 'average', 'holt'], params=None)->np.ndarray:
        '''
        data: (sample, ticks) 只有目标特征
        '''
        result = -np.ones((data.shape[0],data.shape[1]), dtype=float)
        if mode == 'holt':
            holt_params = params['holt']
            alpha=holt_params['alpha']
            beta=holt_params['beta']
            step=holt_params['step']
            s_mat = np.zeros(data.shape)
            t_mat = np.zeros(data.shape)
            for r_idx in range(len(data)):
                s = start_idx[r_idx]
                d = duration[r_idx]
                if d >= 2: # 趋势至少需要两个已知点
                    s_mat[r_idx, s] = data[r_idx, s]
                    t_mat[r_idx, s] = 0 # 使得beta=0时退化到无趋势的指数平滑
                    for t_idx in range(s+1, s+d, 1):
                        s_mat[r_idx, t_idx] = alpha*data[r_idx, t_idx] + (1-alpha)*(s_mat[r_idx, t_idx-1] + t_mat[r_idx, t_idx-1]) # a*x_i+(1-a)*(s_i-1+t_i-1)
                        t_mat[r_idx, t_idx] = beta*(s_mat[r_idx, t_idx] - s_mat[r_idx, t_idx-1]) + (1-beta)*t_mat[r_idx, t_idx-1] # b*(s_i-s_i-1)+(1-beta)*t_i-1
                        result[r_idx, t_idx] = s_mat[r_idx, t_idx-1] + step*t_mat[r_idx, t_idx-1] # h_step: s_i+h*t_i
        
        elif mode == 'nearest':
            for r_idx in range(len(data)):
                s,d = start_idx[r_idx],duration[r_idx]
                if d >= 2:
                    result[r_idx, s+1:s + d] = data[r_idx, s:s+d-1]
        elif mode == 'average':
            for r_idx in range(len(data)):
                s,d = start_idx[r_idx],duration[r_idx]
                if d >= 2:
                    for offset in range(s+1, s+d,1):
                        result[r_idx, offset] = np.mean(data[r_idx, s:offset])
        return result

class SliceLinearRegression:
    def __init__(self, type_dict:dict, params=None) -> None:
        self.model = LinearRegression()
        self.type_dict = type_dict
        self.params = params
        # metadata generated in training phase
        self.norm_dict = None
        self.avg_dict = None
        self.ts_val = None
        self.ctg_feas = None
        self.num_feas = None
        self.columns = None

    def train(self, X_train:pd.DataFrame, Y_train:pd.DataFrame):
        ctg_feas = []
        num_feas = []
        for idx, col in enumerate(X_train.columns):
            if self.type_dict[col] == str:
                ctg_feas.append(idx)
            else:
                num_feas.append(idx)
        self.ctg_feas, self.num_feas = ctg_feas, num_feas
        self.columns = list(X_train.columns)
        X_train, self.ts_val = tools.target_statistic(X_train.to_numpy(), Y_train.to_numpy(), ctg_feas=ctg_feas, mode=self.params['ts_mode'])
        X_train, self.avg_dict = tools.fill_avg(X_train, num_feas=num_feas)
        # normalize时前面TS的也要考虑, 所以不是原来的num_feas
        X_train, self.norm_dict = tools.feature_normalization(X_train, num_feas=list(range(len(self.columns))), norm_dict=None)
        assert(not np.isnan(X_train).any())
        self.model.fit(X_train, Y_train)


    def predict(self, X_test:pd.DataFrame):
        assert(list(X_test.columns) == self.columns) # 确保序号对的上
        X_test = tools.target_statistic(X_test.to_numpy(), Y=None, ctg_feas=self.ctg_feas, mode=self.params['ts_mode'], hist_val=self.ts_val)
        X_test = tools.fill_avg(X_test, num_feas=self.num_feas, avg_dict=self.avg_dict)
        # normalize时前面TS的也要考虑, 所以不是原来的num_feas
        X_test = tools.feature_normalization(X_test, num_feas=list(range(len(self.columns))), norm_dict=self.norm_dict)
        Y_pred = self.model.predict(X_test)
        return Y_pred

    def map_result(self, Y_pred, result, map_table, index):
        for idx in range(result.shape[0]):
            r_idx, c_idx = map_table[index[idx]][0],map_table[index[idx]][1]
            Y_pred[r_idx, c_idx] = result[idx]
        return Y_pred

class SliceCatboostRegression:
    def __init__(self, type_dict:dict, params:dict=None) -> None:
        # param['quantile'] = [0.25, 0.5, 0.75]
        # reference: https://catboost.ai/en/docs/concepts/loss-functions-regression
        # example: Quantile:alpha=0.1
        self.quantile = None
        self.models = []
        if 'quantile' in params.keys():
            self.quantile = params['quantile']
            assert(np.abs(self.quantile[round((len(self.quantile) - 1) / 2)] - 0.5) < 1e-4) # 中间一定是0.5, 即MAE
            for alpha in params['quantile']:
                self.models.append(CatBoostRegressor(
                    iterations=params['iterations'],
                    depth=params['depth'],
                    loss_function=f'Quantile:alpha={alpha:.2f}',
                    learning_rate=params['learning_rate'],
                    od_type = params["od_type"],
                    od_wait = params["od_wait"],
                    verbose=params['verbose']))
        else:
            self.models.append(CatBoostRegressor(
                iterations=params['iterations'],
                depth=params['depth'],
                loss_function=params['loss_function'],
                learning_rate=params['learning_rate'],
                od_type = params["od_type"],
                od_wait = params["od_wait"],
                verbose=params['verbose']))
        self.type_dict = type_dict
        self.params = params
        # metadata generated in training phase
        self.ts_val = None
        self.ctg_feas = None
        self.columns = None
        self.pool_valid = None
        self.X_valid = None

    def train(self, X_train:pd.DataFrame, Y_train:pd.DataFrame):
        ctg_feas = []
        for idx, col in enumerate(X_train.columns):
            if self.type_dict[col] == str:
                ctg_feas.append(idx)
        self.ctg_feas = ctg_feas
        self.columns = list(X_train.columns)
        X_train, self.ts_val = tools.target_statistic(X_train.to_numpy(), Y_train.to_numpy(), ctg_feas=ctg_feas, mode=self.params['ts_mode'])
        assert(not np.isnan(X_train).any())
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.15)
        pool_train = Pool(X_train, Y_train)
        pool_valid = Pool(X_valid, Y_valid)
        self.pool_valid = pool_valid
        self.X_valid = X_valid
        if self.quantile is None:
            self.models[0].fit(pool_train, eval_set=pool_valid, use_best_model=True)
        else:
            for model in self.models:
                model.fit(pool_train, eval_set=pool_valid, use_best_model=True)

    def model_explanation(self):
        if self.quantile:
            idx = round((len(self.quantile) - 1) / 2)
        else:
            idx = 0
        shap_array, shap, sorted_names = tools.test_fea_importance(self.models[idx], self.pool_valid, self.columns)
        return shap_array, shap, sorted_names

    
    def predict(self, X_test:pd.DataFrame) -> np.ndarray:
        assert(list(X_test.columns) == self.columns) # 确保序号对的上
        X_test = tools.target_statistic(X_test.to_numpy(), Y=None, ctg_feas=self.ctg_feas, mode=self.params['ts_mode'], hist_val=self.ts_val)
        pool_test = Pool(data=X_test)
        if self.quantile is None:
            Y_pred = self.models[0].predict(pool_test)
            return Y_pred
        else:
            Y_pred = np.asarray([model.predict(pool_test) for model in self.models])
            return Y_pred

    def map_result(self, Y_pred, result, map_table, index):
        if self.quantile is None:
            for idx in range(result.shape[0]):
                r_idx, c_idx = map_table[index[idx]][0],map_table[index[idx]][1]
                Y_pred[r_idx, c_idx] = result[idx]
        else:
            for idx in range(result.shape[1]):
                r_idx, c_idx = map_table[index[idx]][0],map_table[index[idx]][1]
                Y_pred[:, r_idx, c_idx] = result[:, idx]
        return Y_pred