'''
动态模型baseline
输入dict见analyzer::make_slice
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor, Pool
import tools

'''
    简单的时序预测
    需要目标量的历史值, 有三种预测方式: nearest, average, holt
    对于每个可以预测的时间节点都会预测, 而不是只预测最后一个点
    输入数据需要每个duration大于等于2
'''
class SimpleTimeSeriesPredictor:
    def predict(self, data:np.ndarray, start_idx:np.ndarray, duration:np.ndarray, mode:str=['nearest', 'average', 'holt'], params=None)->np.ndarray:
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
                    result[r_idx, s+1] = data[r_idx, s] # 第二个点
                    s_mat[r_idx, s] = data[r_idx, s]
                    t_mat[r_idx, s] = data[r_idx, s+1] - data[r_idx, s]
                    for t_idx in range(s+2, s+d, 1): # 带趋势的预测会泄露第一个和第二个点, 所以只能从第三个点开始预测
                        s_mat[r_idx, t_idx] = alpha*data[r_idx, t_idx] + (1-alpha)*(s_mat[r_idx, t_idx-1] + t_mat[r_idx, t_idx-1]) # a*x_i+(1-a)*(s_i-1+t_i-1)
                        t_mat[r_idx, t_idx] = beta*(s_mat[r_idx, t_idx] - s_mat[r_idx, t_idx-1]) + (1-beta)*t_mat[r_idx, t_idx-1] # b*(s_i-s_i-1)+(1-beta)*t_i-1
                        result[r_idx, t_idx] = s_mat[r_idx, t_idx-1] + step*t_mat[r_idx, t_idx-1] # h_step: s_i+h*t_i
                    
        elif mode == 'nearest':
            for r_idx in range(len(data)):
                s,d = start_idx[r_idx],duration[r_idx]
                if d >= 2:
                    result[r_idx, s+1:s + d] = \
                        data[r_idx, s:s+d-1]
        elif mode == 'average':
            for r_idx in range(len(data)):
                s,d = start_idx[r_idx],duration[r_idx]
                if d >= 2:
                    result[r_idx, s+1:s+d] = \
                        np.mean(data[r_idx, s:s+d-1])
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

    def train(self, X_train:pd.DataFrame, Y_train:pd.DataFrame):
        ctg_feas = []
        num_feas = []
        for idx, col in enumerate(X_train.columns):
            if self.type_dict[col] == str:
                ctg_feas.append(idx)
            else:
                num_feas.append(idx)
        self.ctg_feas, self.num_feas = ctg_feas, num_feas
        X_train, self.ts_val = tools.target_statistic(X_train.to_numpy(), Y_train.to_numpy(), ctg_feas=ctg_feas, mode=self.params['ts_mode'])
        X_train, self.avg_dict = tools.fill_avg(X_train, num_feas=num_feas)
        X_train, self.norm_dict = tools.feature_normalization(X_train, num_feas=num_feas, norm_dict=None)
        assert(not np.isnan(X_train).any())
        self.model.fit(X_train, Y_train)


    def predict(self, X_test:pd.DataFrame):
        X_test = tools.target_statistic(X_test.to_numpy(), Y=None, ctg_feas=self.ctg_feas, mode=self.params['ts_mode'], hist_val=self.ts_val)
        X_test = tools.fill_avg(X_test, num_feas=self.num_feas, avg_dict=self.avg_dict)
        X_test = tools.feature_normalization(X_test, num_feas=self.num_feas, norm_dict=self.norm_dict)
        Y_pred = self.model.predict(X_test)
        return Y_pred

    def map_result(self, Y_pred, result, map_table, index):
        for idx in range(result.shape[0]):
            r_idx, c_idx = map_table[index[idx]][0],map_table[index[idx]][1]
            Y_pred[r_idx, c_idx] = result[idx]
        return Y_pred


        
           
           
           
                    



