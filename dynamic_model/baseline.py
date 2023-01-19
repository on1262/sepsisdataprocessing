'''
动态模型baseline
输入dict见analyzer::make_slice
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor, Pool

'''
    简单的时序预测
    需要目标量的历史值, 有三种预测方式: nearest, average, holt
'''
class SimpleTimeSeriesPredictor:
    def predict(self, data:np.ndarray, start_idx:np.ndarray, duration:np.ndarray, mode:str=['nearest', 'average', 'holt'], params=None)->np.ndarray:
        result = np.empty((data.shape[0],), dtype=float)
        if mode == 'holt':
            alpha=params['alpha']
            beta=params['beta']
            step=params['step']
            s_mat = np.zeros(data.shape)
            t_mat = np.zeros(data.shape)
        for r_idx in range(len(data)):
            if mode == 'holt':
                if duration[r_idx] >= 2:
                    s_mat[r_idx, start_idx] = data[r_idx, start_idx]
                    t_mat[r_idx, start_idx] = data[r_idx, start_idx+1] - data[r_idx, start_idx]
                    for t_idx in range(start_idx+1, start_idx+duration, 1):
                        s_mat[r_idx, t_idx] = alpha*data[r_idx, t_idx] + (1-alpha)*(s_mat[r_idx, t_idx-1] + t_mat[r_idx, t_idx-1]) # a*x_i+(1-a)*(s_i-1+t_i-1)
                        t_mat[r_idx, t_idx] = beta*(s_mat[r_idx, t_idx] - s_mat[r_idx, t_idx-1]) + (1-beta)*t_mat(r_idx, t_idx-1) # b*(s_i-s_i-1)+(1-beta)*t_i-1
                    end_idx = start_idx+duration-1
                    result[r_idx] = s_mat[r_idx, end_idx] + step*t_mat[r_idx, end_idx] # h_step: s_i+h*t_i
                else: # only one point
                    result[r_idx] = data[r_idx, start_idx + duration - 1]
            else:
                if mode == 'nearest':
                    result[r_idx] = data[r_idx, start_idx + duration - 1]
                elif mode == 'average':
                    result[r_idx] = np.mean(data[r_idx, start_idx:start_idx + duration])
        return result

class SliceLinearRegression:
    def __init__(self, type_dict:dict) -> None:
        self.model = LinearRegression(solver='lbfgs', max_iter=5000)
        self.normalization_dict = {}
        self.type_dict = type_dict
        self.target_statistic = None

    def train(self, train_data:pd.DataFrame):
        pass


    def predict(self, data:pd.DataFrame):
        pass
           
           
           
                    



