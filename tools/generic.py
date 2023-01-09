import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy
from sklearn import random as sk_random
from sklearn.metrics import auc as sk_auc
import pandas as pd
from collections.abc import Iterable
import os, sys
import subprocess
import missingno as msno

'''
清除并且重建一个文件夹和其中所有的内容
'''
def reinit_dir(write_dir_path=None):
    if write_dir_path is not None:
        if os.path.exists(write_dir_path):
            for name in os.listdir(write_dir_path):
                os.remove(os.path.join(write_dir_path, name))
        os.makedirs(write_dir_path, exist_ok=True)

'''
设置matplotlib显示中文, 对于pandas_profile不可用
'''
def set_chinese_font():
    from matplotlib import pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

#  清空文件
def flush_log_file(log_file):
    with open(log_file, 'w+'):
        pass

def set_sk_random_seed(seed:int=100):
    sk_random.seed(seed)

'''
计算ROC, 返回fpr值和tpr值
'''
def cal_roc(Y_test, Y_pred, n_thres=11, log_file='ROC_cal.log', comment=""):
    with open(log_file, 'a') as f:
        f.write(f'Start {comment} \n')
        tpr = []
        fpr = []
        thres_list = np.linspace(0,1, num=n_thres)
        for thres in thres_list:
            tp = np.sum(Y_pred[Y_test > 0.5] > thres)
            fp = np.sum(Y_pred[Y_test < 0.5] > thres)
            fn = np.sum(Y_pred[Y_test > 0.5] < thres)
            tn = np.sum(Y_pred[Y_test < 0.5] < thres)
            f.write(f"tp={tp}, fp={fp}, fn={fn}, tn={tn}, acc={(tp+tn)/(tp+fp+tn+fn):.3f}, tpr={(tp/(tp+fn)):.3f}, fpr={(fp/(fp+tn)):.3f}, sens={(tp/(tp+fn)):3f}, spec={(tn/(tn+fp)):.3f}, thres={thres:3f}\n")
            tpr.append(tp/(tp+fn))
            fpr.append(fp/(fp+tn))
    fpr, tpr = np.asarray(fpr), np.asarray(tpr)
    
    return fpr, tpr, thres_list


def select_na(data:pd.DataFrame, col_thres=0.5, row_thres=0.7): # 获取超过thres的非na比例的数据
    # del column na
    df = msno.nullity_filter(data, filter='top', p=col_thres)
    data = data[df.columns]
    # del row na
    na_mat = data.isna().to_numpy(dtype=np.int32)
    valid_mat = 1 - np.mean(na_mat, axis=1)
    data = data.iloc[valid_mat > row_thres]
    return data

def remove_invalid_rows(data:pd.DataFrame, type_dict:dict) -> pd.DataFrame:
    na_table = data.isna()
    select_table = pd.Series([True for _ in range(len(data.index))], index=data.index, name='bools')
    for col in data.columns:
        for idx in data.index:
            if not na_table.at[idx, col]:
                try:
                    tmp = type_dict[col](data.at[idx, col])
                except Exception as e:
                    select_table[idx] = False
    data = data[select_table.values]
    return data

# all type will be converted to str/float when data is sent to trainer
def convert_type(na_values:dict, data:pd.DataFrame, type_dict:dict):
    assert('_default' in na_values)
    default_val = na_values['_default']
    data = data.fillna(value=na_values.copy().pop('_default'))
    data.fillna(value=default_val, inplace=True)
    apply_dict = type_dict.copy()
    for key in apply_dict.keys():
        if apply_dict[key] != str:
            apply_dict[key] = float
    data = data.astype(dtype=apply_dict, copy=True)
    return data

# TODO: convention should rely on multiple samples. Feature type is the marjority type of a subset samples.
# Detailed feature type is necessary to detect illegal data, which could be a float in a bool type feature.
def check_fea_types(data:pd.DataFrame) -> dict:
    eps = 1e-5
    type_dict = {}
    nas = data.isna()
    priority = {float:0, int:1, bool:2, str:3}

    for col in data.columns: # prevent bad data of str types. Manual check is needed if other type occurs
        type_dict[col] = None
        # random sample 100 non-na values:
        na_fea = np.asarray(nas[col])
        for idx, val in enumerate(data[col]):
            
            if na_fea[idx] == False:
                s_val = str(val)
                num_flag = True
                try:
                    float(s_val)
                except ValueError:
                    num_flag = False
                if num_flag: # str will be covered by numeric
                    f_val = float(s_val)
                    if abs(f_val - round(f_val)) > eps: # bool, int will be covered by float
                        type_dict[col] = (float, priority[float])
                    elif int(f_val) not in [0,1]: # bool will be covered by int
                        if type_dict[col] is None or type_dict[col][1] > priority[int]:
                            type_dict[col] = (int, priority[int])
                    elif type_dict[col] is None or type_dict[col][1] > priority[bool]:
                        type_dict[col] = (bool, priority[bool])
                else:
                    if type_dict[col] is None:
                        type_dict[col] = (str, priority[str])
    for key in type_dict.keys():
        type_dict[key] = type_dict[key][0]
    return type_dict

# 将str类型的特征解析成有限个类别
def detect_category_fea(data:pd.DataFrame, type_dict:dict, cluster_perc=0.05):
    category_dict = {}
    n_sample = len(data)
    # build categories
    for name in type_dict.keys():
        if type_dict[name] == str:
            nd = {}
            for val in data[name]:
                if val not in nd.keys():
                    nd[val] = 1
                else:
                    nd[val] += 1
            # sort
            classes = list(nd.keys())
            classes = sorted(classes, key=lambda n:nd[n], reverse=True)
            for key in nd.keys():
                if nd[key] / n_sample > cluster_perc:
                    nd[key] = key
                else:
                    nd[key] = 'OTHERS'
            category_dict[name] = nd
    return category_dict

# 合并长尾类别
def apply_category_fea(data:pd.DataFrame, category_dict:dict):
    # apply categories
    for name in category_dict.keys():
        for idx, _ in data.iterrows():
            data.at[idx,name] = category_dict[name][data.at[idx, name]]
    return data
    
def create_4_cls_label(y_ards:np.ndarray, y_death:np.ndarray):
    Y_4cls = (y_ards * y_death) + 2*(y_ards * np.invert(y_death)) + 3*((np.invert(y_ards) * y_death))
    print('4 cls label:')
    print('+ards+death:', (Y_4cls == 1).mean())
    print('+ards-death:', (Y_4cls == 2).mean())
    print('-ards+death:', (Y_4cls == 3).mean())
    print('-ards-death:', (Y_4cls == 0).mean())
    return Y_4cls


def one_hot_decoding(data:pd.DataFrame, cluster_dict:dict):
    print('one hot decoding')
    # cluster_dict = {'new_name1':{'old1','old2',...}}
    for new_name in cluster_dict.keys():
        new_col = [0 for _ in range(len(data))]
        old_names = cluster_dict[new_name]
        cols = data[old_names] # select columns
        valids = cols.isna().to_numpy(dtype=np.int32)
        for idx in range(len(data)):
            row = cols.loc[idx].to_numpy(dtype=bool)
            row = row * (1 - valids[idx,:])
            row = np.asarray(row, dtype=bool)
            if row.sum() == 1:
                new_col[idx] = np.asarray(old_names)[row][0]
            else:
                new_col[idx] = '+'.join(np.asarray(old_names)[row])
        data[new_name] = new_col
        data.drop(old_names, axis=1, inplace=True)
    return data

def fill_default(data:pd.DataFrame, default_dict:dict):
    na_dict = {}
    for key in default_dict.keys():
        assert(key in data.columns)
        if len(default_dict[key]) == 1:
            na_dict[key] = default_dict[key][0]
        else:
            data.replace(default_dict[key][0], {key:default_dict[key][1]}, inplace=True)
    data.fillna(value=na_dict, inplace=True)

def normalize(x:np.ndarray, axis=1):
    x_std = x.std(axis=1,keepdims=True)
    # x_std = np.clip(x_std, a_min=1e-3, a_max=1000)
    x = (x - x.mean(axis=axis, keepdims=True)) / x_std
    return x

def target_statistic(X:np.ndarray, Y:np.ndarray, ctg_feas:list, mode='greedy'):
    X_ts = X.copy()
    assert(mode=='greedy')
    hist_val = {}
    for r_idx in range(X.shape[0]):
        for c_idx in ctg_feas:
            if c_idx not in hist_val.keys():
                hist_val[c_idx] = {}
            if X[r_idx, c_idx] not in hist_val[c_idx].keys():
                hist_val[c_idx][X[r_idx, c_idx]] = (Y[r_idx],1)
            else:
                val,count = hist_val[c_idx][X[r_idx, c_idx]]
                hist_val[c_idx][X[r_idx, c_idx]] = (val+Y[r_idx], count+1)

    alpha = round(0.01*X.shape[0])
    prior = Y.mean()
    for ctg_fea_idx in hist_val:
        for cls in hist_val[ctg_fea_idx]:
            val, count = hist_val[ctg_fea_idx][cls] 
            hist_val[ctg_fea_idx][cls] = (val + alpha * prior) / (count + alpha)
    for r_idx in range(X.shape[0]):
        for c_idx in ctg_feas:
            X_ts[r_idx, c_idx] = hist_val[c_idx][X[r_idx, c_idx]]
    X_ts = X_ts.astype(dtype=np.float32)
    return X_ts


# use feature average value to fill na values, used for better linear model performance
def fill_avg(X:np.ndarray, num_feas:list, na_sign=-1, eps=1e-3):
    X_avg = X.copy()
    avg_dict = {}
    for c_idx in num_feas:
        choosed = (X_avg[:,c_idx] < na_sign + eps) * (X_avg[:,c_idx] > na_sign - eps)
        avg_dict[c_idx] = X_avg[np.invert(choosed), c_idx].mean()
        X_avg[choosed, c_idx] = avg_dict[c_idx]
    return X_avg, avg_dict


