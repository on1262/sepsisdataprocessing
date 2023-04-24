import numpy as np
import json
import datetime
from sklearn import random as sk_random
import pandas as pd
import re
import os, sys
import missingno as msno
import hashlib
from .colorful_logging import logger

'''
    管理静态特征+动态特征
'''
class FeatureManager():
    def __init__(self):
        self.dyn_fea_names = {} # key=name, value=time list
        self.sta_fea_names = []
        # attributes below are dirty updated. They should be accessed by function
        
    # origin_name: 和dataframe的columns保持一致, fea_name和configs保持一致
    def add_dyn(self, origin_name:str, fea_name:str, time=float):
        if self.dyn_fea_names.get(fea_name) is None:
            self.dyn_fea_names[fea_name] = [(time, origin_name)]
        else:
            self.dyn_fea_names[fea_name].append((time, origin_name))
            # 保持序列整齐
            self.dyn_fea_names[fea_name] = sorted(self.dyn_fea_names[fea_name], key=lambda x: x[0])
    
    def add_sta(self, fea_name):
        self.sta_fea_names.append(fea_name)

    def remove_fea(self, fea_name:str):
        if fea_name in self.sta_fea_names:
            self.sta_fea_names.remove(fea_name)
        elif fea_name in self.dyn_fea_names.keys():
            self.dyn_fea_names.pop(fea_name)
        else:
            logger.warning(f"FeatureManager: feature not founded: {fea_name}")
    
    # 获取所有特征的名称, expand_dyn表示将动态特征的时间展开
    def get_names(self, sta=False, dyn=False, expand_dyn=False):
        result = []
        if sta:
            result += self.sta_fea_names
        if dyn:
            if expand_dyn:
                for vlist in self.dyn_fea_names.values():
                    result += [val[1] for val in vlist]
            else:
                result += list(self.dyn_fea_names.keys())
        return result

    # 获取一个动态特征对应的所有名字和对应的时间
    def get_expanded_fea(self, dyn_name:str) -> list:
        assert(dyn_name in self.dyn_fea_names.keys()), f"dyn_name {dyn_name} should in dyn_fea_names"
        return self.dyn_fea_names[dyn_name].copy()

    # 获取一个动态特征最靠近某个时间点的名字
    def get_nearest_fea(self, dyn_name:str, time:float) -> str:
        assert(dyn_name in self.dyn_fea_names.keys()), f"dyn_name {dyn_name} should in dyn_fea_names"
        delta = abs(self.dyn_fea_names[dyn_name][0][0]-time)+1
        best_fea = None
        for item in self.dyn_fea_names[dyn_name]:
            if abs(item[0]-time) < delta:
                best_fea = item[1]
                delta = abs(item[0]-time)
        return best_fea

'''
清除并且重建一个文件夹和其中所有的内容
'''
def reinit_dir(write_dir_path=None, build=True):
    if write_dir_path is not None:
        if os.path.exists(write_dir_path):
            for name in os.listdir(write_dir_path):
                p = os.path.join(write_dir_path, name)
                if os.path.isdir(p):
                    reinit_dir(p, build=False)
                    os.rmdir(p)
                elif os.path.isfile(p):
                    os.remove(p)
        if build:
            os.makedirs(write_dir_path, exist_ok=True)

'''
设置matplotlib显示中文, 对于pandas_profile不可用
'''
def set_chinese_font():
    logger.info("Set Chinese Font in Matplotlib")
    from matplotlib import pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

#  清空文件
def clear_file(name):
    with open(name, 'w+'):
        pass

def set_sk_random_seed(seed:int=100):
    sk_random.seed(seed)

def remove_slash(name:str):
    return name.replace('/','%')

'''
输入文件名, 返回文件的MD5字符串
'''
def cal_file_md5(filename:str) -> str:
    with open(filename, 'rb') as fp:
        data = fp.read()
    file_md5= hashlib.md5(data).hexdigest()
    return file_md5


# 获取超过thres的非na比例的数据
def select_na(data:pd.DataFrame, col_thres=0.5, row_thres=0.7, fea_manager:FeatureManager=None):
    if fea_manager is None:
        # del column na
        valid_cols = msno.nullity_filter(data, filter='top', p=col_thres).columns
        data = data[valid_cols]
        # del row na
        na_mat = data.isna().to_numpy(dtype=np.int32)
        valid_mat = 1 - np.mean(na_mat, axis=1)
        data = data.iloc[valid_mat > row_thres]
    else:
        dyn_check_names = [] # 动态特征只check第一天
        static_check_names = fea_manager.get_names(sta=True)
        dyn_feas = fea_manager.get_names(dyn=True)
        for fea_name in dyn_feas:
            name = fea_manager.get_expanded_fea(fea_name)[0][1]
            dyn_check_names.append(name)
        data_check = data.loc[:, dyn_check_names + static_check_names]
        valid_cols = set(msno.nullity_filter(data_check, filter='top', p=col_thres).columns)
        col_expanded = set() # 需要把之后的特征加在第一天特征的后面(如果第一天有效)
        for idx, name in enumerate(dyn_check_names):
            if name in valid_cols:
                col_expanded.update([vol[1] for vol in fea_manager.get_expanded_fea(dyn_feas[idx])])
            else:
                fea_manager.remove_fea(dyn_feas[idx]) # remove dynamic features
        for col in valid_cols:
            if col not in col_expanded:
                col_expanded.add(col)
        dropped_cols = 0
        static_check_names = set(static_check_names)
        for col in data.columns:
            if col not in col_expanded:
                if col in static_check_names:
                    fea_manager.remove_fea(col) # remove static featur
                logger.warning(f"select na: feature {col} dropped")
                dropped_cols += 1
        logger.info(f"select na: drop {dropped_cols} features")
        data = data[list(col_expanded)]
        na_mat = data_check.isna().to_numpy(dtype=np.int32)
        valid_mat = 1 - np.mean(na_mat, axis=1)
        logger.info(f"select na: drop {len(data) - (valid_mat > row_thres).sum()}/{len(data)} na rows")
        data = data.iloc[valid_mat > row_thres]
    return data

'''
移除不符合类型要求的行
条件1: 存在某个特征不符合type_dict的类型
条件2: 存在某个特征不符合interval_dict的正常范围
条件3: target_fea的最大连续长度小于2
'''
def remove_invalid_rows(data:pd.DataFrame, type_dict:dict, interval_dict:dict=None, expanded_target:list=None) -> pd.DataFrame:
    data.reset_index(drop=True, inplace=True)
    na_table = data.isna()
    select_table = pd.Series([True for _ in range(len(data.index))], index=data.index, name='bools')
    for col in data.columns:
        flag_numeric = True if (interval_dict is not None and type_dict[col] != str and col in interval_dict.keys()) else False
        if flag_numeric:
            interval_min = interval_dict[col][0]
            interval_max = interval_dict[col][1]
        for idx in data.index:
            if not na_table.at[idx, col]:
                try:
                    tmp = type_dict[col](data.at[idx, col])
                    if flag_numeric: # 检查是否超出正常范围
                        assert(tmp >= interval_min and tmp <= interval_max)
                except Exception as e:
                    select_table[idx] = False
    ori_len = len(data)
    if expanded_target is not None:
        duration = cal_available_time(data, expanded_target=expanded_target)[:,1]
        data = data.iloc[select_table.values * (duration > 24)]
    else:
        data = data.iloc[select_table.values]
    logger.info(f"remove_invalid_rows: {(ori_len - len(data))}/{ori_len} rows are removed.")
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

# Detailed feature type is necessary to detect illegal data, which could be a float in a bool type feature.
# type_dict[col] = type
def check_fea_types(data:pd.DataFrame, max_err_rows=10) -> dict:
    logger.debug('Checking feature types')
    eps = 1e-5
    type_dict = {}
    nas = data.isna()
    priority = [bool, int, float, str] # left-right, high-low

    for col in data.columns: # prevent bad data of str types. Manual check is needed if other type occurs
        type_dict[col] = None
        
        table = {float:0, int:0, bool:0, str:0}
        # random sample 100 non-na values:
        valid_rows = np.asarray(1 - nas[col], dtype=bool)
        col_valid = data.iloc[valid_rows][col].values
        n_sample = min(200, len(col_valid))
        for val in sk_random.sample(list(col_valid), k=n_sample):
            s_val = str(val)
            num_flag = True
            try:
                float(s_val)
            except ValueError:
                num_flag = False
            if num_flag: # str will be covered by numeric
                f_val = float(s_val)
                if abs(f_val - round(f_val)) > eps: # bool, int will be covered by float
                    table[float] += 1
                    # type_dict[col] = (float, priority[float])
                elif int(f_val) not in [0,1]: # bool will be covered by int
                    table[int] += 1
                else:
                    table[bool] += 1
            else:
                table[str] += 1
        
        for key in reversed(priority):
            if table[key] > max_err_rows:
                type_dict[col] = key
                break
        assert(type_dict[col] is not None)
    return type_dict

# 只能在NA填充之后才使用
def apply_fea_types(data:pd.DataFrame, type_dict:dict) -> pd.DataFrame:
    type_dict = type_dict.copy()
    for key in type_dict.keys():
        if type_dict[key] == str:
            type_dict[key] = 'str'
        elif type_dict[key] == float:
            type_dict[key] = 'float32'
        if type_dict[key] == int:
            type_dict[key] = 'int32'
        if type_dict[key] == bool:
            type_dict[key] = 'bool'
    return data.astype(type_dict)

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
    logger.debug('4 cls label:')
    logger.debug('+ards+death:', (Y_4cls == 1).mean())
    logger.debug('+ards-death:', (Y_4cls == 2).mean())
    logger.debug('-ards+death:', (Y_4cls == 3).mean())
    logger.debug('-ards-death:', (Y_4cls == 0).mean())
    return Y_4cls


def one_hot_decoding(data:pd.DataFrame, cluster_dict:dict, fea_manager:FeatureManager=None):
    logger.debug('one hot decoding')
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
        if fea_manager is not None:
            for name in old_names:
                fea_manager.remove_fea(name)
            fea_manager.add_sta(new_name)
    return data

def fill_default(data:pd.DataFrame, sta_dict:dict, dyn_dict:dict=None, fea_manager:FeatureManager=None):
    logger.debug('fill_default')
    na_dict = {}
    for key in sta_dict.keys():
        if key not in data.columns:
            logger.warning(f"fill_default: feature not founded: {key}")
            continue
        if not isinstance(sta_dict[key], list):
            na_dict[key] = sta_dict[key]
        else:
            data.replace(sta_dict[key][0], {key:sta_dict[key][1]}, inplace=True)
    if dyn_dict is not None and fea_manager is not None:
        dyn_names = set(fea_manager.get_names(dyn=True))
        for key in dyn_dict.keys():
            if key in data.columns: # 静态
                if not isinstance(dyn_dict[key], list):
                    na_dict[key] = dyn_dict[key]
                else:
                    data.replace(dyn_dict[key][0], {key:dyn_dict[key][1]}, inplace=True)
            elif key in dyn_names:
                for name in [val[1] for val in fea_manager.get_expanded_fea(key)]:
                    if not isinstance(dyn_dict[key], list):
                        na_dict[name] = dyn_dict[key]
                    else:
                        data.replace(dyn_dict[key][0], {name:dyn_dict[key][1]}, inplace=True)
    data.fillna(value=na_dict, inplace=True)

def normalize(x:np.ndarray, axis=1):
    x_std = x.std(axis=1,keepdims=True)
    # x_std = np.clip(x_std, a_min=1e-3, a_max=1000)
    x = (x - x.mean(axis=axis, keepdims=True)) / x_std
    return x

'''
    进行训练时和测试时的正则化
    norm_dict: 每个key表示一个column名字, value=(coeff, bias) 给出正则化的参数
    如果没有norm_dict, 则代表训练模式, 会返回计算好的normdict
    如果有norm_dict, 则代表测试模式, 按照给定的参数进行正则化
'''
def feature_normalization(data:np.ndarray, num_feas:list, norm_dict=None):
    if norm_dict is not None:
        for idx in num_feas:
            if np.abs(norm_dict[idx][1]) > 1e-4: 
                data[:,idx] = (data[:,idx] - norm_dict[idx][0]) / norm_dict[idx][1]
            else:
                data[:,idx] = (data[:,idx] - norm_dict[idx][0])
        return data
    else:
        norm_dict = {}
        for idx in num_feas:
            arr = data[:,idx]
            mean = np.nanmean(arr)
            std = np.nanstd(arr)
            norm_dict[idx] = [mean, std]
            if std < 1e-4:
                logger.warning(f'Std near zero in normalization: {idx}, std ignored')
                arr = arr - mean
            else:
                arr = (arr - mean) / std
            data[:, idx] = arr
        return data, norm_dict


def target_statistic(X:np.ndarray, Y:np.ndarray, ctg_feas:list, mode='greedy', hist_val=None, mask=None):
    # mask以外的列不会参与TS
    # TS要求Y的mask内的列都是有效的, X可以是有效也可以是无效(-1), 不影响TS结果
    if mask is None:
        mask = np.ones((X.shape[0]), dtype=bool)
    if len(ctg_feas) == 0:
        logger.warning('Target statistic: len(ctg_feas)=0')
        return X
    X_ts = X.copy()
    assert(mode=='greedy')
    return_flag = 0
    if hist_val is None:
        return_flag = 1
        alpha = round(0.01*X.shape[0])
        prior = Y.mean()
        hist_val = {}
        for r_idx in range(X.shape[0]):
            if not mask[r_idx]:
                continue
            for c_idx in ctg_feas:
                if c_idx not in hist_val.keys():
                    hist_val[c_idx] = {}
                if X[r_idx, c_idx] not in hist_val[c_idx].keys():
                    hist_val[c_idx][X[r_idx, c_idx]] = (Y[r_idx],1)
                else:
                    val,count = hist_val[c_idx][X[r_idx, c_idx]]
                    hist_val[c_idx][X[r_idx, c_idx]] = (val+Y[r_idx], count+1)
        for ctg_fea_idx in hist_val:
            for cls in hist_val[ctg_fea_idx]:
                val, count = hist_val[ctg_fea_idx][cls] 
                hist_val[ctg_fea_idx][cls] = (val + alpha * prior) / (count + alpha)
    
    # produce average val for unknown category
    avg_val = {key:np.asarray(list(hist_val[key].values())).mean() for key in hist_val.keys()}
    for r_idx in range(X.shape[0]):
        for c_idx in ctg_feas:
            val = hist_val[c_idx].get(X[r_idx, c_idx])
            if val is not None:
                X_ts[r_idx, c_idx] = val
            else:
                logger.warning(f'Unknown category {X[r_idx, c_idx]} in target statistic')
                X_ts[r_idx, c_idx] = avg_val[c_idx]

    X_ts = X_ts.astype(dtype=np.float32)
    if return_flag == 1:
        return X_ts, hist_val
    else:
        return X_ts


# use feature average value to fill na values, used for better linear model performance
def fill_avg(X:np.ndarray, num_feas:list, na_sign=-1, eps=1e-3, avg_dict=None):
    X_avg = X.copy()
    if avg_dict is None:
        avg_dict = {}
        for c_idx in num_feas:
            choosed = (X_avg[:,c_idx] < na_sign + eps) * (X_avg[:,c_idx] > na_sign - eps)
            avg_dict[c_idx] = X_avg[np.invert(choosed), c_idx].mean()
            X_avg[choosed, c_idx] = avg_dict[c_idx]
        return X_avg, avg_dict
    else:
        for c_idx in num_feas:
            choosed = (X_avg[:,c_idx] < na_sign + eps) * (X_avg[:,c_idx] > na_sign - eps)
            X_avg[choosed, c_idx] = avg_dict[c_idx]
        return X_avg

# 验证name是否可以表现为prefix+name(in name_set)的形式
# time_prefix: ['day_','[day]','_T_', '[period]']
# name_set: {'feature1', 'feature2'} 不需要带前缀
# 返回值: (name, hours) 其中name在name_set中, hours是前缀解析得到的
def match_dyn_feature(name:str, time_prefix:list, name_set:set, signs={"[day]", "[period]"}):
    m_str = ""
    hours = 0
    for p_str in time_prefix:
        assert(isinstance(p_str, str))
        if p_str in signs:
            m_str += "(\d)"
        else:
            m_str += p_str
    result = re.match(pattern=m_str, string=name)
    if result is None:
        return None

    residual = name.replace(result.group(0), "")
    if residual not in name_set:
        return None

    idx = 1 # group(0) is the entire string
    for p_str in time_prefix:
        if p_str == "[day]":
            hours += 24 * (int(result.group(idx))-1)
            idx += 1
        elif p_str == "[period]":
            hours += 6 * (int(result.group(idx))-1)
            idx += 1
    return (residual, hours)

'''
计算可用的时间分布, 第一段连续的非NA值
input: 
    expanded_target: list((time, old_name)) 升序排列
return:
    array[:,0]: 起始时间, array[:,1]: 持续时间
'''
def cal_available_time(data:pd.DataFrame, expanded_target:list):
    assert(isinstance(expanded_target[0], tuple))
    # expanded_target = self.fea_manager.get_expanded_fea(dyn_fea)
    offset = expanded_target[1][0]
    names = [val[1] for val in expanded_target]
    valid_mat = (1 - data[names].isna()).astype(bool)
    result = np.empty((len(data),2)) # start_time, duration
    for r_idx in range(len(data)):
        end_time = -1
        start_time = 0
        for time, name in expanded_target:
            # TODO: this '==' can not written as 'is'for unknown reason
            if valid_mat.at[r_idx, name] == True:
                end_time = time + offset
            else:
                if end_time > 0:
                    break
                start_time = time + offset
        duration = end_time - start_time if end_time - start_time > 0 else 0
        start_time = start_time if duration > 0 else 0
        result[r_idx, 0] = start_time
        result[r_idx, 1] = duration
    return result

def assert_no_na(dataset:pd.DataFrame):
    try:
        assert(not np.any(dataset.isna().to_numpy()))
    except Exception as e:
        na_mat = dataset.isna()
        for col in dataset.columns:
            if np.any(na_mat[col].to_numpy()):
                logger.error(f'assert_na: NA in feature:{col}')
                assert(0)


class Config:
    '''
        加载配置表
        cache_path: 自动配置
        manual_path: 手动配置
    '''
    def __init__(self, cache_path, manual_path) -> None:
        self.cache_path = cache_path
        self.manual_path = manual_path
        self.configs = {}
        with open(manual_path, 'r', encoding='utf-8') as fp:
            manual_conf = json.load(fp)
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as fp:
                self.configs = json.load(fp)
        for key in manual_conf.keys(): # 覆盖
            self.configs[key] = manual_conf[key]

    def __getitem__(self, idx):
        return self.configs.copy()[idx]
    
    def dump(self):
        with open(self.manual_path, 'w', encoding='utf-8') as fp:
            json.dump(self.configs, fp)

class TimeConverter:
    '''
    将一段时间字符串转化为时间戳
    '''
    def __init__(self, format:str=None, out_unit=['day','hour','minute']) -> None:
        '''
        format: 年%Y 月%m 日%d 小时%H 分钟%M 秒%S"
        '''
        self.format = format
        coeff = 1
        if out_unit == 'day':
            coeff *= 60*60*24
        elif out_unit == 'hour':
            coeff *= 60*60
        elif out_unit == 'minute':
            coeff *= 60
        self.coeff = coeff
    
    def __call__(self, in_str:str) -> float:
        dt = datetime.datetime.strptime(in_str, self.format)
        return dt.timestamp() / self.coeff

def make_mask(m_shape, seq_lens) -> np.ndarray:
    '''
        m_shape: (batch, seq_lens) 或者 (batch, n_fea, seq_lens)
        mask: (batch, seq_lens) or (batch, n_fea, seq_lens) 取决于m_shape
    '''
    mask = np.zeros(m_shape, dtype=bool)
    if len(m_shape) == 2:
        for idx in range(m_shape[0]):
            mask[idx, :seq_lens[idx]] = True
    elif len(m_shape) == 3:
        for idx in range(m_shape[0]):
            mask[idx, :, :seq_lens[idx]] = True
    else:
        assert(0)
    return mask

def label_smoothing(centers:list, nums:np.ndarray, band=50):
    '''
    标签平滑
    centers: 每个class的中心点, 需要是递增的, n_cls = len(centers)
    nums: 输入(in_shape,) 可以是任意的
    band: 在两个class之间进行线性平滑, band是需要平滑的总宽度
        当输入在band外时(靠近各个中心或者超过两侧), 是硬标签, 只有在band内才是软标签
    return: (..., len(centers)) 其中 (...) = nums.shape
    '''
    num_classes = len(centers)
    smoothed_labels = np.zeros((nums.shape + (num_classes,)))
    for i in range(num_classes-1):
        center_i = centers[i]
        center_j = centers[i+1]
        lower = 0.5*(center_i + center_j) - band/2
        upper = 0.5*(center_i + center_j) + band/2
        mask = np.logical_and(nums > lower, nums <= upper)
        hard_i = np.logical_and(nums >= center_i, nums <= lower)
        hard_j = np.logical_and(nums < center_j, nums > upper)
        if mask.any() and band > 0:
            diff = (nums - center_i) / (center_j - center_i)
            smooth_i = 1 - diff
            smooth_j = diff
            smoothed_labels[..., i][mask] = smooth_i[mask]
            smoothed_labels[..., i+1][mask] = smooth_j[mask]
        smoothed_labels[..., i][hard_i] = 1
        smoothed_labels[..., i+1][hard_j] = 1
    smoothed_labels[..., 0][nums <= centers[0]] = 1
    smoothed_labels[..., -1][nums > centers[-1]] = 1
    return smoothed_labels


def find_latest(path_dir):
    '''寻找不含子文件的文件夹中最新文件的full path'''
    # get a list of all files in the directory
    all_files = os.listdir(path_dir)
    # get the most recently modified file
    latest_file = max(all_files, key=os.path.getmtime)
    # get the full path of the latest file
    latest_file_path = os.path.join(path_dir, latest_file)
    return latest_file_path


set_chinese_font()