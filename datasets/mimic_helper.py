import os
import tools
import pickle
import numpy as np
import pandas as pd
from tools import GLOBAL_CONF_LOADER
from tools import logger
from sklearn.model_selection import KFold
import math

class Admission:
    '''
    代表一段连续的、环境较稳定的住院经历，原subject/admission/stay/transfer的四级结构被精简到subject/admission的二级结构
    label: 代表急诊室数据或ICU数据
    admittime: 起始时间
    dischtime: 结束时间
    '''
    def __init__(self, unique_id:int, admittime:float, dischtime:float, label=['ed', 'icu']) -> None:
        self.dynamic_data = {} # dict(fea_name:ndarray(value, time))
        assert(admittime < dischtime)
        self.label = label # ed or icu
        self.unique_id = unique_id # hadm_id+stay_id or hadm_id+transfer_id, 16 digits
        self.admittime = admittime
        self.dischtime = dischtime
        self.data_updated = False
    
    def append_dynamic(self, itemid, time:float, value):
        assert(not self.data_updated)
        if self.dynamic_data.get(itemid) is None:
            self.dynamic_data[itemid] = [(value, time)]
        else:
            self.dynamic_data[itemid].append((value, time))

    def pop_dynamic(self, itemid):
        if self.dynamic_data.get(itemid) is not None:
            self.dynamic_data.pop(itemid)
        
    def update_data(self):
        '''绝对时间变为相对时间，更改动态特征的格式'''
        if not self.data_updated:
            self.data_updated = True
            for key in self.dynamic_data:
                if isinstance(self.dynamic_data[key], list):
                    arr = np.asarray(sorted(self.dynamic_data[key], key=lambda x:x[1]))
                    arr[:, 1] -= self.admittime
                    self.dynamic_data[key] = arr
    
    def duration(self):
        return max(0, self.dischtime - self.admittime)
    
    def empty(self):
        return True if len(self.dynamic_data) == 0 else False

    def __getitem__(self, idx):
        return self.dynamic_data[idx]
    
    def __len__(self):
        return len(self.dynamic_data)

    def keys(self):
        return self.dynamic_data.keys()


class Subject:
    '''
    每个患者有一张表, 每列是一个指标, 每行是一次检测结果, 每个结果包含一个(值, 时间戳)的结构
    static data: dict(feature name: (value, charttime))
    dyanmic data: admissions->(id, chart time, value)
    '''
    def __init__(self, subject_id, anchor_year:int) -> None:
        self.subject_id = subject_id
        self.anchor_year = anchor_year
        self.static_data:dict[str, np.ndarray] = {} # dict(fea_name:value)
        self.admissions:list[Admission] = []
    
    def append_admission(self, admission:Admission):
        self.admissions.append(admission)
        # 维护时间序列
        if len(self.admissions) >= 1:
            self.admissions = sorted(self.admissions, key=lambda adm:adm.admittime)

    def append_static(self, charttime:float, name, value):
        if charttime is None:
            self.static_data[name] = value
        else:
            if name not in self.static_data:
                self.static_data[name] = [(value, charttime)]
            else:
                self.static_data[name].append((value, charttime))
    
    def nearest_static(self, key, time=None):
        '''返回与输入时间最接近的静态特征取值'''
        if key not in self.static_data.keys():
            return -1

        if not isinstance(self.static_data[key], np.ndarray): # single value
            return self.static_data[key]
        else:
            nearest_idx, delta = 0, np.inf
            assert(time is not None)
            for idx in range(self.static_data[key].shape[0]):
                new_delta = np.abs(time-self.static_data[key][idx, 1])
                if new_delta < delta:
                    delta = new_delta
                    nearest_idx = idx
            return self.static_data[key][nearest_idx, 0]

    def append_dynamic(self, charttime:float, itemid, value):
        '''添加一个动态特征到合适的admission中'''
        for adm in self.admissions: # search admission by charttime
            if adm.admittime < charttime and charttime < adm.dischtime:
                adm.append_dynamic(itemid, charttime, value)

    def update_data(self):
        '''将数据整理成连续形式'''
        for adm in self.admissions:
            adm.update_data()

    def del_empty_admission(self):
        # 删除空的admission
        new_adm = []
        for idx in range(len(self.admissions)):
            if not self.admissions[idx].empty():
                new_adm.append(self.admissions[idx])
        self.admissions = new_adm
    
    def empty(self):
        for adm in self.admissions:
            if not adm.empty():
                return False
        return True
    
def reduce_peak(x: np.ndarray):
    '''
    清除x中的异常峰
    x: 1d ndarray
    '''
    window_size = 3  # Size of the sliding window
    threshold = 1.4  # Anomaly threshold at (thredhold-1)% higher than nearby points
    for i in range(len(x)):
        left_window_size = min(window_size, i)
        right_window_size = min(window_size, len(x) - i - 1)
        window = x[i - left_window_size: i + right_window_size + 1]
        
        avg_value = (np.sum(window) - x[i]) / (len(window)-1)
        if x[i] >= avg_value * threshold and x[i] > 110:
            x[i] = avg_value
    return x

def load_sepsis_patients(csv_path:str) -> dict:
    '''
    从csv中读取按照mimic-iv pipeline生成的sepsis3.0表格
    sepsis_dict: dict(int(subject_id):list(occur count, elements))
        element: [sepsis_time(float), stay_id, sofa_score, respiration, liver, cardiovascular, cns, renal]
        多次出现会记录多个sepsis_time
    '''
    converter = tools.TimeConverter(format="%Y-%m-%d %H:%M:%S", out_unit='hour')
    sepsis_dict = {}

    def extract_time(row): # 提取sepsis发生时间, 返回float，注意这里包含了对sepsis发生时间的定义
        return min(converter(row['antibiotic_time']), converter(row['culture_time']))
    
    def build_dict(row): # 提取sepsis dict
        id = int(row['subject_id'])
        element = {k:row[k] for k in ['sepsis_time', 'stay_id', 'sofa_score', 'respiration', 'liver', 'cardiovascular', 'cns', 'renal']}
        element['stay_id'] = int(element['stay_id'])
        if id in sepsis_dict:
            sepsis_dict[id].append(element)
        else:
            sepsis_dict[id] = [element]
        
    df = pd.read_csv(csv_path, encoding='utf-8')
    df['sepsis_time'] = df.apply(extract_time, axis=1)
    df.apply(build_dict, axis=1)
    logger.info(f'Load {len(sepsis_dict.keys())} sepsis subjects based on sepsis3.csv')
    return sepsis_dict