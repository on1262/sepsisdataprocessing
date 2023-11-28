import os
import tools
import pickle
import numpy as np
import pandas as pd
from tools import GLOBAL_CONF_LOADER
from tools import logger
from sklearn.model_selection import KFold
import math
from tqdm import tqdm

def interp(fx:np.ndarray, fy:np.ndarray, x_start:float, interval:float, n_bins:int, missing=-1, fill_bin=['avg', 'latest']):
    # fx, (N,), N >= 1, sample time for each data point (irrgular)
    # fy: same size as fx, sample value for each data point
    # x: dim=1
    assert(fx.shape[0] == fy.shape[0] and len(fx.shape) == len(fy.shape) and len(fx.shape) == 1 and fx.shape[0] >= 1)
    assert(interval > 0 and n_bins > 0)
    assert(fill_bin in ['avg', 'latest'])
    result = np.ones((n_bins)) * missing
    
    for idx in range(n_bins):
        t_bin_start = x_start + (idx - 1) * interval
        t_bin_end = x_start + idx * interval
        valid_mask = np.logical_and(fx > t_bin_start, fx <= t_bin_end) # (start, end]
        if np.any(valid_mask): # have at least one point
            if fill_bin == 'avg':
                result[idx] = np.mean(fy[valid_mask])
            elif fill_bin == 'latest':
                result[idx] = fy[valid_mask][-1]
            else:
                assert(0)
        else: # no point in current bin
            if idx == 0:
                result[idx] = missing
            else:
                result[idx] = result[idx-1] # history is always available
    return result

class Admission:
    '''
    代表一段连续的、环境较稳定的住院经历，原subject/admission/stay/transfer的四级结构被精简到subject/admission的二级结构
    label: 代表急诊室数据或ICU数据
    admittime: 起始时间
    dischtime: 结束时间
    '''
    def __init__(self, unique_id:int, admittime:float, dischtime:float) -> None:
        self.dynamic_data = {} # dict(fea_name:ndarray(value, time))
        assert(admittime < dischtime)
        self.unique_id = unique_id # 16 digits
        self.admittime = admittime
        self.dischtime = dischtime
        self._data_updated = False
    
    def append_dynamic(self, itemid, time:float, value):
        assert(not self._data_updated)
        if self.dynamic_data.get(itemid) is None:
            self.dynamic_data[itemid] = [(value, time)]
        else:
            self.dynamic_data[itemid].append((value, time))

    def pop_dynamic(self, itemid):
        if self.dynamic_data.get(itemid) is not None:
            self.dynamic_data.pop(itemid)
        
    def update_data(self):
        '''绝对时间变为相对时间，更改动态特征的格式'''
        if not self._data_updated:
            self._data_updated = True
            for key in self.dynamic_data:
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
    def __init__(self, subject_id, birth_year:int) -> None:
        self.subject_id = subject_id
        self.birth_year = birth_year
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
    
    def latest_static(self, key, time=None):
        if key not in self.static_data.keys():
            return None

        if not isinstance(self.static_data[key], np.ndarray): # single value
            return self.static_data[key]
        else:
            assert(time is not None)
            idx = np.argmin(time - self.static_data[key][:, 1])
            if time - self.static_data[key][idx, 1] >= 0:
                return self.static_data[key][idx, 0]
            else:
                return None # input time is too early
    
    def nearest_static(self, key, time=None):
        if key not in self.static_data.keys():
            return None

        if not isinstance(self.static_data[key], np.ndarray): # single value
            return self.static_data[key]
        else:
            assert(time is not None)
            idx = np.argmin(np.abs(self.static_data[key][:, 1] - time))
            return self.static_data[key][idx, 0]
    
    def append_dynamic(self, charttime:float, itemid, value):
        '''添加一个动态特征到合适的admission中'''
        for adm in self.admissions: # search admission by charttime
            if adm.admittime < charttime and charttime < adm.dischtime:
                adm.append_dynamic(itemid, charttime, value)

    def update_data(self):
        '''将数据整理成连续形式'''
        for adm in self.admissions:
            adm.update_data()

    def find_admission(self, unique_id:int):
        for adm in self.admissions:
            if adm.unique_id == unique_id:
                return adm
        return None
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

class KFoldIterator:
    def __init__(self, dataset, k):
        self._current = -1
        self._k = k
        self._dataset = dataset

    def __iter__(self):
        return self

    def __next__(self):
        self._current += 1
        if self._current < self._k:
            return self._dataset.set_kf_index(self._current)
        else:
            raise StopIteration
        
def load_all_subjects(patient_table_path:str) -> set:
    # return a dict with key=subject_id, value=None
    patient_set = set()
    patients = pd.read_csv(os.path.join(patient_table_path), encoding='utf-8')
    for row in tqdm(patients.itertuples(), 'Find all subjects', total=len(patients)):
        patient_set.add(row.subject_id)
    return patient_set
