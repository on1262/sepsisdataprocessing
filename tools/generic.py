import numpy as np
import yaml
import datetime
from sklearn import random as sk_random
import pandas as pd
import pickle
import os, sys
import missingno as msno
import hashlib
from .logging import logger

def reinit_dir(write_dir_path=None, build=True):
    '''清除并且重建一个文件夹和其中所有的内容'''
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
    plt.rcParams['font.family'] = ['Arial Unicode MS']

def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pkl(path):
    with open(path, 'rb') as f:
        result = pickle.load(f)
    return result

#  清空文件
def clear_file(name):
    with open(name, 'w+'):
        pass

def set_sk_random_seed(seed:int=100):
    sk_random.seed(seed)

def remove_slash(name:str):
    return name.replace('/','%')


def cal_file_md5(filename:str) -> str:
    '''输入文件名, 返回文件的MD5字符串'''
    with open(filename, 'rb') as fp:
        data = fp.read()
    file_md5= hashlib.md5(data).hexdigest()
    return file_md5


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
    def __init__(self, conf_path) -> None:
        self.conf_path = conf_path
        with open(conf_path, 'r', encoding='utf-8') as fp:
            self.configs = yaml.load(fp, Loader=yaml.SafeLoader)

    def __getitem__(self, idx):
        return self.configs.copy()[idx]
    
    def dump(self):
        with open(self.conf_path, 'w', encoding='utf-8') as fp:
            yaml.dump(self.configs, fp)

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


def find_best(path_dir, prefix='best'):
    '''寻找不含子文件的文件夹中最新文件的full path'''
    # get a list of all files in the directory
    all_files = os.listdir(path_dir)
    for file in all_files:
        if str(file).startswith(prefix):
            # get the full path of the latest file
            best_file_path = os.path.join(path_dir, file)
            return best_file_path
    assert(0)
    return None

# set_chinese_font()