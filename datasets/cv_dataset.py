import os, sys
import tools
from os.path import join as osjoin, exists
import pickle
import numpy as np
import pandas as pd
from tools import GLOBAL_CONF_LOADER
from tools import logger
from tqdm import tqdm
import time
from sklearn.model_selection import KFold

class Converter:
    def __init__(self) -> None:
        # self.date_converter = datetime.datetime()
        self.col = None

    def switch_col(self, new_col):
        self.col = new_col
    
    def __call__(self, x):
        if self.col == '性别':
            assert(x in ['男', '女'])
            return 1 if x == '男' else 0
        elif self.col == '出生年月': # 只保留年份
            assert(x != '')
            return int(x.split('T ')[-1])
        elif self.col == '入住ICU日期': # 只保留年份
            return int(x.split('-')[0])
        else:
            try:
                x = float(x)
                return x
            except Exception as e:
                return -1
    

class CrossValidationDataset():
    __name = 'cv'

    @classmethod
    def name(cls):
        return cls.__name
    
    def __init__(self) -> None:
        self.paths = GLOBAL_CONF_LOADER['paths']['cv']
        self.loc_conf = tools.Config(self.paths['conf_manual_path'])
        self.data = None # (samples, fea, ticks)
        self.total_keys = None
        self.preprocess()
        self.seqs_len = np.ones(self.data.shape[0], dtype=np.int64) * (7*24*2)

    def preprocess(self, load_pkl=False):
        pkl_path = osjoin(self.paths['cache_dir'], 'data.pkl')
        if exists(pkl_path) and load_pkl:
            with open(pkl_path, 'rb') as fp:
                result = pickle.load(fp)
                self.data = result['data']
                self.total_keys = result['total_keys']
            return
        # preprocess
        if not exists(self.paths['cache_dir']):
            tools.reinit_dir(self.paths['cache_dir'])
        data_path = self.paths['data_dir']
        data = pd.read_csv(data_path)
        dt_head = [f'd{d}_t{t}_' for t in range(1,5) for d in range(1,8)]
        t_head = [f'd{d}_' for d in range(1,8)]
        # step1: 展开需要提取的列
        extract_cols = self.loc_conf['extract_cols']
        col_maps = {} # 特征和其对应名字的从属关系
        expanded_cols = []
        for col in extract_cols:
            col = str(col)
            if col.startswith('#'):
                col = col.replace('#', '')
                sublist = []
                for day in range(1, 8):
                    sublist.append(col.replace('dX', f'd{day}'))
                if 'tX' in col:
                    subsublist = []
                    for s in sublist:
                        for time in range(1, 5):
                            subsublist.append(s.replace('tX', f't{time}'))
                    expanded_cols += subsublist
                    col_maps[col] = subsublist
                else:
                    expanded_cols += sublist
                    col_maps[col] = sublist
            else:
                expanded_cols.append(col)
                col_maps[col] = [col]
        for col in expanded_cols:
            assert(col in data.columns)
        data = data[expanded_cols]
        # step2: 均值填充
        converter = Converter()
        for col in data.columns:
            converter.switch_col(col)
            data[col] = data[col].apply(converter)
            col_data = data[col].to_numpy()
            fill_flag = (col_data < -0.99) * (col_data > -1.01)
            col_mean = None
            if np.any(fill_flag):
                col_mean = np.sum(col_data * (1-fill_flag)) / np.sum(1-fill_flag)
                data.loc[fill_flag, col] = col_mean
        assert(np.sum(np.abs(data.to_numpy(np.float32) + 1) < 0.01) == 0) # 确保所有均值都被填充了
        # step3: 静态特征加工
        deprecated_cols = [
            '入住ICU日期', '出生年月', 'dX_tX_最高心率（次/min）', 'dX_tX_最低心率（次/min）',
            'dX_tX_最高SPO2（%）', 'dX_tX_最低SPO2（%）', 'dX_tX_最高体温（℃）',
            'dX_tX_最高呼吸频率（次/min）', 'dX_tX_最低呼吸频率（次/min）'
        ]
        data['年龄'] = data['入住ICU日期'] - data['出生年月']
        col_maps['年龄'] = ['年龄']
        for idx, head in enumerate(dt_head):
            if idx == 0:
                col_maps['dX_tX_心率'] = []
                col_maps['dX_tX_SPO2'] = []
                col_maps['dX_tX_体温'] = []
                col_maps['dX_tX_呼吸频率'] = [] # 次/min
            
            data[head + '心率'] = 0.5 * data[head + '最高心率（次/min）'] + 0.5 * data[head + '最低心率（次/min）']
            col_maps['dX_tX_心率'].append(head + '心率')
            data[head + 'SPO2'] = 0.5 * data[head + '最高SPO2（%）'] + 0.5 * data[head + '最低SPO2（%）']
            col_maps['dX_tX_SPO2'].append(head + 'SPO2')
            data[head + '体温'] = data[head + '最高体温（℃）']
            col_maps['dX_tX_体温'].append(head + '体温')
            data[head + '呼吸频率（次/min）'] = 0.5 * data[head + '最高呼吸频率（次/min）'] + 0.5 * data[head + '最低呼吸频率（次/min）']
            col_maps['dX_tX_呼吸频率'].append(head + '呼吸频率（次/min）')
        
        for idx, head in enumerate(t_head):
            if idx == 0:
                col_maps['dX_FiO2(%)'] = []
            data[head + 'FiO2(%)'] = data[head + 'PaO2（mmHg）'] / data[head + 'PaO2（mmHg） / FiO2（%）']
            col_maps['dX_FiO2(%)'].append(head + 'FiO2(%)')
        for feas in deprecated_cols:
            col_maps.pop(feas)
        data = data.copy()
        total_keys = list(col_maps.keys())
        total_keys = sorted(total_keys, key=lambda x: len(x) if x != 'dX_PaO2（mmHg） / FiO2（%）' else 999) # move target to the end
        # step4: 沿时间轴展开
        table = -np.ones((len(data.index), len(col_maps), 28))
        for idx, col in enumerate(total_keys):
            if len(col_maps[col]) > 1:
                if len(col_maps[col]) == 7: # 只有天数，没有间隔
                    for sub_idx, subcol in enumerate(col_maps[col]):
                        np_arr = data[subcol].to_numpy()[..., None]
                        table[:, idx, 4*sub_idx:4*(sub_idx+1)] = np_arr
                elif len(col_maps[col]) == 28:
                    for sub_idx, subcol in enumerate(col_maps[col]):
                        np_arr = data[subcol].to_numpy()
                        table[:, idx, sub_idx] = np_arr
                else:
                    assert(0)
            else:
                assert(col in data.columns)
                np_arr = data[col].to_numpy()[..., None] # (subjects, 1)
                table[:, idx, :] =  np_arr# 没有时间信息，直接展开
        # step5: 插值得到最小粒度为半小时的表
        assert(np.sum((table < -0.99) * (table > -1.01)) == 0)
        final_table = np.zeros((table.shape[0], table.shape[1], 7*24*2))
        for t in range(28):
            final_table[:, :, 12*t:12*(t+1)] = table[:,:,[t]]
        # step7: 清除异常值
        for idx, col in enumerate(total_keys):
            if col == '体重（kg）':
                final_table[:, idx, :] = np.clip(final_table[:, idx, :], 40, 200)
            elif col == '身高（cm）':
                final_table[:, idx, :] = np.clip(final_table[:, idx, :], 150, 200)
            elif col == 'dX_tX_心率':
                final_table[:, idx, :] = np.clip(final_table[:, idx, :], 50, 200)
            elif col == 'dX_tX_体温':
                final_table[:, idx, :] = np.clip(final_table[:, idx, :], 30, 45)
            elif col == 'dX_tX_SPO2': # 0-100
                final_table[:, idx, :] = np.clip(final_table[:, idx, :], 60, 100)
            elif col == 'dX_FiO2(%)': # 0-1
                final_table[:, idx, :] = np.clip(final_table[:, idx, :], 0.21, 1)
            elif col == 'dX_PaO2（mmHg）':
                final_table[:, idx, :] = np.clip(final_table[:, idx, :], 100, 500)
            elif col == 'dX_PaO2（mmHg） / FiO2（%）':
                final_table[:, idx, :] = np.clip(final_table[:, idx, :], 100, 500)
        # step6: 打印结果
        print('total keys: ', total_keys)
        print('table size: ', final_table.shape)
        for idx, col in enumerate(total_keys):
            print(f'col{idx}: ', col, 'avg: ', final_table[:, idx, :].mean(), 
                  'min: ', final_table[:, idx, :].min(), 'max: ', final_table[:, idx, :].max())
        # last step: 加载并保存信息
        self.data = final_table
        self.total_keys = total_keys
        with open(pkl_path, 'wb') as fp:
            pickle.dump({
                'data': final_table,
                'total_keys': total_keys
            }, fp)

    def mode(self, mode=['train', 'valid', 'test', 'all']):
        '''切换dataset的模式, train/valid/test需要在register_split方法调用后才能使用'''
        pass

    def __getitem__(self, idx):
        return {'data': self.data[idx, :, :], 'length': self.data.shape[-1]}

    def __len__(self):
        return self.data.shape[0]