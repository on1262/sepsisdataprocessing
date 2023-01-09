import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy
from sklearn.metrics import auc as sk_auc
import pandas as pd
from collections.abc import Iterable
import os, sys
import subprocess
import json
import missingno as msno

'''
合并两个文件中cmp_cols都相同的样本, 同时列标签加上new和old
'''
def combine_and_select_samples(data_a: pd.DataFrame, data_b: pd.DataFrame, rename_prefix:list):
    cmp_cols = [u'唯一号', u'住院号', u'姓名', u'年龄']
    for col in cmp_cols:
        assert(col in data_a.columns and col in data_b.columns)
    # make hash dict
    a_dict = {}
    b_dict = {}
    for name in ['a', 'b']:
        data_dict = a_dict if name == 'a' else b_dict
        data_pd = data_a if name == 'a' else data_b
        for r_idx, row in data_pd.iterrows():
            key = '+'.join([str(row[col]) for col in cmp_cols])
            if key in data_dict:
                print(f'Conflict: {key}')
            else:
                data_dict[key] = r_idx
    a_rows, b_rows = [], []
    for key, val in a_dict.items():
        if key in b_dict.keys():
            a_rows.append(val)
            b_rows.append(b_dict[key])
    print(f'Detected {len(a_rows)} rows able to be combined')
    data_a, data_b = data_a.loc[a_rows,:], data_b.loc[b_rows, :]
    data_a = data_a.rename(columns={col:rename_prefix[0] + col for col in data_a.columns})
    data_b = data_b.rename(columns={col:rename_prefix[1] + col for col in data_b.columns})
    data_b.index = data_a.index
    return pd.concat([data_a, data_b], axis=1, join='inner')

# 通过给定的json进行特征离散化, json格式为:
def feature_discretization(config_path:str, df:pd.DataFrame):
    print('feature_discretization')
    with open(config_path, 'r', encoding='utf-8') as fp:
        config = json.load(fp)
    used_fea = config[u"可用特征"]
    thres_dict = config[u"离散化阈值"]
    df = df.loc[:, used_fea]
    df = df.astype({col: 'str' for col in df.columns})
    for col in df.columns:
        if col not in thres_dict.keys():
            print('skipped feature_discretization on:', col)
            continue
        for ridx in range(len(df)):
            cond_flag = False
            for cond in thres_dict[col]: # dict, example: {"大于等于":200, "小于":300,"名称":"轻度ARDS"}
                val = cond[u"名称"]
                if not pd.isna(df.at[ridx,col]):
                    flag = True
                    df_val = float(df.at[ridx,col])
                    for cond_key in cond.keys():
                        if u"大于等于" == cond_key:
                            flag = False if df_val < cond[cond_key] else flag
                        elif u"大于" in cond_key:
                            flag = False if df_val <= cond[cond_key] else flag
                        elif u"小于等于" == cond_key:
                            flag = False if df_val > cond[cond_key] else flag
                        elif u"小于" in cond_key:
                            flag = False if df_val >= cond[cond_key] else flag
                        elif u"等于" == cond_key:
                            flag = False if df_val != cond[cond_key] else flag
                    if flag:
                        df.at[ridx,col] = val
                        cond_flag = True
                        break
            if cond_flag == False:
                df.at[ridx, col] = u"NAN" # 包括正常指标和缺失值, 正常值在apriori中不予考虑
    # 预处理
    df = df.reset_index(drop=True)
    for col in df.columns:
        for ridx in range(len(df)):
            df.at[ridx, col] = col + "=" + str(df.at[ridx, col])
    return df

"""
第一次数据存在一些问题, 这段代码将第二次数据的PaO2/FiO2拷贝到第一次数据的氧合指数上, 并且从出院诊断中重建ARDS标签
拼接依赖于唯一码, 这段代码应当只用一次
"""
def fix_feature_error_in_old_sys(old_csv: str, combined:str, output:str):
    def detect_ards_label(in_str:str)->bool:
        for fea in [u'ARDS', u'急性呼吸窘迫综合征']:
            if fea in in_str:
                return True
        return False

    old_data = pd.read_csv(old_csv, encoding='utf-8')
    combined_data = pd.read_csv(combined, encoding='utf-8')
    try:
        for fea in [u'ARDS', u'唯一码', u'姓名', u'SOFA_氧合指数', u'SOFA_氧合指数分值', u'出院诊断/死亡诊断']:
            assert(fea in old_data.columns)
        for fea in [u'oldsys_唯一号', u'oldsys_姓名', u'newsys_D1_PaO2/FiO2']:
            assert(fea in combined_data.columns)
    except Exception as e:
        print('Error: 特征缺失')
        return
    # 统计信息
    statistics = {'hash_target':0, 'ARDS_target':0}
    # 重建ARDS标签
    old_data.reset_index(drop=True, inplace=True)
    combined_data.reset_index(drop=True, inplace=True)
    for r_idx in range(len(old_data)):
        in_str = str(old_data.at[r_idx, u'出院诊断/死亡诊断'])
        if detect_ards_label(in_str):
            old_data.at[r_idx, u'ARDS'] = 1
            statistics['ARDS_target'] += 1
        else:
            old_data.at[r_idx, u'ARDS'] = 0
    # 构建氧合指数哈希表
    hash_dict = {}
    for r_idx in range(len(combined_data)):
        hash_dict['+'.join([combined_data.at[r_idx, u'oldsys_唯一号'], combined_data.at[r_idx, u'oldsys_姓名']])] = \
            combined_data.at[r_idx, u'newsys_D1_PaO2/FiO2']
    for r_idx in range(len(old_data)):
        result = hash_dict.get(
            '+'.join([old_data.at[r_idx, u'唯一码'], old_data.at[r_idx, u'姓名']])
        )
        if result is not None:
            statistics['hash_target'] += 1
        old_data.at[r_idx, u'SOFA_氧合指数'] = result
    old_data.to_csv(output, encoding='utf-8')
    print(f'combined_data样本量={len(combined_data)}, \
        old_data样本量={len(old_data)}, hash_table命中=', statistics['hash_target'])
    print('ARDS标签占比=', statistics['ARDS_target'] / len(old_data))
    print(f'Output to {output}')


