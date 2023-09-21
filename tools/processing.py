import pandas as pd
import json
from .colorful_logging import logger
from .generic import reinit_dir
import os

# 通过给定的json进行特征离散化
def feature_discretization(config_path:str, df:pd.DataFrame):
    logger.info('feature_discretization')
    with open(config_path, 'r', encoding='utf-8') as fp:
        config = json.load(fp)
    used_fea = config[u"可用特征"]
    thres_dict = config[u"离散化阈值"]
    df = df.loc[:, used_fea]
    df = df.astype({col: 'str' for col in df.columns})
    for col in df.columns:
        if col not in thres_dict.keys():
            logger.warning('skipped feature_discretization on:', col)
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

def split_csv(path, max_len=1000000, out_folder:str=None):
    '''分割一个csv到若干个小csv, 复制表头'''
    _, name = os.path.split(path)
    reinit_dir(out_folder, build=True)
    out_prefix = name.split('.')[0]
    file_count = 1
    nfp = None
    with open(path, 'r',encoding='utf-8') as fp:
        title = fp.readline()
        tmp = fp.readline()
        count = 0
        while(True):
            if count == max_len:
                nfp.close()
                count = 0
                nfp = None
                file_count += 1
            if not tmp:
                break
            if count == 0:
                cache_name = os.path.join(out_folder, out_prefix + str(file_count) + '.csv')
                nfp = open(cache_name, 'w', encoding='utf-8')
                logger.info(f'Writing {cache_name}')
                nfp.write(title)
            nfp.write(tmp)
            count += 1
            tmp = fp.readline()
        if nfp is not None:
            nfp.close()

