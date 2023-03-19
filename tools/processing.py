import pandas as pd
import json
from .colorful_logging import logger
from .generic import reinit_dir


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
                logger.warning(f'Conflict: {key}')
            else:
                data_dict[key] = r_idx
    a_rows, b_rows = [], []
    for key, val in a_dict.items():
        if key in b_dict.keys():
            a_rows.append(val)
            b_rows.append(b_dict[key])
    logger.info(f'Detected {len(a_rows)} rows able to be combined')
    data_a, data_b = data_a.loc[a_rows,:], data_b.loc[b_rows, :]
    data_a = data_a.rename(columns={col:rename_prefix[0] + col for col in data_a.columns})
    data_b = data_b.rename(columns={col:rename_prefix[1] + col for col in data_b.columns})
    data_b.index = data_a.index
    return pd.concat([data_a, data_b], axis=1, join='inner')

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

"""
第一次数据存在一些问题, 这段代码将第二次数据的PaO2/FiO2拷贝到第一次数据的氧合指数上
拼接依赖于唯一码, 这段代码应当只用一次
"""
def fix_feature_error_in_old_sys(old_csv: str, new_csv:str, output:str, rebuild_ards=False):
    def detect_ards_label(in_str:str)->bool:
        for fea in [u'ARDS', u'急性呼吸窘迫综合征']:
            if fea in in_str:
                return True
        return False

    old_data = pd.read_csv(old_csv, encoding='utf-8')
    new_data = pd.read_csv(new_csv, encoding='utf-8')
    try:
        now_fea = None
        now_file = 'old'
        for fea in [
            u'ARDS', u'唯一号', u'姓名', u'SOFA_氧合指数', \
            u'CIS诊断', u'CIS诊断1', u'出院诊断/死亡诊断','PaO2' \
            ]:
            now_fea = fea
            assert(fea in old_data.columns)
        now_file = 'new'
        for fea in [u'唯一号', u'姓名', u'D1_PaO2/FiO2', u'D1_PaO2']:
            now_fea = fea
            assert(fea in new_data.columns)
    except Exception as e:
        logger.error(f'fix_feature_error_in_old_sys: 特征缺失: {now_fea} in {now_file}')
        return
    # 统计信息
    statistics = {'hash_target':0, 'ARDS_target':0}
    old_data.reset_index(drop=True, inplace=True)
    new_data.reset_index(drop=True, inplace=True)
    # ARDS 标签不一定重建, 按需要调整, 
    logger.info(f'Rebuild ARDS Switch={rebuild_ards}')
    if not rebuild_ards:
        for r_idx in range(len(old_data)):
            if old_data.at[r_idx, u'ARDS'] == 1:
                statistics['ARDS_target'] += 1
    else:
        for r_idx in range(len(old_data)):
            in_str = str(old_data.at[r_idx, u'出院诊断/死亡诊断']) + \
                str(old_data.at[r_idx, u'CIS诊断']) + str(old_data.at[r_idx, u'CIS诊断1'])
            in_str = str(old_data.at[r_idx, u'CIS诊断']) + str(old_data.at[r_idx, u'CIS诊断1'])
            if detect_ards_label(in_str):
                old_data.at[r_idx, u'ARDS'] = 1
                statistics['ARDS_target'] += 1
            else:
                old_data.at[r_idx, u'ARDS'] = 0
    # 构建氧合指数哈希表
    hash_dict = {}
    for r_idx in range(len(new_data)):
        hash_dict['+'.join([new_data.at[r_idx, u'唯一号'], new_data.at[r_idx, u'姓名']])] = \
            (new_data.at[r_idx, u'D1_PaO2/FiO2'], new_data.at[r_idx, u'D1_PaO2'])
    for r_idx in range(len(old_data)):
        result = hash_dict.get(
            '+'.join([old_data.at[r_idx, u'唯一号'], old_data.at[r_idx, u'姓名']])
        )
        if result is not None:
            statistics['hash_target'] += 1
            old_data.at[r_idx, u'SOFA_氧合指数'] = result[0]
            old_data.at[r_idx, u'PaO2'] = result[1]
        else:
            old_data.at[r_idx, u'SOFA_氧合指数'] = None
            old_data.at[r_idx, u'PaO2'] = None

    old_data.to_csv(output, encoding='utf-8', index=False)
    logger.info(f'new_data样本量={len(new_data)}, \
        old_data样本量={len(old_data)}, hash_table命中=%d' % statistics['hash_target'])
    logger.info('ARDS标签占比=%.3f' % (statistics['ARDS_target'] / len(old_data)))
    logger.info(f'Output to {output}')


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