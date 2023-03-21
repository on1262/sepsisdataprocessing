from tools import logger as logger
import pickle
import numpy as np
import tools
import os

def generate_labels(dataset, data, target_idx, generator, out_dir):
    '''生成标签的通用代码'''
    dataset.mode('all')
    pkl_path = os.path.join(out_dir, 'dataset_derived.pkl')
    if os.path.exists(pkl_path):
        logger.info(f'Load derived data set from {pkl_path}')
        with open(pkl_path, 'rb') as fp:
            mask, label = pickle.load(fp)
    else:
        logger.info('Generating label')
        mask = tools.make_mask((data.shape[0], data.shape[2]), dataset.seqs_len) # -> (batch, seq_lens)
        mask[:, -1] = False # 最后一格无法预测
        label = generator(data[:, target_idx, :], mask)
    return mask, label

def detect_adm_data(id:str, subjects:dict):
    '''直接打印某个id的输出'''
    for s_id, s in subjects.items():
        for adm in s.admissions:
            logger.info(adm[id][:,0])
            input()

def map_func(a:np.ndarray):
    '''
    将4分类的结果map到2分类的结果
    默认是[0,1,2,3]对应[重度,中度,轻度,无]
    映射是ARDS=[0,1,2], No ARDS=[3]
    a: (..., n_cls) 可以是软标签
    return (..., 2) 其中[...,0]代表无ARDS, [...,1]代表有ARDS, 可以是软标签
    '''
    a_shape = a.shape
    a_shape[-1] = 2
    result = np.zeros(a_shape)
    result[..., 0] = a[..., 3]
    result[..., 1] = a[..., 0] + a[..., 1] + a[..., 2]
    return result

def create_final_result(out_dir):
    '''收集各个文件夹里面的result.log, 合并为final result.log'''
    logger.info('Creating final result')
    with open(os.path.join(out_dir, 'final_result.log'), 'w') as final_f:
        for dir in os.listdir(out_dir):
            p = os.path.join(out_dir, dir)
            if os.path.isdir(p):
                if 'result.log' in os.listdir(p):
                    rp = os.path.join(p, 'result.log')
                    logger.info(f'Find: {rp}')
                    with open(rp, 'r') as f:
                        final_f.write(f.read())
                        final_f.write('\n')
    logger.info(f'Final result saved at ' + os.path.join(out_dir, 'final_result.log'))