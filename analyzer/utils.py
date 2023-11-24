from tools import logger as logger
import pickle
import numpy as np
import tools
import os


def map_func(a:np.ndarray):
    '''
    Map the results of 4 classifications to the results of 2 classifications
    Default is [0,1,2,3] corresponding to [Severe,Moderate,Mild,No]
    Mapping is ARDS=[0,1,2], No ARDS=[3]
    a: (... , n_cls) can be soft-labeled
    return (... , 2) where [... ,0] represents no ARDS, [... ,1] means ARDS, can be soft labeling.
    '''
    a_shape = list(a.shape)
    a_shape[-1] = 2
    result = np.zeros(tuple(a_shape))
    result[..., 0] = a[..., 3]
    result[..., 1] = a[..., 0] + a[..., 1] + a[..., 2]
    return result

def create_final_result(out_dir):
    '''Collect result.log from each folder, merge to final result.log'''
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

def cal_label_weight(n_cls, label:np.ndarray):
    '''
    Get the weight of n_cls inversely proportional to the number.
    label: (..., n_cls)
    return: (n_cls,)
    '''
    hard_label = np.argmax(label, axis=-1).flatten()
    weight = np.asarray([np.mean(hard_label == c) for c in range(n_cls)])
    logger.info(f'4cls Label proportion: {weight}')
    weight = 1 / weight
    weight = weight / np.sum(weight)
    logger.info(f'4cls weight: {weight}')
    return weight

