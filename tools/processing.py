import pandas as pd
from .colorful_logging import logger
from .generic import reinit_dir
import os

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

