from tools import logger as logger
import os


def detect_adm_data(id:str, subjects:dict):
    '''直接打印某个id的输出'''
    for s_id, s in subjects.items():
        for adm in s.admissions:
            logger.info(adm[id][:,0])
            input()


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