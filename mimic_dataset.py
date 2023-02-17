"""mimic数据集抽象"""

import os
import pickle
import json
import tools
import numpy as np
import pandas as pd
from tools import GLOBAL_CONF_LOADER
from tools import logger
from tqdm import tqdm

class Config:
    '''
        加载mimic对应的配置表
    '''
    def __init__(self, cache_path, manual_path) -> None:
        self.cache_path = cache_path
        self.manual_path = manual_path
        self.configs = {}
        with open(manual_path, 'r', encoding='utf-8') as fp:
            manual_conf = json.load(fp)
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as fp:
                self.configs = json.load(fp)
        for key in manual_conf.keys(): # 覆盖
            self.configs[key] = manual_conf[key]

    def __getitem__(self, idx):
        return self.configs.copy()[idx]
    
    def dump(self):
        with open(self.manual_path, 'w', encoding='utf-8') as fp:
            json.dump(self.configs, fp)

class Admission:
    '''
    代表Subject/Admission
    '''
    def __init__(self, hadm_id, admittime:float, dischtime:float) -> None:
        self.dynamic_data = {} # dict(fea_name:ndarray(value, time))
        assert(admittime < dischtime)
        self.hadm_id = hadm_id
        self.admittime = admittime
        self.dischtime = dischtime
    
    def append_value(self, itemid, time:float, value):
        if self.dynamic_data.get(itemid) is None:
            self.dynamic_data[itemid] = [(value, time)]
        else:
            self.dynamic_data[itemid].append((value, time))

    def update_data(self):
        for key in self.dynamic_data:
            if isinstance(self.dynamic_data[key], list):
                arr = np.asarray(self.dynamic_data[key])
                arr[:, 1] -= self.admittime
                self.dynamic_data[key] = arr



        
class Subject:
    '''
    每个患者有一张表, 每列是一个指标, 每行是一次检测结果, 每个结果包含一个(值, 时间戳)的结构
    '''
    def __init__(self, subject_id, anchor_year:int) -> None:
        self.subject_id = subject_id
        self.anchor_year = anchor_year
        self.static_data = {} # dict(fea_name:value)
        self.admissions = []
    
    def append_admission(self, admission:Admission):
        self.admissions.append(admission)
        # 维护时间序列
        if len(self.admissions) >= 1:
            self.admissions = sorted(self.admissions, key=lambda adm:adm.admittime)

    def append_value(self, charttime:float, itemid, value):
        # search admission by charttime
        assert(len(self.admissions) > 0)
        for adm in self.admissions:
            if adm.admittime < charttime and adm.dischtime > charttime:
                adm.append_value(itemid, charttime, value)

    def update_data(self):
        '''将数据整理成连续形式'''
        for adm in self.admissions:
            adm.update_data()




class MIMICIV:
    '''
    MIMIC-IV底层抽象, 对源数据进行抽取/合并/移动, 生成中间数据集, 并将中间数据存储到pickle文件中
    这一步的数据是不对齐的, 仅考虑抽取和类型转化问题
    '''
    def __init__(self):
        self.g_conf = GLOBAL_CONF_LOADER['mimic-iv']
        # paths
        self.mimic_dir = self.g_conf['paths']['mimic_dir']
        # configs
        self.configs = Config(cache_path=self.g_conf['paths']['conf_cache_path'], manual_path=self.g_conf['paths']['conf_manual_path'])
        self.procedure_flag = 'init' # 控制标志, 进行不同阶段的cache和dump
        self.converter = tools.TimeConverter(format="%Y-%m-%d %H:%M:%S", out_unit='hour')
        # variable for phase 1
        self.subject_ids = None
        self.sepsis_icds = None
        self.hosp_item = None
        self.icu_item = None
        # variable for phase 2
        self.subjects = {} # subject_id:Subject

    def preprocess(self):
        self.preprocess_phase1()
        self.preprocess_phase2()
        self.preprocess_phase3()

    def preprocess_phase1(self, from_pkl=True):
        pkl_path = os.path.join(self.g_conf['paths']['cache_dir'], 'phase1.pkl')
        if from_pkl and os.path.exists(pkl_path):
            logger.info(f'load pkl for phase 1 from {pkl_path}')
            with open(pkl_path, 'rb') as fp:
                result = pickle.load(fp)
                self.sepsis_icds = result[0]
                self.subject_ids = result[1] # int
                self.icu_item = result[2]
                self.hosp_item = result[3]
            self.procedure_flag = 'phase1'
            return
        logger.info(f'MIMIC-IV: processing dim file, flag={self.procedure_flag}')
        # 抽取icd名称中含有sepsis的icd编号
        d_diagnoses = pd.read_csv(os.path.join(self.mimic_dir, 'hosp', 'd_icd_diagnoses.csv'), encoding='utf-8')
        sepsis_icds = set()
        for _,row in tqdm(d_diagnoses.iterrows(), desc='sepsis icds'):
            if str(row['long_title']).lower().find('sepsis') != -1:
                sepsis_icds.add(row['icd_code'])
        logger.info('Sepsis icd code: ')
        logger.info(sepsis_icds)
        # 抽取符合条件的患者id
        subject_ids = set()
        subj_diagnoses = pd.read_csv(os.path.join(self.mimic_dir, 'hosp', 'diagnoses_icd.csv'), encoding='utf-8')
        for _,row in tqdm(subj_diagnoses.iterrows(), desc='select subjects', total=len(subj_diagnoses)):
            if str(row['icd_code']) in sepsis_icds:
                subject_ids.add(row['subject_id'])
        logger.info(f'Extracted {len(subject_ids)} sepsis subjects')
        # 建立hospital lab_item编号映射
        d_hosp_item = pd.read_csv(os.path.join(self.mimic_dir, 'hosp', 'd_labitems.csv'), encoding='utf-8')
        hosp_item = {}
        for _,row in tqdm(d_hosp_item.iterrows(), desc='hosp items'):
            hosp_item[row['itemid']] = (row['label'], row['fluid'], row['category'])
        # 建立icu lab_item编号映射
        d_icu_item = pd.read_csv(os.path.join(self.mimic_dir, 'icu', 'd_items.csv'), encoding='utf-8')
        icu_item = {}
        for _,row in tqdm(d_icu_item.iterrows(), desc='icu items'):
            icu_item[row['itemid']] = (row['label'], row['lownormalvalue'], row['highnormalvalue'])
        # 存储cache
        self.subject_ids = subject_ids
        self.sepsis_icds = sepsis_icds
        self.hosp_item = hosp_item
        self.icu_item = icu_item
        with open(pkl_path, 'wb') as fp:
            pickle.dump([sepsis_icds, subject_ids, icu_item, hosp_item], fp)
        self.procedure_flag = 'phase1'
    
    def preprocess_phase2(self):
        logger.info(f'MIMIC-IV: processing subject, flag={self.procedure_flag}')
        # 构建subject
        patients = pd.read_csv(os.path.join(self.mimic_dir, 'hosp', 'patients.csv'), encoding='utf-8')
        for _,row in tqdm(patients.iterrows(), 'construct subject', total=len(patients)):
            s_id = row['subject_id']
            if s_id in self.subject_ids:
                self.subjects[s_id] = Subject(row['subject_id'], anchor_year=row['anchor_year'])
                self.subjects[s_id].static_data['age'] = row['anchor_age']
        self.subject_ids = set(self.subjects.keys())
        # 抽取admission
        table_admission = pd.read_csv(os.path.join(self.mimic_dir, 'hosp', 'admissions.csv'), encoding='utf-8')
        
        for _,row in tqdm(table_admission.iterrows(), desc='extract admission', total=len(table_admission)):
            s_id = row['subject_id']
            if s_id in self.subjects:
                try:
                    adm = Admission(hadm_id=row['hadm_id'], admittime=self.converter(row['admittime']), dischtime=self.converter(row['dischtime']))
                    self.subjects[s_id].append_admission(adm)
                except Exception as e:
                    logger.warning('Invalid admission:' + str(row['hadm_id']))
        # TODO 采集静态数据
        self.procedure_flag = 'phase2'


    def preprocess_phase3(self):
        logger.info(f'MIMIC-IV: processing dynamic data, flag={self.procedure_flag}')
        # 配置准入itemid
        target_icu_ids = self.configs['extract']['target_icu_id']
        for id in target_icu_ids:
            assert(id in self.icu_item)
            logger.info(f'Extract itemid={id}')
        # 采集icu内的动态数据
        logger.info('Loading icu events')
        # TODO 这里不能直接load整个dataset
        # 考虑把原始的icu_event拆成若干份, 然后分开加载
        out_cache_dir = os.path.join(self.g_conf['paths']['cache_dir'], 'icu_events')
        tools.split_csv(os.path.join(self.mimic_dir, 'icu', 'chartevents.csv'), out_folder=out_cache_dir)

        logger.info('Loading icu events:Done')
        icu_events = None
        for file_name in sorted(os.listdir(out_cache_dir)):
            icu_events = pd.read_csv(os.path.join(out_cache_dir, file_name), encoding='utf-8')
            for _,row in tqdm(icu_events.iterrows(), desc='Processing ICU table', total=len(icu_events)):
                s_id, itemid = row['subject_id'], row['itemid']
                if s_id in self.subjects and itemid in target_icu_ids:
                    # TODO hadm_id内部可能会不连续, 这里可能要用stay_id
                    self.subjects[s_id].append_value(charttime=self.converter(row['charttime']), itemid=itemid, value=row['value'])
            del icu_events # release memory
        for s_id in self.subjects:
            self.subjects[s_id].update_data()
        # 保存subjects
        logger.info('Dump subjects')
        with open(os.path.join(self.g_conf['paths']['cache_dir'], 'subjects.pkl'), 'wb') as fp:
            pickle.dump(self.subjects, fp)


class MIMICDataset:
    '''
    MIMIC-IV上层抽象, 从中间文件读取数据, 进行预处理, 得到最终数据集
    '''
    def __init__(self):
        pass


if __name__ == '__main__':
    mimiciv = MIMICIV()
    mimiciv.preprocess()