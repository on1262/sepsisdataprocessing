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
    def duration(self):
        return max(0, self.dischtime - self.admittime)
    
    def empty(self):
        return True if len(self.dynamic_data) == 0 else False

        
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
        for adm in self.admissions:
            if adm.admittime < charttime and adm.dischtime > charttime:
                adm.append_value(itemid, charttime, value)

    def update_data(self):
        '''将数据整理成连续形式'''
        for adm in self.admissions:
            adm.update_data()

    def del_empty_admission(self):
        new_adm = []
        for idx in range(len(self.admissions)):
            if not self.admissions[idx].empty():
                new_adm.append(self.admissions[idx])
        self.admissions = new_adm
    
    def empty(self):
        for adm in self.admissions:
            if not adm.empty():
                return False
        return True


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

        self.preprocess()
        self.del_invalid_subjects()
        logger.info('MIMICIV inited')

    def preprocess(self, from_pkl=True, split_csv=False):
        self.preprocess_phase1(from_pkl)
        self.preprocess_phase2(from_pkl)
        self.preprocess_phase3(split_csv, from_pkl)

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
    
    def preprocess_phase2(self, from_pkl=True):
        if from_pkl:
            return
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


    def preprocess_phase3(self, split_csv=False, from_pkl=True):
        pkl_path = os.path.join(self.g_conf['paths']['cache_dir'], 'subjects.pkl')
        if from_pkl:
            with open(pkl_path, 'rb') as fp:
                self.subjects = pickle.load(fp)
            self.procedure_flag = 'phase3'
            return
        logger.info(f'MIMIC-IV: processing dynamic data, flag={self.procedure_flag}')
        # 配置准入itemid
        target_icu_ids = self.configs['extract']['target_icu_id']
        for id in target_icu_ids:
            assert(id in self.icu_item)
            logger.info(f'Extract itemid={id}')
        # 采集icu内的动态数据
        out_cache_dir = os.path.join(self.g_conf['paths']['cache_dir'], 'icu_events')
        if split_csv:
            tools.split_csv(os.path.join(self.mimic_dir, 'icu', 'chartevents.csv'), out_folder=out_cache_dir)
        icu_events = None
        logger.info('Loading icu events')
        p_bar = tqdm(total=len(os.listdir(out_cache_dir)))
        # 这里需要30min, 随着feature增加会更多
        for file_name in sorted(os.listdir(out_cache_dir)):
            icu_events = pd.read_csv(os.path.join(out_cache_dir, file_name), encoding='utf-8')[['subject_id', 'itemid', 'charttime', 'valuenum']].to_numpy()
            for idx in range(len(icu_events)):
                s_id, itemid = icu_events[idx, 0], icu_events[idx, 1]
                if s_id in self.subjects and itemid in target_icu_ids:
                    self.subjects[s_id].append_value(charttime=self.converter(icu_events[idx, 2]), itemid=itemid, value=icu_events[idx, 3])
            p_bar.update(1)
        for s_id in tqdm(self.subjects, desc='update data'):
            self.subjects[s_id].update_data()
        # 保存subjects
        logger.info('Dump subjects')
        with open(pkl_path, 'wb') as fp:
            pickle.dump(self.subjects, fp)
        logger.info('Dump subjects: Done')
        self.procedure_flag = 'phase3'

    def del_invalid_subjects(self):
        pop_list = []
        for s_id in self.subjects:
            if self.subjects[s_id].empty():
                pop_list.append(s_id)
            else:
                self.subjects[s_id].del_empty_admission()
        for s_id in pop_list:
            self.subjects.pop(s_id)
        logger.info(f'del_invalid_subjects: Deleted {len(pop_list)}/{len(pop_list)+len(self.subjects)} subjects')

    def make_report(self):
        '''进行数据集的信息统计和测试'''
        out_path = os.path.join(self.g_conf['paths']['out_dir'], 'dataset_report.txt')
        write_lines = []
        # basic statistics
        write_lines.append('='*10 + 'basic' + '='*10)
        write_lines.append(f'subjects:{len(self.subjects)}')
        adm_nums = np.mean([len(s.admissions) for s in self.subjects.values()])
        write_lines.append(f'average admission number per subject:{adm_nums}')
        avg_adm_time = []
        for s in self.subjects.values():
            for adm in s.admissions:
                avg_adm_time.append(adm.duration())

        write_lines.append(f'average admission time(hour): {np.mean(avg_adm_time)}')
        
        with open(out_path, 'w', encoding='utf-8') as fp:
            for line in write_lines:
                fp.write(line + '\n')
        logger.info(f'Generated report at {out_path}')
            

            




class MIMICDataset:
    '''
    MIMIC-IV上层抽象, 从中间文件读取数据, 进行处理, 得到最终数据集
    '''
    def __init__(self):
        self.mimiciv = MIMICIV()
        self.subjects = self.mimiciv.subjects

if __name__ == '__main__':
    dataset = MIMICDataset()
    print('test Done')
    