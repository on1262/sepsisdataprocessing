"""mimic数据集抽象"""

import os
import pickle
import json
import tools
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tools import GLOBAL_CONF_LOADER
from tools import logger
from tqdm import tqdm
from .abstract_dataset import AbstractDataset
# import multiprocessing as mp


class Admission:
    '''
    代表Subject/Admission
    admittime: 起始时间
    dischtime: 结束时间
    '''
    def __init__(self, hadm_id, admittime:float, dischtime:float) -> None:
        self.dynamic_data = {} # dict(fea_name:ndarray(value, time))
        assert(admittime < dischtime)
        self.hadm_id = hadm_id
        self.admittime = admittime
        self.dischtime = dischtime
    
    def append_dynamic(self, itemid, time:float, value):
        if self.dynamic_data.get(itemid) is None:
            self.dynamic_data[itemid] = [(value, time)]
        else:
            self.dynamic_data[itemid].append((value, time))

    def update_data(self):
        '''绝对时间变为相对时间'''
        for key in self.dynamic_data:
            if isinstance(self.dynamic_data[key], list):
                arr = np.asarray(sorted(self.dynamic_data[key], key=lambda x:x[1]))
                arr[:, 1] -= self.admittime
                self.dynamic_data[key] = arr
    
    def duration(self):
        return max(0, self.dischtime - self.admittime)
    
    def empty(self):
        return True if len(self.dynamic_data) == 0 else False

    def __getitem__(self, idx):
        return self.dynamic_data[idx]

    def keys(self):
        return self.dynamic_data.keys()

        
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

    def append_static(self, charttime:float, name, value):
        if charttime is None:
            self.static_data[name] = value
        else:
            if name not in self.static_data:
                self.static_data[name] = [(value, charttime)]
            else:
                self.static_data[name].append((value, charttime))
    
    def nearest_static(self, key, time):
        if key not in self.static_data.keys():
            return -1

        if not isinstance(self.static_data[key], np.ndarray): # single value
            return self.static_data[key]
        else:
            nearest_idx, delta = 0, np.inf
            for idx in range(self.static_data[key].shape[0]):
                new_delta = np.abs(time-self.static_data[key][idx, 1])
                if new_delta < delta:
                    delta = new_delta
                    nearest_idx = idx
            return self.static_data[key][nearest_idx, 0]

    def append_dynamic(self, charttime:float, itemid, value):
        # search admission by charttime
        for adm in self.admissions:
            if adm.admittime < charttime and charttime < adm.dischtime:
                adm.append_dynamic(itemid, charttime, value)

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
        self.gbl_conf = GLOBAL_CONF_LOADER['mimic-iv']
        # paths
        self.mimic_dir = self.gbl_conf['paths']['mimic_dir']
        # configs
        self.loc_conf = Config(cache_path=self.gbl_conf['paths']['conf_cache_path'], manual_path=self.gbl_conf['paths']['conf_manual_path'])
        self.procedure_flag = 'init' # 控制标志, 进行不同阶段的cache和dump
        self.converter = tools.TimeConverter(format="%Y-%m-%d %H:%M:%S", out_unit='hour')
        self.target_icu_ids = self.loc_conf['extract']['target_icu_id']
        # variable for phase 1
        self.sepsis_result = None
        self.hosp_item = None
        self.icu_item = None
        # variable for phase 2
        self.subjects = {} # subject_id:Subject

        self.preprocess()
        # post process
        self.remove_invalid_data(rules=self.loc_conf['extract']['remove_rule'])
        logger.info('MIMICIV inited')

    def preprocess(self, from_pkl=True, split_csv=False):
        self.preprocess_phase1(from_pkl)
        self.preprocess_phase2(from_pkl)
        self.preprocess_phase3(split_csv, from_pkl)
 
    def preprocess_phase1(self, from_pkl=True):
        pkl_path = os.path.join(self.gbl_conf['paths']['cache_dir'], '1_phase1.pkl')
        if from_pkl and os.path.exists(pkl_path):
            logger.info(f'load pkl for phase 1 from {pkl_path}')
            with open(pkl_path, 'rb') as fp:
                result = pickle.load(fp)
                self.sepsis_result = result[0] # int
                self.icu_item = result[1]
                self.hosp_item = result[2]
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
        sepsis_result = load_sepsis_patients(self.gbl_conf['paths']['sepsis_patient_path'])
        logger.info(f'Extracted {len(self.sepsis_result)} sepsis subjects')
        # 建立hospital lab_item编号映射
        d_hosp_item = pd.read_csv(os.path.join(self.mimic_dir, 'hosp', 'd_labitems.csv'), encoding='utf-8')
        hosp_item = {}
        for _,row in tqdm(d_hosp_item.iterrows(), desc='hosp items'):
            hosp_item[str(row['itemid'])] = (row['label'], row['fluid'], row['category'])
        # 建立icu lab_item编号映射
        d_icu_item = pd.read_csv(os.path.join(self.mimic_dir, 'icu', 'd_items.csv'), encoding='utf-8')
        icu_item = {}
        for _,row in tqdm(d_icu_item.iterrows(), desc='icu items'):
            icu_item[str(row['itemid'])] = (row['label'], row['category'], row['param_type'], row['lownormalvalue'], row['highnormalvalue'])
        # 存储cache
        self.hosp_item = hosp_item
        self.icu_item = icu_item
        self.sepsis_result = sepsis_result
        with open(pkl_path, 'wb') as fp:
            pickle.dump([sepsis_result, icu_item, hosp_item], fp)
        self.procedure_flag = 'phase1'
    
    def preprocess_phase2(self, from_pkl=True):
        pkl_path = os.path.join(self.gbl_conf['paths']['cache_dir'], '2_phase2.pkl')
        if from_pkl and os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as fp:
                (self.subjects, self.sepsis_result) = pickle.load(fp)
            logger.info(f'load pkl for phase 2 from {pkl_path}')
            self.procedure_flag = 'phase2' 
            return
        logger.info(f'MIMIC-IV: processing subject, flag={self.procedure_flag}')
        # 抽取在sepsis dict中的s_id, 构建subject
        patients = pd.read_csv(os.path.join(self.mimic_dir, 'hosp', 'patients.csv'), encoding='utf-8')
        for _,row in tqdm(patients.iterrows(), 'construct subject', total=len(patients)):
            s_id = row['subject_id']
            if s_id in self.sepsis_result:
                sepsis_time, stay_id, sofa_score, respiration, liver, cardiovascular, cns, renal = self.sepsis_result[s_id]
                self.subjects[s_id] = Subject(row['subject_id'], anchor_year=row['anchor_year'])
                self.subjects[s_id].append_static(None, 'age', row['anchor_age'])
                self.subjects[s_id].append_static(None, 'gender', row['gender'])
                self.subjects[s_id].append_static(None, 'sepsis_time', sepsis_time)
                self.subjects[s_id].append_static(None, 'sofa_score', sofa_score)
                self.subjects[s_id].append_static(None, 'respiration', respiration)
                self.subjects[s_id].append_static(None, 'liver', liver)
                self.subjects[s_id].append_static(None, 'cardiovascular', cardiovascular)
                self.subjects[s_id].append_static(None, 'cns', cns)
                self.subjects[s_id].append_static(None, 'renal', renal)
        # 更新sepsis_result, 去除不存在的s_id
        self.sepsis_result = {key:self.sepsis_result[key] for key in self.subjects.keys()}
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
        omr = pd.read_csv(os.path.join(self.mimic_dir, 'hosp', 'omr.csv'), encoding='utf-8') # [subject_id,chartdate,seq_num,result_name,result_value]
        converter = tools.TimeConverter(format="%Y-%m-%d", out_unit='hour')
        omr = omr.to_numpy()
        for idx in range(omr.shape[0]):
            s_id = omr[idx, 0]
            if s_id in self.subjects and int(omr[idx, 2]) == 1:
                self.subjects[s_id].append_static(converter(omr[idx, 1]), omr[idx, 3], omr[idx, 4])
        # dump
        with open(pkl_path, 'wb') as fp:
            pickle.dump((self.subjects, self.sepsis_result), fp)
        logger.info(f'Phase 2 dumped at {pkl_path}')
        self.procedure_flag = 'phase2'


    def preprocess_phase3(self, split_csv=False, from_pkl=True):
        pkl_path = os.path.join(self.gbl_conf['paths']['cache_dir'], '3_subjects.pkl')
        if from_pkl and os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as fp:
                self.subjects = pickle.load(fp)
            logger.info(f'load pkl for phase 3 from {pkl_path}')
            self.procedure_flag = 'phase3'
            return
        logger.info(f'MIMIC-IV: processing dynamic data, flag={self.procedure_flag}')
        # 配置准入itemid
        for id in self.target_icu_ids:
            assert(id in self.icu_item)
            logger.info(f'Extract itemid={id}')
        enabled_item_id = set()
        for id in self.icu_item:
            if self.icu_item[id][2] in ['Numeric', 'Numeric with tag'] and self.icu_item[id][1] not in ['Alarms']:
                enabled_item_id.add(id)
        # 采集icu内的动态数据
        out_cache_dir = os.path.join(self.gbl_conf['paths']['cache_dir'], 'icu_events')
        if split_csv:
            tools.split_csv(os.path.join(self.mimic_dir, 'icu', 'chartevents.csv'), out_folder=out_cache_dir)
        icu_events = None
        logger.info('Loading icu events')
        p_bar = tqdm(total=len(os.listdir(out_cache_dir)))
        for file_name in sorted(os.listdir(out_cache_dir)):
            icu_events = pd.read_csv(os.path.join(out_cache_dir, file_name), encoding='utf-8')[['subject_id', 'itemid', 'charttime', 'valuenum']].to_numpy()
            for idx in range(len(icu_events)):
                s_id, itemid = icu_events[idx, 0], str(icu_events[idx, 1])
                if s_id in self.subjects and itemid in enabled_item_id:
                    self.subjects[s_id].append_dynamic(charttime=self.converter(icu_events[idx, 2]), itemid=itemid, value=icu_events[idx, 3])
            p_bar.update(1)
        for s_id in tqdm(self.subjects, desc='update data'):
            self.subjects[s_id].update_data()
        # 删去空的subject和空的admission
        self.remove_invalid_data(rules=None)
        # 保存subjects
        logger.info('Dump subjects')
        with open(pkl_path, 'wb') as fp:
            pickle.dump(self.subjects, fp)
        logger.info('Dump subjects: Done')
        self.procedure_flag = 'phase3'

    def remove_invalid_data(self, rules=None):
        '''当rules=None时, 只清除空的subject和admission. 当rules不为None时, 会检查target_id是否都满足采集要求'''
        if rules is not None:
            for s_id in self.subjects:
                if not self.subjects[s_id].empty():
                    new_adm_idx = []
                    for idx, adm in enumerate(self.subjects[s_id].admissions):
                        flag = 1
                        # 检查duration, points, interval
                        for target_id in rules['target_id']:
                            if target_id not in adm.keys():
                                flag = 0
                        if flag != 0:
                            pao2_id =  rules['target_id'][0]
                            fio2_id = rules['target_id'][1]
                            start_time = max(adm[pao2_id][0,1], adm[fio2_id][0,1])
                            end_time = min(adm[pao2_id][-1,1], adm[fio2_id][-1,1])
                            dur = end_time - start_time
                            if dur > max(rules['min_duration'],0) and dur < rules['max_duration']:
                                points = min(np.sum((adm[pao2_id][:,1] >= start_time) * (adm[pao2_id][:,1] <= end_time)), \
                                    np.sum((adm[fio2_id][:,1] >= start_time) * (adm[fio2_id][:,1] <= end_time)))
                                if points >= rules['min_points'] and dur/points <= rules['max_avg_interval']:
                                    flag *= 1
                                else:
                                    flag = 0
                            else:
                                flag = 0
                        
                        if flag != 0:
                            new_adm_idx.append(idx)
                    self.subjects[s_id].admissions = [self.subjects[s_id].admissions[idx] for idx in new_adm_idx]
        pop_list = []
        for s_id in self.subjects:
            if self.subjects[s_id].empty():
                pop_list.append(s_id)
            else:
                self.subjects[s_id].del_empty_admission()
        for s_id in pop_list:
            self.subjects.pop(s_id)
        logger.info(f'del_invalid_subjects: Deleted {len(pop_list)}/{len(pop_list)+len(self.subjects)} subjects')

    def generate_subdataset(self, p_origin, p_save, sids:set):
        '''
        抽取首列符合subject id的所有项, 用于debug
        p_origin: 源文件path, 可以是目录(会读里面所有源文件)/path list/path, 要求文件首列是subject id
        p_save: 新文件path
        sids: 待选subject id (int)
        '''
        titled = False
        logger.info('Generating sub dataset')
        assert(os.path.exists(p_origin))
        if os.path.isdir(p_origin):
            file_list = [os.path.join(p_origin, name) for name in sorted(os.listdir(p_origin))]
        elif os.path.isfile(p_origin):
            file_list = [p_origin]
        elif isinstance(p_origin, list):
            file_list = p_origin
        else:
            assert(0)
        with open(p_save, 'w', encoding='utf-8') as tp:
            for p_file in tqdm(file_list, desc='icu-events'):
                with open(p_file, mode='r', encoding='utf-8') as fp:
                    s_list = fp.readlines() # contain \n
                    for idx, sentence in enumerate(s_list):
                        if idx == 0 and not titled:
                            tp.write(sentence)
                            titled = True
                        if idx > 0 and len(sentence) > 10:
                            id = int(sentence.split(',')[0])
                            if id in sids:
                                tp.write(sentence)


        
    def make_report(self):
        '''进行数据集的信息统计'''
        out_path = os.path.join(self.gbl_conf['paths']['out_dir'], 'dataset_report.txt')
        dist_dir = os.path.join(self.gbl_conf['paths']['out_dir'], 'report_dist')
        tools.reinit_dir(dist_dir, build=True)
        logger.info('MIMIC-IV: generating dataset report')
        write_lines = []
        # basic statistics
        write_lines.append('='*10 + 'basic' + '='*10)
        write_lines.append(f'subjects:{len(self.subjects)}')
        adm_nums = np.mean([len(s.admissions) for s in self.subjects.values()])
        write_lines.append(f'average admission number per subject:{adm_nums:.2f}')
        avg_adm_time = []
        for s in self.subjects.values():
            for adm in s.admissions:
                avg_adm_time.append(adm.duration())
        write_lines.append(f'average admission time(hour): {np.mean(avg_adm_time):.2f}')
        for id in self.target_icu_ids:
            fea_name = self.icu_item[id][0]
            write_lines.append('='*10 + f'{fea_name}({id})' + '='*10)
            arr_points = []
            arr_duration = []
            arr_frequency = []
            arr_min_interval = []
            arr_max_interval = []
            arr_avg_value = []
            for s in self.subjects.values():
                for adm in s.admissions:
                    if id in adm.keys():
                        arr_points.append(adm[id].shape[0])
                        arr_duration.append(adm[id][-1,1] - adm[id][0,1])
                        arr_frequency.append(arr_points[-1] / arr_duration[-1])
                        arr_min_interval.append(np.diff(adm[id][:,1]).min())
                        arr_max_interval.append(np.diff(adm[id][:,1]).max())
                        arr_avg_value.append(adm[id][:,0].mean())
                        assert(arr_duration[-1] > 0)
            arr_points, arr_duration, arr_frequency, arr_min_interval,arr_max_interval,arr_avg_value = np.asarray(arr_points), np.asarray(arr_duration), np.asarray(arr_frequency), np.asarray(arr_min_interval), np.asarray(arr_max_interval), np.asarray(arr_avg_value)
            write_lines.append(f'average points per admission: {arr_points.mean():.3f}')
            write_lines.append(f'average duration(hour) per admission: {arr_duration.mean():.3f}')
            write_lines.append(f'average frequency(point/hour) per admission: {arr_frequency.mean():.3f}')
            write_lines.append(f'average min interval per admission: {arr_min_interval.mean():.3f}')
            write_lines.append(f'average avg value per admission: {arr_avg_value.mean():.3f}')
            # plot distribution
            tools.plot_single_dist(
                data=arr_points, data_name=f'Points of {fea_name}', 
                save_path=os.path.join(dist_dir, 'point_' + str(id) + '.png'), discrete=False, restrict_area=True)
            tools.plot_single_dist(
                data=arr_duration, data_name=f'Duration of {fea_name}(Hour)', 
                save_path=os.path.join(dist_dir, 'duration_' + str(id) + '.png'), discrete=False, restrict_area=True)
            tools.plot_single_dist(
                data=arr_frequency, data_name=f'Frequency of {fea_name}(Point/Hour)', 
                save_path=os.path.join(dist_dir, 'freq_' + str(id) + '.png'), discrete=False, restrict_area=True)
            tools.plot_single_dist(
                data=arr_avg_value, data_name=f'Avg Value of {fea_name}', 
                save_path=os.path.join(dist_dir, 'avgv_' + str(id) + '.png'), discrete=False, restrict_area=True)
            tools.plot_single_dist(
                data=arr_min_interval, data_name=f'Min interval of {fea_name}', 
                save_path=os.path.join(dist_dir, 'mininterv_' + str(id) + '.png'), discrete=False, restrict_area=True)
            tools.plot_single_dist(
                data=arr_max_interval, data_name=f'Max interval of {fea_name}', 
                save_path=os.path.join(dist_dir, 'maxinterv_' + str(id) + '.png'), discrete=False, restrict_area=True)
        # itemid hit rate
        hit_table = {}
        adm_count = np.sum([len(s.admissions) for s in self.subjects.values()])
        for sub in self.subjects.values():
            for adm in sub.admissions:
                for id in adm.keys():
                    if hit_table.get(id) is None:
                        hit_table[id] = 1
                    else:
                        hit_table[id] += 1
        key_list = sorted(hit_table.keys(), key= lambda key:hit_table[key], reverse=True)
        write_lines.append('='*10 + 'Feature hit table(>0.5)' + '='*10)
        for key in key_list:
            fea_name = self.icu_item[key][0]
            value = hit_table[key] / adm_count
            if value < 0.5:
                continue
            write_lines.append(f"{value:.2f}\t({key}){fea_name} ")

        with open(out_path, 'w', encoding='utf-8') as fp:
            for line in write_lines:
                fp.write(line + '\n')
        logger.info(f'Report generated at {out_path}')


class MIMICDataset(AbstractDataset):
    '''
    MIMIC-IV上层抽象, 从中间文件读取数据, 进行处理, 得到最终数据集
    '''
    def __init__(self):
        super().__init__()
        self.mimiciv = MIMICIV()
        self.subjects = self.mimiciv.subjects
        self.g_conf = self.mimiciv.gbl_conf
        self.configs = self.mimiciv.loc_conf
        # preload data
        self.data = None # ndarray(samples, n_fea, ticks)
        self.norm_dict = None # key=str(name/id) value={'mean':mean, 'std':std}
        self.static_keys = None # list(str)
        self.dynamic_keys = None # list(str)
        self.seqs_len = None # list(available_len)

        self.preprocess()
        
        logger.info(f'Dataset length={self.data.shape[0]} admissions')
        self.target_name = self.configs['process']['target_label']
        self._additional_feas = self.configs['process']['additional_features']
        self.target_idx = self.data.shape[1] - 1
        # idx_dict: fea_name->idx
        self.idx_dict = dict(
            {str(key):val for val, key in enumerate(self.static_keys)}, \
                **{str(key):val+len(self.static_keys) for val, key in enumerate(self.dynamic_keys)})
        
        # 这里设置str是为了使得特征名可以作为dict的keyword索引
        self.norm_dict = {str(key):val for key, val in self.norm_dict.items()}
        self.static_keys = [str(key) for key in self.static_keys]
        self.dynamic_keys = [str(key) for key in self.dynamic_keys]
        self.total_keys = self.static_keys + self.dynamic_keys
        self.icu_item = {str(key):val for key, val in self.mimiciv.icu_item.items()}
        self.hosp_item = {str(key):val for key, val in self.mimiciv.hosp_item.items()}
        # mode switch
        self.index = None # 当前模式(train/test)的index list, None表示使用全部数据
        self.train_index = None
        self.valid_index = None
        self.test_index = None

    def get_fea_label(self, key_or_idx):
        '''输入key/idx得到关于特征的简短描述, 从icu_item中提取'''
        if isinstance(key_or_idx, int):
            name = self.total_keys[key_or_idx]
        else:
            name = key_or_idx
        if self.icu_item.get(name) is not None:
            return self.icu_item[name][0]
        else:
            logger.warning(f'No fea label for: {name} return name')
            return name

    def register_split(self, train_index, valid_index, test_index):
        self.train_index = train_index
        self.valid_index = valid_index
        self.test_index = test_index

    def mode(self, mode=['train', 'valid', 'test', 'all']):
        '''切换dataset的模式, train/valid/test需要在register_split方法调用后才能使用'''
        if mode == 'train':
            self.index = self.train_index
            assert(self.index is not None)
        elif mode =='valid':
            self.index = self.valid_index
            assert(self.index is not None)
        elif mode =='test':
            self.index = self.test_index
            assert(self.index is not None)
        elif mode == 'all':
            self.index = None
        else:
            assert(0)

    def get_norm_array(self):
        '''返回一个array, [:,0]代表各个feature的均值, [:,1]代表方差'''
        means = [[self.norm_dict[key]['mean'] , self.norm_dict[key]['std']] for key in self.total_keys]
        return np.asarray(means)

    def restore_norm(self, name_or_idx, data:np.ndarray, mask=None) -> np.ndarray:
        '''
        缩放到正常范围
        mask: 只变换mask=True的部分
        '''
        # TODO 这个地方有bug, -1和norm后的值域重合了
        if isinstance(name_or_idx, int):
            name_or_idx = self.total_keys[name_or_idx]
        norm = self.norm_dict[name_or_idx]
        if mask is None:
            return data * norm['std'] + norm['mean']
        else:
            assert(mask.shape == data.shape)
            return (data * norm['std'] + norm['mean']) * (mask > 0) + data * (mask <= 0)

    def preprocess(self, from_pkl=True):
        p_numeric_subject = os.path.join(self.g_conf['paths']['cache_dir'], 'numeric_subject.pkl')
        p_norm_dict = os.path.join(self.g_conf['paths']['cache_dir'], 'norm_dict.pkl')
        # preprocessing
        self.static_feas = set()
        self.target_label = self.configs['process']['target_label']
        self.dyamic_ids = self.configs['process']['dynamic_id']

        if from_pkl and os.path.exists(p_numeric_subject):
            with open(p_numeric_subject, 'rb') as fp:
                self.subjects = pickle.load(fp)
            logger.info(f'Load numeric subject data from {p_numeric_subject}')
        else:
            # 这里不要随便改变调用顺序
            self.preprocess_to_num() # 这里会改变static_data的key list
            with open(p_numeric_subject, 'wb') as fp:
                pickle.dump(self.subjects, fp)
            logger.info(f'Numeric subjects dumped at {p_numeric_subject}')
        
        for s in self.subjects.values():
            for key in s.static_data.keys():
                self.static_feas.add(key)
        
        if from_pkl and os.path.exists(p_norm_dict):
            with open(p_norm_dict, 'rb') as fp:
                self.norm_dict = pickle.load(fp)
            logger.info(f'Load norm dict from {p_norm_dict}')
        else:
            self.norm_dict = self.preprocess_norm()
            with open(p_norm_dict, 'wb') as fp:
                pickle.dump(self.norm_dict, fp)
            logger.info(f'Norm dict dumped at {p_norm_dict}')
        self.preprocess_table(from_pkl=from_pkl)
        
    def preprocess_to_num(self):
        '''
        将所有特征转化为数值型, 并且对于异常值进行处理
        1. 对特定格式的特征进行转换(血压)
        2. 检测不能转换为float的静态特征
        3. 对FiO2进行缩放, 对PaO2进行正常值约束
        4. 对PaO2异常峰检测并硬平滑
        '''
        for s in self.subjects.values():
            for key in list(s.static_data.keys()):
                if key == 'gender':
                    s.static_data[key] = 0 if s.static_data[key] == 'F' else 1
                elif 'Blood Pressure' in key:
                    s.static_data['systolic pressure'] = []
                    s.static_data['diastolic pressure'] = []
                    for idx in range(len(s.static_data[key])):
                        p_result = s.static_data[key][idx][0].split('/')
                        time = s.static_data[key][idx][1]
                        vs, vd = float(p_result[0]), float(p_result[1])
                        s.static_data['systolic pressure'].append((vs, time))
                        s.static_data['diastolic pressure'].append((vd, time))
                    s.static_data.pop(key)
                    s.static_data['systolic pressure'] = np.asarray(s.static_data['systolic pressure'])
                    s.static_data['diastolic pressure'] = np.asarray(s.static_data['diastolic pressure'])
                elif key != 'age':
                    valid_idx = []
                    for idx in range(len(s.static_data[key])):
                        v,t = s.static_data[key][idx]
                        try:
                            v = float(v)
                            s.static_data[key][idx] = (v,t)
                            valid_idx.append(idx)
                        except Exception as e:
                            logger.warning(f'Invalid value {v} for {key}')
                    s.static_data[key] = np.asarray(s.static_data[key])[valid_idx, :]
            for adm in s.admissions:
                for id in adm.keys():
                    if id == "223835": # fio2, 空气氧含量
                        data = adm[id][:,0]
                        adm[id][:,0] = (data * (data > 20) + 21*np.ones(data.shape) * (data <= 20)) * 0.01
                    elif id == "220224":
                        data = adm[id][:,0]
                        adm[id][:,0] = (data * (data < 600) + 600*np.ones(data.shape) * (data >= 600))
            # PaO2 异常峰检测进行硬平滑
            logger.info('Anomaly peak detection')
            for adm in s.admissions:
                if "220224" in adm.keys():
                    adm["220224"][:, 0] = reduce_peak(adm["220224"][:, 0])
    
    def preprocess_norm(self) -> dict:
        '''制作norm_dict'''
        norm_dict = {}
        for s in self.subjects.values():
            # static data
            for key in s.static_data.keys():
                if key not in norm_dict:
                    norm_dict[key] = [s.static_data[key]]
                else:
                    norm_dict[key].append(s.static_data[key])
            # dynamic data
            for adm in s.admissions:
                for key in adm.keys():
                    if key not in norm_dict:
                        norm_dict[key] = [adm[key]]
                    else:
                        norm_dict[key].append(adm[key])
        for key in norm_dict:
            if isinstance(norm_dict[key][0], (float, int)):
                norm_dict[key] = np.asarray(norm_dict[key])
            elif isinstance(norm_dict[key][0], np.ndarray):
                norm_dict[key] = np.concatenate(norm_dict[key], axis=0)[:, 0]
            else:
                assert(0)
        for key in norm_dict:
            mean, std = np.mean(norm_dict[key]), np.std(norm_dict[key])
            norm_dict[key] = {'mean':mean, 'std':std}
        return norm_dict

    def preprocess_table(self, from_pkl=True, t_step=0.5):
        '''对每个subject生成时间轴对齐的表, tick(hour)是生成的表的间隔'''
        p_origin_table = os.path.join(self.g_conf['paths']['cache_dir'], 'table_origin.pkl')
        p_final_table = os.path.join(self.g_conf['paths']['cache_dir'], 'table_norm.pkl')
        p_length = os.path.join(self.g_conf['paths']['cache_dir'], 'table_length.pkl')
        # step1: 插值并生成表格
        if from_pkl and os.path.exists(p_origin_table):
            with open(p_origin_table, 'rb') as fp:
                self.data, self.norm_dict, self.static_keys, self.dynamic_keys = pickle.load(fp)
            logger.info(f'load original aligned table from {p_origin_table}')
        else:
            data = []
            align_id = self.configs['process']['align_target_id'] # 用来确认对齐的基准时间
            static_keys = list(self.static_feas)
            dynamic_keys = self.dyamic_ids + self._additional_feas
            # 基准id和对应的index
            pao2_id, fio2_id =  "220224", "223835"
            index_dict = {'pao2':dynamic_keys.index("220224"), 'fio2':dynamic_keys.index("223835"), 
                'hr':dynamic_keys.index("220045"), 'sbp':dynamic_keys.index("220050"), 'dbp':dynamic_keys.index("220051")}
            index_dict = {key:index_dict[key] + len(static_keys) for key in index_dict.keys()}
            additional_vals = {key:[] for key in self._additional_feas} # 记录特征工程生成的特征, 用于更新norm_dict
            for s_id in tqdm(self.subjects.keys(), desc='Generate aligned table'):
                for adm in self.subjects[s_id].admissions:
                    if align_id not in adm.keys():
                        logger.warning('Invalid admission: no align id')
                        continue
                    # 起始时刻和终止时刻为pao2和fio2重合时段, 并且限制在sepsis出现时刻之后
                    t_start = max(adm[pao2_id][0,1], adm[fio2_id][0,1])
                    t_end = min(adm[pao2_id][-1,1], adm[fio2_id][-1,1])
                    sepsis_time = (self.subjects[s_id].static_data['sepsis_time'] - adm.admittime)
                    if sepsis_time >= t_end:
                        logger.warning('Invalid admission: no matching sepsis time')
                        continue
                    else:
                        t_start = max(sepsis_time, t_start)
                    
                    ticks = np.arange(t_start, t_end, t_step) # 最后一个会确保间隔不变且小于t_end
                    # 生成表本身, 缺失值为-1
                    table = -np.ones((len(static_keys) + len(dynamic_keys), ticks.shape[0]), dtype=np.float32)
                    # 填充static data
                    static_data = np.zeros((len(static_keys)))
                    for idx, key in enumerate(static_keys):
                        static_data[idx] = self.subjects[s_id].nearest_static(key, adm[align_id][0, 1] + adm.admittime)
                    table[:len(static_keys), :] = static_data[:, None]
                    # 插值dynamic data
                    for idx, key in enumerate(dynamic_keys[:-1]):
                        if key not in adm.keys():
                            continue
                        table[static_data.shape[0]+idx, :] = np.interp(x=ticks, xp=adm[key][:, 1], fp=adm[key][:, 0])
                    # 检查
                    if not np.all(table[index_dict['fio2'], :] > 0):
                        logger.warning('Skipped Zero FiO2 table')
                        continue
                    self._feature_engineering(table, index_dict, self._additional_feas) # 特征工程
                    for idx, key in enumerate(reversed(self._additional_feas)): # 记录附加信息
                        additional_vals[key].append(table[-(idx+1), :])
                    data.append(table)
            additional_vals = {key:np.concatenate(val, axis=0) for key, val in additional_vals.items()}
            additional_vals = {key:{'mean':val.mean(), 'std':val.std()} for key,val in additional_vals.items()}
            for key in additional_vals:
                self.norm_dict[key] = additional_vals[key]
            self.data = data
            self.static_keys = static_keys
            self.dynamic_keys = dynamic_keys
            with open(p_origin_table, 'wb') as fp:
                pickle.dump([self.data, self.norm_dict, static_keys, dynamic_keys], fp)
            logger.info(f'data table dumped at {p_origin_table}')
            
        # step2: 补充缺失值
        if from_pkl and os.path.exists(p_final_table):
            with open(p_final_table, 'rb') as fp:
                self.data = pickle.load(fp)
            logger.info(f'load normalized aligned table from {p_final_table}')
        else:
            # 缺失值采用均值填充
            for s_id in tqdm(range(len(self.data)), desc='normalize'):
                for idx, key in enumerate(self.static_keys):
                    if np.abs(self.data[s_id][idx, 0] + 1) < 1e-4:
                        self.data[s_id][idx, :] = self.norm_dict[str(key)]['mean']
                
                for idx, key in enumerate(self.dynamic_keys):
                    arr_idx = idx + len(self.static_keys)
                    if np.abs(self.data[s_id][arr_idx, 0] + 1) < 1e-4:
                        self.data[s_id][arr_idx, :] = self.norm_dict[str(key)]['mean']
                
            with open(p_final_table, 'wb') as fp:
                pickle.dump(self.data, fp)
            logger.info(f'data table dumped at {p_final_table}')
        # step3: 时间轴长度对齐, 生成seqs_len, 进行某些特征的最后处理
        if from_pkl and os.path.exists(p_length):
            with open(p_length, 'rb') as fp:
                seqs_len, self.static_keys, self.data = pickle.load(fp)
                self.seqs_len = seqs_len
            logger.info(f'load length aligned table from {p_length}')
        else:
            seqs_len = [d.shape[1] for d in self.data]
            self.seqs_len = seqs_len
            max_len = max(seqs_len)
            n_fea = len(self.static_keys) + len(self.dynamic_keys)
            for t_idx in tqdm(range(len(self.data)), desc='length alignment'):
                if seqs_len[t_idx] == max_len:
                    continue
                new_table = -np.ones((n_fea, max_len - seqs_len[t_idx]))
                self.data[t_idx] = np.concatenate([self.data[t_idx], new_table], axis=1)
            self.data = np.stack(self.data, axis=0) # (n_sample, n_fea, seqs_len)
            # 合并weight/height的重复特征
            rest_feas = list(range(self.data.shape[1]))
            pop_list = []
            for key in ['Weight', 'Height', 'Blood Pressure', 'BMI']:
                if key in self.static_keys:
                    idx = max([0 if self.static_keys[idx] != key else idx for idx in range(len(self.static_keys))])
                    pop_list.append(idx)
            pop_list = sorted(pop_list, reverse=True) # 从大到小
            for idx in pop_list:
                self.static_keys.pop(idx)
                rest_feas.pop(idx)
            self.data = self.data[:,  rest_feas, :]
            with open(p_length, 'wb') as fp:
                pickle.dump((seqs_len, self.static_keys, self.data), fp)
            logger.info(f'length aligned table dumped at {p_length}')

    def _feature_engineering(self, table:np.ndarray, index_dict:dict, addi_feas:list):
        '''
        特征工程, 增加某些计算得到的特征
        table: (n_feas, ticks) 单个subject的table
        index_dict: dict(str:int(index)) 特征名字对应的位置, 已经加上了len(static_keys)
        addi_feas: list(str) 需要计算的特征名字
            默认addi_feas在table的末尾, 载入的顺序和addi_feas相同
        '''
        for idx, name in enumerate(reversed(addi_feas)):
            t_idx = -(idx+1)
            if name == 'PF_ratio':
                table[t_idx, :] = table[index_dict['pao2'], :] / table[index_dict['fio2'], :]
            elif name == 'shock_index':
                if np.all(table[index_dict['sbp'], :] > 0):
                    table[t_idx, :] = table[index_dict['hr'], :] / table[index_dict['sbp'], :]
                else:
                    table[t_idx, :] = 0
                    logger.warning('feature_engineering: skip shock_index with zero sbp')
            elif name == 'MAP':
                table[t_idx, :] = (table[index_dict['sbp'], :] + 2*table[index_dict['dbp'], :]) / 3
            elif name == 'PPD':
                table[t_idx, :] = table[index_dict['sbp'], :] - table[index_dict['dbp'], :]
            else:
                logger.error(f'Invalid feature name:{name}')
                assert(0)



    def __getitem__(self, idx):
        if self.index is None:
            return {'data': self.data[idx, :, :], 'length': self.seqs_len[idx]}
        else:
            return {'data': self.data[self.index[idx], :, :], 'length': self.seqs_len[self.index[idx]]}

    def __len__(self):
        if self.index is None:
            return self.data.shape[0]
        else:
            return len(self.index)

def generate_subset():
    p_origin = os.path.join(dataset.mimiciv.mimic_dir, 'icu', 'outputevents.csv')
    p_save = os.path.join(GLOBAL_CONF_LOADER['mimic_analyzer']['paths']['out_dir'], 'subset_output.csv')
    dataset.mimiciv.generate_subdataset(p_origin=p_origin, p_save=p_save, sids={10161042, 10014610, 10122371})

def reduce_peak(x: np.ndarray):
    '''
    清除x中的异常峰
    x: 1d ndarray
    '''
    window_size = 3  # Size of the sliding window
    threshold = 1.4  # Anomaly threshold at (thredhold-1)% higher than nearby points
    for i in range(len(x)):
        left_window_size = min(window_size, i)
        right_window_size = min(window_size, len(x) - i - 1)
        window = x[i - left_window_size: i + right_window_size + 1]
        
        avg_value = (np.sum(window) - x[i]) / (len(window)-1)
        if x[i] >= avg_value * threshold and x[i] > 110:
            x[i] = avg_value
    return x

def load_sepsis_patients(csv_path:str):
    '''
    从csv中读取按照mimic-iv pipeline生成的sepsis3.0表格
    sepsis_dict: dict(int(subject_id):list)
        list: [sepsis_time(float), stay_id, sofa_score, respiration, liver, cardiovascular, cns, renal]
    '''
    converter = tools.TimeConverter(format="%Y-%m-%d %H:%M:%S", out_unit='hour')
    sepsis_dict = {}
    def extract_time(row): # 提取sepsis发生时间, 返回float
        return min(converter(row['antibiotic_time'], converter(row['culture_time'])))
    def build_dict(row): # 提取sepsis dict
        id = row['subject_id']
        sepsis_time = row['sepsis_time']
        if (id in sepsis_dict and sepsis_dict[id][0] > sepsis_time) or id not in sepsis_dict: # only keep earliest sepsis occurence
                sepsis_dict[id] = [sepsis_time, \
                    int(row['stay_id']), row['sofa_score'], row['respiration'], \
                    row['liver'], row['cardiovascular'], row['cns'], row['renal']]
        
    df = pd.read_csv(csv_path, encoding='utf-8')
    df['sepsis_time'] = df.apply(extract_time, axis=1)
    df.apply(build_dict, axis=1)
    logger.info(f'Load {len(sepsis_dict)} sepsis patients based on sepsis3.csv')
    return sepsis_dict



if __name__ == '__main__':
    dataset = MIMICDataset()
    dataset.mimiciv.make_report()
    