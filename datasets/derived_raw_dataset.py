import os
import tools
import numpy as np
from tools import logger
from tqdm import tqdm
from .helper import Subject, Admission, load_all_subjects
from collections import namedtuple, Counter
from random import choice
from os.path import join as osjoin
from .mimiciv_core import MIMICIV_Core

class MIMICIV_Raw_Dataset(MIMICIV_Core):
    _name = 'mimic-iv-raw'
    
    def __init__(self):
        super().__init__(self.name())

    def on_extract_subjects(self) -> dict:
        # extract all subjects
        patient_set = load_all_subjects(osjoin(self._mimic_dir, 'hosp', 'patients.csv'))
        return patient_set, None
    
    def on_build_subject(self, subject_id:int, subject:Subject, row:namedtuple, patient_set:set, extra_data:object) -> Subject:
        '''
        subject: Subject()
        row: dict, {column_name:value}
        extract_value: value of _extract_reuslt[id]
        '''
        ymd_convertor = tools.TimeConverter(format="%Y-%m-%d", out_unit='hour')
        subject.append_static(0, 'age', -1)
        subject.append_static(0, 'gender', row.gender)
        if row.dod is not None and isinstance(row.dod, str):
            subject.append_static(ymd_convertor(row.dod), 'dod', ymd_convertor(row.dod))
        return subject
    
    def on_extract_admission(self, subject:Subject, source:str, row:namedtuple) -> bool:
        ymdhms_converter = tools.TimeConverter(format="%Y-%m-%d %H:%M:%S", out_unit='hour')
        if source == 'admission':
            admittime = ymdhms_converter(row.admittime)
            dischtime = ymdhms_converter(row.dischtime)
            if dischtime <= admittime:
                return False
            adm = Admission(
                unique_id=int(row.hadm_id*1e8),
                admittime=admittime, 
                dischtime=dischtime
            )
            discretizer = self._loc_conf['category_to_numeric']
            subject.append_admission(adm)
            for name, val in zip(
                ['insurance', 'language', 'race', 'marital_status'],
                [row.insurance, row.language, row.race, row.marital_status]
            ):
                subject.append_static(admittime, name, discretizer[name][val] if val in discretizer[name] else discretizer[name]['Default'])
            
            subject.append_static(admittime, 'hosp_expire', row.hospital_expire_flag)
            return True
        elif source == 'icu':
            return False
        elif source == 'transfer':
            if not np.isnan(row.hadm_id):
                adm = subject.find_admission(int(row.hadm_id*1e8))
                if adm is not None:
                    discretizer = self._loc_conf['category_to_numeric']
                    careunit = discretizer['careunit'][row.careunit] if row.careunit in discretizer['careunit'] else discretizer['careunit']['Default']
                    adm.append_dynamic('careunit', ymdhms_converter(row.intime), careunit)
            return False
        else:
            assert(0)
            return False
        
    
    def on_select_feature(self, subject_id:int, row:dict, source:str=['icu', 'hosp', 'ed']):
        if source == 'icu':
            if row['type'] in ['Numeric', 'Numeric with tag'] and row['category'] != 'Alarms':
                return True # select
            else:
                return False # not select
        elif source == 'ed':
            return True
        elif source == 'hosp':
            return True
    
    def on_convert_numeric(self, s:Subject) -> Subject:
        '''
        1. 对特定格式的特征进行转换(血压)
        2. 检测不能转换为float的静态特征
        '''
        # step1: convert static data
        invalid_record = {}
        def add_invalid(key, value):
            if key not in invalid_record:
                invalid_record[key] = {'count':1, 'examples':set()}
                invalid_record[key]['examples'].add(value)
            else:
                invalid_record[key]['count'] += 1
                if len(invalid_record[key]['examples']) < 5:
                    invalid_record[key]['examples'].add(value)
        
        static_data = s.static_data
        pop_keys = []
        for key in list(static_data.keys()):
            if key == 'gender':
                # female: 0, male: 1
                static_data[key] = 0 if static_data[key][0][0] == 'F' else 1
            elif 'Blood Pressure' in key:
                static_data['systolic pressure'] = []
                static_data['diastolic pressure'] = []
                for idx in range(len(static_data[key])):
                    p_result = static_data[key][idx][0].split('/')
                    time = static_data[key][idx][1]
                    vs, vd = float(p_result[0]), float(p_result[1])
                    static_data['systolic pressure'].append((vs, time))
                    static_data['diastolic pressure'].append((vd, time))
                static_data.pop(key)
                static_data['systolic pressure'] = np.asarray(static_data['systolic pressure'])
                static_data['diastolic pressure'] = np.asarray(static_data['diastolic pressure'])
            elif isinstance(static_data[key], list):
                valid_idx = []
                for idx in range(len(static_data[key])):
                    value, t = static_data[key][idx]
                    try:
                        v = float(value)
                        assert(not np.isnan(v))
                        valid_idx.append(idx)
                    except Exception as e:
                       add_invalid(key, value)
                if len(valid_idx) == 0: # no valid
                    pop_keys.append(key)
                else:
                    static_data[key] = np.asarray(static_data[key])[valid_idx, :].astype(np.float64)
        for key in pop_keys:
            static_data.pop(key)
                
        s.static_data = static_data
        # step2: convert dynamic data in admissions
        # NOTE: just throw away invalid row
        for adm in s.admissions:
            pop_keys = []
            for key in adm.dynamic_data:
                valid_idx = []
                for idx, row in enumerate(adm.dynamic_data[key]):
                    value = row[0]
                    try:
                        v = float(value)
                        assert(not np.isnan(v))
                        valid_idx.append(idx)
                    except Exception as e:
                        add_invalid(key, value)
                if len(valid_idx) == 0:
                    pop_keys.append(key)
                elif len(valid_idx) < len(adm.dynamic_data[key]):
                    adm.dynamic_data[key] = np.asarray(adm.dynamic_data[key])[valid_idx, :].astype(np.float64) # TODO 这部分代码需要复制给另外两个dataset
                else:
                    adm.dynamic_data[key] = np.asarray(adm.dynamic_data[key]).astype(np.float64)
            for key in pop_keys:
                adm.dynamic_data.pop(key)
        
        return invalid_record
    
    def on_select_admissions(self, rule:dict, subjects:dict[int, Subject]):
        invalid_record = {'age': 0, 'duration_positive':0, 'duration_limit':0, 'empty':0}
        for s_id in subjects:
            if not subjects[s_id].empty():
                retain_adms = []
                for idx, adm in enumerate(subjects[s_id].admissions):
                    age = int(adm.admittime / (24*365) + 1970 - subjects[s_id].birth_year)
                    if age <= 18:
                        invalid_record['age'] += 1
                        continue
                    dur = adm.dischtime - adm.admittime
                    if dur <= 0:
                        invalid_record['duration_positive'] += 1
                        continue
                    if 'duration_minmax' in rule:
                        dur_min, dur_max = rule['duration_minmax']
                        if not (dur > dur_min and dur < dur_max):
                            invalid_record['duration_limit'] += 1
                            continue
                    retain_adms.append(idx)
                subjects[s_id].admissions = [subjects[s_id].admissions[idx] for idx in retain_adms]
            else:
                invalid_record['empty'] += 1
        pop_list = []
        for s_id in subjects:
            subjects[s_id].del_empty_admission()
            if subjects[s_id].empty():
                pop_list.append(s_id)
        for s_id in pop_list:
            subjects.pop(s_id)
        
        logger.info(f'invalid admissions with age <= 19: {invalid_record["age"]}')
        logger.info(f'invalid admissions without positive duration: {invalid_record["duration_positive"]}')
        logger.info(f'invalid admissions exceed duration limitation: {invalid_record["duration_limit"]}')
        logger.info(f'invalid subjects with no admission (empty): {invalid_record["empty"]}')
        logger.info(f'remove_pass1: Deleted {len(pop_list)}/{len(pop_list)+len(subjects)} subjects')

        return subjects
    
    def on_remove_missing_data(self, rule:dict, subjects: dict[int, Subject]) -> dict[int, Subject]:
        n_iter = 0
        while n_iter < len(rule['max_col_missrate']) or (len(post_dynamic_keys)+len(post_static_keys) != len(col_missrate)) or (len(pop_subject_ids) > 0):
            # step1: create column missrate dict
            prior_static_keys = [k for s in subjects.values() for k in s.static_data.keys()]
            prior_dynamic_keys = [k for s in subjects.values() if not s.empty() for k in choice(s.admissions).keys()] # random smapling strategy
            col_missrate = {k:1-v/len(subjects) for key_list in [prior_static_keys, prior_dynamic_keys] for k,v in Counter(key_list).items()}
            # step2: remove invalid columns and admissions
            for s_id, s in subjects.items():
                for adm in s.admissions:
                    pop_keys = [k for k in adm.keys() if k not in col_missrate or col_missrate[k] > rule['max_col_missrate'][min(n_iter, len(rule['max_col_missrate'])-1)]]
                    for key in pop_keys:
                        adm.pop_dynamic(key)
                s.del_empty_admission()
                if len(s.admissions) > 1:
                    retain_adm = np.random.randint(0, len(s.admissions)) if rule['adm_select_strategy'] == 'random' else 0
                    s.admissions = [s.admissions[retain_adm]]
            # step3: create subject missrate dict
            post_static_keys = set([k for s in subjects.values() for k in s.static_data.keys()])
            post_dynamic_keys = set([k for s in subjects.values() if not s.empty() for k in s.admissions[0].keys()])
            subject_missrate = {
                s_id:1 - (len(s.static_data) + len(s.admissions[0]))/(len(post_static_keys)+len(post_dynamic_keys)) \
                    if not s.empty() else 1 for s_id, s in subjects.items()
            }
            # step4: remove invalid subject
            pop_subject_ids = set([s_id for s_id in subjects if subject_missrate[s_id] > rule['max_subject_missrate'][min(n_iter, len(rule['max_subject_missrate'])-1)] or subjects[s_id].empty()])
            for s_id in pop_subject_ids:
                subjects.pop(s_id)
            # step5: calculate removed subjects/columns
            logger.info(f'remove_pass2: iter[{n_iter}] Retain {len(self._subjects)}/{len(pop_subject_ids)+len(self._subjects)} subjects')
            logger.info(f'remove_pass2: iter[{n_iter}] Retain {len(post_dynamic_keys)+len(post_static_keys)}/{len(col_missrate)} keys in selected admission')
            n_iter += 1
        return subjects

    def on_build_table(self, subject:Subject, key, value, t_start):
        admittime = subject.admissions[0].admittime
        if key == 'dod':
            if abs(value+1.0) < 1e-3:
                return -1
            else:
                delta_year = np.floor(value / (24*365) - ((t_start+admittime) / (24*365))) # 经过多少年死亡, 两个都是timestamp，不需要加上1970
                assert(-100 < delta_year < 100)
                return delta_year
        elif key == 'age':
            age = (admittime + t_start) // (24*365) + 1970 - subject.birth_year # admittime从1970年开始计时
            return age
        else:
            return value
    
    def on_feature_engineering(self, tables:list[np.ndarray], norm_dict:dict, static_keys:list, dynamic_keys):
        return {
            'tables': tables,
            'norm_dict': norm_dict,
            'static_keys': static_keys,
            'dynamic_keys': dynamic_keys
        }