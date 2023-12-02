import os
import tools
import numpy as np
from tools import logger
from tqdm import tqdm
from .helper import Subject, Admission
from collections import namedtuple, Counter
from random import choice
from .mimiciv_core import MIMICIV_Core
import pandas as pd

def load_sepsis_patients(csv_path:str) -> tuple:
    '''
    load extra information of sepsis patients from sepsis3.csv
    sepsis_dict: dict(int(subject_id):list(occur count, elements))
        element: [sepsis_time(float), stay_id, sofa_score, respiration, liver, cardiovascular, cns, renal]
    '''
    converter = tools.TimeConverter(format="%Y-%m-%d %H:%M:%S", out_unit='hour')
    sepsis_dict = {}

    def extract_time(row): # extracts the sepsis occurrence time, returns a float, note that this contains a definition of the sepsis occurrence time
        return min(converter(row['antibiotic_time']), converter(row['culture_time']))
    
    def build_dict(row): # extract sepsis dict
        id = int(row['subject_id'])
        element = {k:row[k] for k in ['sepsis_time', 'stay_id', 'sofa_score', 'respiration', 'liver', 'cardiovascular', 'cns', 'renal']}
        element['stay_id'] = int(element['stay_id'])
        if id in sepsis_dict:
            sepsis_dict[id].append(element)
        else:
            sepsis_dict[id] = [element]
    
    df = pd.read_csv(csv_path, encoding='utf-8')
    df['sepsis_time'] = df.apply(extract_time, axis=1)
    df.apply(build_dict, axis=1)
    logger.info(f'Load {len(sepsis_dict.keys())} sepsis subjects based on sepsis3.csv')
    return set(sepsis_dict.keys()), sepsis_dict

class MIMICIV_ARDS_Dataset(MIMICIV_Core):
    _name = 'mimic-iv-ards'
    
    def __init__(self):
        super().__init__(self.name())

    def on_extract_subjects(self) -> tuple:
        sepsis_patient_path = self._paths['sepsis_patient_path']
        patient_set, extra_data = load_sepsis_patients(sepsis_patient_path)
        return patient_set, extra_data
    
    def on_build_subject(self, subject_id:int, subject:Subject, row:namedtuple, patient_set:set, extra_data:dict) -> Subject:
        '''
        subject: Subject()
        row: dict, {column_name:value}
        extract_value: value of _extract_reuslt[id]
        '''
        '''
            NOTE: How sepsis time is handled
            sepsis time is treated as a static feature added to a subject, a subject can have more than one sepsis time, it is assumed that sepsis time is covered by stay.
            If there is no sepsis time for a submission, then the submission is invalid.
            In the final 3D data, sepsis_time will be the relative value of the distance from the start point t_start (sep-t_start).
            Since the start point is set to max(sepsis, t_start), sepsis_time will only be negative or 0.
            When sepsis_time < 0, that sepsis occurs early, for some models, sepsis time can not be too small, can be used to filter the data
        '''
        ymd_convertor = tools.TimeConverter(format="%Y-%m-%d", out_unit='hour')
        subject.append_static(0, 'age', -1)
        for ele_dict in extra_data[subject_id]: # dict(list(dict))
            sepsis_time = ele_dict['sepsis_time']
            subject.append_static(sepsis_time, 'gender', row.gender)
            if row.dod is not None and isinstance(row.dod, str):
                subject.append_static(sepsis_time, 'dod', ymd_convertor(row.dod))
            else:
                subject.append_static(sepsis_time, 'dod', -1)
            subject.append_static(sepsis_time, 'sepsis_time', sepsis_time)
            subject.append_static(sepsis_time, 'sofa_score', ele_dict['sofa_score'])
            subject.append_static(sepsis_time, 'respiration', ele_dict['respiration'])
            subject.append_static(sepsis_time, 'liver', ele_dict['liver'])
            subject.append_static(sepsis_time, 'cardiovascular', ele_dict['cardiovascular'])
            subject.append_static(sepsis_time, 'cns', ele_dict['cns'])
            subject.append_static(sepsis_time, 'renal', ele_dict['renal'])
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
        # step1: convert static data
        invalid_count = 0
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
                    v,t = static_data[key][idx]
                    try:
                        v = float(v)
                        assert(not np.isnan(v))
                        valid_idx.append(idx)
                    except Exception as e:
                        invalid_count += 1
                if len(valid_idx) == 0: # no valid
                    pop_keys.append(key)
                else:
                    static_data[key] = np.asarray(static_data[key])[valid_idx, :].astype(np.float64)
        for key in pop_keys:
            static_data.pop(key)
        # logger.info(f'Convert Numeric: pop_keys in static data: {pop_keys}')
                
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
                        value = float(value)
                        assert(not np.isnan(value))
                        valid_idx.append(idx)
                    except Exception as e:
                        invalid_count += 1
                
                if len(valid_idx) == 0:
                    pop_keys.append(key)
                elif len(valid_idx) < len(adm.dynamic_data[key]):
                    adm.dynamic_data[key] = np.asarray(adm.dynamic_data[key])[valid_idx, :].astype(np.float64)
                else:
                    adm.dynamic_data[key] = np.asarray(adm.dynamic_data[key]).astype(np.float64)
            for key in pop_keys:
                adm.dynamic_data.pop(key)
        return {'all': {'count': invalid_count, 'examples':set()}}

    def on_select_admissions(self, rule:dict, subjects:dict[int, Subject]):
        invalid_record = {'sepsis_time':0, 'target':0, 'age': 0, 'duration_positive':0, 'duration_limit':0, 'empty':0}
        for s_id in subjects:
            if subjects[s_id].empty():
                invalid_record['empty'] += 1
                continue
            retain_adms = []
            for idx, adm in enumerate(subjects[s_id].admissions):
                age = int(adm.admittime / (24*365) + 1970 - subjects[s_id].birth_year)
                if age <= 18:
                    invalid_record['age'] += 1
                    continue
                flag = 1
                # 检查target
                if 'target_id' in rule and len(rule['target_id']) > 0:
                    for target_id in rule['target_id']:
                        if target_id not in adm.keys():
                            flag = 0
                if flag == 0:
                    invalid_record['target'] += 1
                    continue

                start_time, end_time = None, None
                for id in rule['target_id']:
                    start_time = max(adm[id][0,1], start_time) if start_time is not None else adm[id][0,1]
                    end_time = min(adm[id][-1,1], end_time) if end_time is not None else adm[id][-1,1]
                dur = end_time - start_time
                if dur <= 0:
                    invalid_record['duration_positive'] += 1
                    continue
                if 'duration_minmax' in rule:
                    dur_min, dur_max = rule['duration_minmax']
                    if not (dur > dur_min and dur < dur_max):
                        invalid_record['duration_limit'] += 1
                        continue
                if 'check_sepsis_time' in rule:
                    t_min, t_max = rule['check_sepsis_time']
                    sepsis_time = subjects[s_id].nearest_static('sepsis_time', start_time)
                    time_delta = sepsis_time - (adm.admittime + start_time)
                    if not (time_delta > t_min and time_delta < t_max):
                        invalid_record['sepsis_time'] += 1
                        continue
                    
                retain_adms.append(idx)
            subjects[s_id].admissions = [subjects[s_id].admissions[idx] for idx in retain_adms]

        # delete empty admission and subjects
        pop_list = []
        for s_id in subjects:
            subjects[s_id].del_empty_admission()
            if subjects[s_id].empty():
                pop_list.append(s_id)
                
        for s_id in pop_list:
            subjects.pop(s_id)
        
        logger.info(f'invalid admissions with age <= 18: {invalid_record["age"]}')
        logger.info(f'invalid admissions with no target: {invalid_record["target"]}')
        logger.info(f'invalid admissions exceed sepsis_time limitation: {invalid_record["sepsis_time"]}')
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
        if key == 'sepsis_time': 
            return value - (t_start+admittime) # sepsis time: The datum becomes the start of the table.
        elif key == 'dod':
            if abs(value+1.0)<1e-3:
                return -1
            else:
                delta_year = np.floor(value / (24*365) - ((t_start+admittime) / (24*365))) # After how many years of death, both are timestamp, no need to add 1970
                assert(-100 < delta_year < 100)
                return delta_year
        elif key == 'age':
            age = (admittime + t_start) // (24*365) + 1970 - subject.birth_year # admit time starts from 1970 year
            return age
        else:
            return value
    
    def on_feature_engineering(self, tables:list[np.ndarray], norm_dict:dict, static_keys:list, dynamic_keys):
        addi_dynamic = ['shock_index', 'MAP', 'PPD', 'PF_ratio']
        dynamic_keys += addi_dynamic
        collect_keys = static_keys + dynamic_keys
        index_dict = {key:val for val, key in enumerate(collect_keys)} # used for finding index
        norm_data = []
        for t_idx, table in enumerate(tables):
            addi_table = np.zeros((len(addi_dynamic), table.shape[1]))
            for idx, name in enumerate(addi_dynamic):
                if name == 'PF_ratio':
                    addi_table[idx, :] = np.clip(table[index_dict['220224'], :] / (table[index_dict['223835'], :]*0.01), 0, 500)
                elif name == 'shock_index':
                    if np.all(table[index_dict['systolic pressure'], :] > 0):
                        addi_table[idx, :] = table[index_dict['220045'], :] / table[index_dict['systolic pressure'], :]
                    else:
                        addi_table[idx, :] = -1 # missing value
                        # logger.warning('feature_engineering: skip shock_index with zero sbp')
                elif name == 'MAP':
                    addi_table[idx, :] = (table[index_dict['systolic pressure'], :] + 2*table[index_dict['diastolic pressure'], :]) / 3
                elif name == 'PPD':
                    addi_table[idx, :] = table[index_dict['systolic pressure'], :] - table[index_dict['diastolic pressure'], :]
                else:
                    logger.error(f'Invalid feature name:{name}')
                    assert(0)
            tables[t_idx] = np.concatenate([table, addi_table], axis=0)
            norm_data.append(addi_table)
        # update norm dict for additional features
        norm_data = np.concatenate(norm_data, axis=-1)
        for idx, key in enumerate(addi_dynamic):
            mean, std = np.mean(norm_data[idx, :]), np.std(norm_data[idx, :])
            norm_dict[key] = {'mean': mean, 'std': std}
        # update norm dict for specific features
        for idx, key in enumerate(['dod', 'sepsis_time', 'age']):
            norm_data = np.asarray([table[index_dict[key], 0] for table in tables]).flatten()
            norm_data = norm_data[np.abs(norm_data+1.0)>1e-3] # pick valid data
            norm_dict[key] = {'mean': norm_data.mean(), 'std': norm_data.std()}
        for key in addi_dynamic:
            self._register_item(key, key)
        return {
            'tables': tables,
            'norm_dict': norm_dict,
            'static_keys': static_keys,
            'dynamic_keys': dynamic_keys
        }
    