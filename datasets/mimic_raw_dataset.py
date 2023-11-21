import os
import tools
import numpy as np
from tools import logger
from tqdm import tqdm
from .mimic_helper import Subject, Admission, load_sepsis_patients
from collections import namedtuple, Counter
from random import choice
from .mimiciv import MIMICIV

class MIMICIV_Raw_Dataset(MIMICIV):
    __name = 'mimic-iv-raw'

    @classmethod
    def name(cls):
        return cls.__name
    
    def __init__(self):
        super().__init__(self.name())

    def on_select_admissions(self, rule:dict, subjects:dict[int, Subject]):
        '''
        remove pass1: 检查是否为空、有无target特征
        '''
        for s_id in subjects:
            if not subjects[s_id].empty():
                retain_adms = []
                for idx, adm in enumerate(subjects[s_id].admissions):
                    flag = 1
                    # 检查target
                    if flag != 0  and 'target_id' in rule and len(rule['target_id']) > 0:
                        for target_id in rule['target_id']:
                            if target_id not in adm.keys():
                                flag = 0
                    if flag != 0:
                        start_time, end_time = None, None
                        for id in rule['target_id']:
                            start_time = max(adm[id][0,1], start_time) if start_time is not None else adm[id][0,1]
                            end_time = min(adm[id][-1,1], end_time) if end_time is not None else adm[id][-1,1]
                        dur = end_time - start_time
                        if dur <= 0:
                            continue
                        if 'duration_minmax' in rule:
                            dur_min, dur_max = rule['duration_minmax']
                            if not (dur > dur_min and dur < dur_max):
                                continue
                    if flag != 0:
                        retain_adms.append(idx)
                subjects[s_id].admissions = [subjects[s_id].admissions[idx] for idx in retain_adms]
       
        # 删除空的admission和subjects
        pop_list = []
        for s_id in subjects:
            subjects[s_id].del_empty_admission() # 删除空的admission
            if subjects[s_id].empty():
                pop_list.append(s_id)
                
        for s_id in pop_list:
            subjects.pop(s_id)
        
        logger.info(f'remove_pass1: Deleted {len(pop_list)}/{len(pop_list)+len(subjects)} subjects')
        return subjects

    def on_remove_missing_data(self, rule:dict, subjects: dict[int, Subject]) -> dict[int, Subject]:
        '''按照传入的配置去除无效特征
        '''
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
        logger.info(f'remove_pass2: selected {len(self._subjects)} subjects')
        return subjects

    def on_extract_subjects(self) -> dict:
        sepsis_patient_path = self._paths['sepsis_patient_path']
        sepsis_result = load_sepsis_patients(sepsis_patient_path)
        return sepsis_result
    
    def on_build_subject(self, id:int, subject:Subject, row:namedtuple, _extract_result:dict) -> Subject:
        '''
        subject: Subject()
        row: dict, {column_name:value}
        extract_value: value of _extract_reuslt[id]
        '''
        '''
            NOTE: sepsis time的处理方式
            sepsis time被看作一个静态特征添加到subject下, 一个subject可以有多个sepsis time, 这里假设sepsis time都被stay覆盖
            如果一个admission没有sepsis time对应, 那么这个admission无效
            在最终的三维数据上, sepsis_time会变为距离起始点t_start的相对值(sep-t_start)
            由于起始点设为max(sepsis, t_start), 所以sepsis_time只会是负数或者0
            当sepsis_time<0的时候, 表明sepsis发生得早, 对于一些模型, sepsis time不能太小, 可以用来筛选数据
        '''
        ymd_convertor = tools.TimeConverter(format="%Y-%m-%d", out_unit='hour')
        subject.append_static(0, 'age', -1)
        for ele_dict in _extract_result[id]: # dict(list(dict))
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
    
    def on_extract_admission(self, subject:Subject, source:str, row:namedtuple):
        ymdhms_converter = tools.TimeConverter(format="%Y-%m-%d %H:%M:%S", out_unit='hour')
        if source == 'admission': # 覆盖最全面，最开始进行筛选
            admittime = ymdhms_converter(row.admittime)
            dischtime = ymdhms_converter(row.dischtime)
            if dischtime < admittime:
                return
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
        elif source == 'icu':
            pass
        elif source == 'transfer':
            if not np.isnan(row.hadm_id):
                adm = subject.find_admission(int(row.hadm_id*1e8))
                if adm is not None:
                    discretizer = self._loc_conf['category_to_numeric']
                    careunit = discretizer['careunit'][row.careunit] if row.careunit in discretizer['careunit'] else discretizer['careunit']['Default']
                    adm.append_dynamic('careunit', ymdhms_converter(row.intime), careunit)
        else:
            assert(0)

    def on_select_feature(self, id:int, row:dict, source:str=['icu', 'hosp', 'ed']):
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
                elif len(valid_idx) < adm.dynamic_data[key].shape[0]:
                    adm.dynamic_data[key] = adm.dynamic_data[key][valid_idx, :].astype(np.float64)
                else:
                    adm.dynamic_data[key] = adm.dynamic_data[key].astype(np.float64)
            for key in pop_keys:
                adm.dynamic_data.pop(key)
        return invalid_count

    def on_build_table(self, subject:Subject, key, value, t_start):
        admittime = subject.admissions[0].admittime
        if key == 'sepsis_time': # sepsis time 基准变为表格的起始点
            return value - (t_start+admittime)
        elif key == 'dod':
            if abs(value+1.0)<1e-3:
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
        '''
        特征工程, 增加某些计算得到的特征
        '''
        addi_dynamic = ['PF_ratio']
        dynamic_keys += addi_dynamic # NOTE: 如果对static keys添加特征，需要重新排序table
        collect_keys = static_keys + dynamic_keys
        index_dict = {key:val for val, key in enumerate(collect_keys)} # used for finding index
        norm_data = []
        for t_idx, table in enumerate(tables):
            addi_table = -np.ones((len(addi_dynamic), table.shape[1]))
            for idx, name in enumerate(addi_dynamic):
                if name == 'PF_ratio':
                    addi_table[idx, :] = table[index_dict['220224'], :] / (np.clip(table[index_dict['223835'], :]*0.01, 0.21, 1.0))
                else:
                    logger.error(f'Invalid feature name:{name}')
                    assert(0)
            self.check_nan(addi_table)
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
        return {
            'tables': tables,
            'norm_dict': norm_dict,
            'static_keys': static_keys,
            'dynamic_keys': dynamic_keys
        }
    
    def make_report(self, version_name, params:dict):
        '''进行数据集的信息统计'''
        # switch version
        self.load_version(version_name)
        self.mode('all')
        out_path = os.path.join(self._paths['out_dir'], f'dataset_report_{version_name}.txt')
        dist_dir = os.path.join(self._paths['out_dir'], 'report_dist')
        dir_names = ['points', 'duration', 'frequency', 'value', 'from_sepsis', 'static_value']
        tools.reinit_dir(dist_dir, build=True)
        for name in dir_names:
            os.makedirs(os.path.join(dist_dir, name))
        logger.info('MIMIC-IV: generating dataset report')
        write_lines = []
        if params['basic']:
            # basic statistics
            write_lines.append('='*10 + 'basic' + '='*10)
            write_lines.append(f'Version: {version_name}')
            write_lines.append(f'Static keys: {len(self.static_keys)}')
            write_lines.append(f'Dynamic keys: {len(self.dynamic_keys)}')
            write_lines.append(f'Subjects:{len(self)}')
            write_lines.append(f'Static feature: {[self.fea_label(id) for id in self.static_keys]}')
            write_lines.append(f'Dynamic feature: {[self.fea_label(id) for id in self.dynamic_keys]}')
        # write report
        with open(out_path, 'w', encoding='utf-8') as fp:
            for line in write_lines:
                fp.write(line + '\n')
        logger.info(f'Report generated at {out_path}')
