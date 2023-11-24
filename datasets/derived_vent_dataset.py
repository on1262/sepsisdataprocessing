import os
import tools
import numpy as np
from tools import logger
from tqdm import tqdm
from .helper import Subject, Admission, load_all_subjects
from collections import namedtuple, Counter
from random import choice
from .mimiciv_core import MIMICIV_Core
from os.path import join as osjoin
import pandas as pd

def load_ventilation_table(ventilation_csv_path:str, icu_stays_csv_path:str) -> dict:
    vent_dict = {}
    query_dict = {}
    # connect stay id to subjects
    icu_stays_table = pd.read_csv(icu_stays_csv_path, encoding='utf-8')
    for row in tqdm(icu_stays_table.itertuples(), 'connect icu stays', total=len(icu_stays_table)):
        s_id, stay_id = row.subject_id, row.stay_id
        query_dict[stay_id] = s_id
    # load ventilation
    vent_table = pd.read_csv(ventilation_csv_path, encoding='utf-8')
    for row in tqdm(vent_table.itertuples(), 'Find all subjects', total=len(vent_table)):
        stay_id = row.stay_id
        subject_id = query_dict[stay_id]
        item = {
            'starttime': row.starttime,
            'endtime': row.endtime,
            'ventilation_status': row.ventilation_status
        }
        if subject_id not in vent_dict:
            vent_dict[subject_id] = [item]
        else:
            vent_dict[subject_id].append(item)
    return vent_dict

class MIMICIV_Vent_Dataset(MIMICIV_Core):
    __name = 'mimic-iv-vent'

    @classmethod
    def name(cls):
        return cls.__name
    
    def __init__(self):
        super().__init__(self.name())

    def on_extract_subjects(self) -> dict:
        # extract all subjects
        patient_set = load_all_subjects(osjoin(self._mimic_dir, 'hosp', 'patients.csv'))
        # load ventilation result
        extra_data = load_ventilation_table(self._paths['ventilation_path'])
        return patient_set, extra_data
    
    def on_build_subject(self, subject_id:int, subject:Subject, row:namedtuple, patient_set:set, extra_data:object) -> Subject:
        '''
        subject: Subject()
        row: dict, {column_name:value}
        extract_value: value of _extract_reuslt[id]
        '''
        ymd_convertor = tools.TimeConverter(format="%Y-%m-%d %H:%M:%S", out_unit='hour')
        subject.append_static(0, 'age', -1)
        subject.append_static(0, 'gender', row.gender)
        if row.dod is not None and isinstance(row.dod, str):
            subject.append_static(0, 'dod', ymd_convertor(row.dod))
        else:
            subject.append_static(0, 'dod', -1)
        if subject_id in extra_data:
            for vent_dict in extra_data[subject_id]:
                vent_start = ymd_convertor(vent_dict['starttime'])
                vent_end = ymd_convertor(vent_dict['endtime'])
                subject.append_static(vent_start, 'ventilation_start', vent_start)
                subject.append_static(vent_start, 'ventilation_num', self._loc_conf['ventilation_to_numeric'][vent_dict['ventilation_status']]) # to be numeric
                subject.append_static(vent_start, 'ventilation_end', vent_end) # the time of 'end time' is just a key for search
        return subject
    
    def on_extract_admission(self, subject:Subject, source:str, row:namedtuple):
        ymdhms_converter = tools.TimeConverter(format="%Y-%m-%d %H:%M:%S", out_unit='hour')
        if source == 'admission':
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
                elif len(valid_idx) < adm.dynamic_data[key].shape[0]:
                    adm.dynamic_data[key] = adm.dynamic_data[key][valid_idx, :].astype(np.float64)
                else:
                    adm.dynamic_data[key] = adm.dynamic_data[key].astype(np.float64)
            for key in pop_keys:
                adm.dynamic_data.pop(key)
        return invalid_count

    def on_select_admissions(self, rule:dict, subjects:dict[int, Subject]):
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
       
        pop_list = []
        for s_id in subjects:
            subjects[s_id].del_empty_admission()
            if subjects[s_id].empty():
                pop_list.append(s_id)
                
        for s_id in pop_list:
            subjects.pop(s_id)
        
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
        logger.info(f'remove_pass2: selected {len(self._subjects)} subjects')
        return subjects

    def on_build_table(self, subject:Subject, key, value, t_start):
        admittime = subject.admissions[0].admittime
        if key in ['ventilation_start', 'ventilation_end']:
            return value - t_start # adjust time
        if key == 'dod':
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
        addi_dynamic = ['vent_status']
        vent_end_idx = static_keys.index('ventilation_end')
        vent_num_idx = static_keys.index('ventilation_num')

        dynamic_keys += addi_dynamic # NOTE: If you add features to static keys, you need to reorder the table.
        collect_keys = static_keys + dynamic_keys
        index_dict = {key:val for val, key in enumerate(collect_keys)} # used for finding index
        norm_data = []
        for t_idx, table in enumerate(tables):
            addi_table = np.zeros((len(addi_dynamic), table.shape[1])) # default: no ventilation
            vent_num = table[vent_num_idx, :]
            vent_end = table[vent_end_idx, :]
            ticks = np.linspace(0, table.shape[1]*self._loc_conf['generate_table']['delta_t_hour'], table.shape[1])
            addi_table[0, :] = vent_num * (vent_end <= ticks) # we assume that in no ventilation for a gap between last ventilation end and next ventilation start
            tables[t_idx] = np.concatenate([table, addi_table], axis=0)
            norm_data.append(addi_table)
        # update norm dict for additional features
        norm_data = np.concatenate(norm_data, axis=-1)
        for idx, key in enumerate(addi_dynamic):
            mean, std = np.mean(norm_data[idx, :]), np.std(norm_data[idx, :])
            norm_dict[key] = {'mean': mean, 'std': std}
        # update norm dict for specific features
        for idx, key in enumerate(['dod', 'age']):
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
        logger.info('MIMIC-IV-ARDS: generating dataset report')
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
        if params['dynamic_dist']:
            # dynamic feature explore
            for id in tqdm(self.dynamic_keys, 'plot dynamic dist'):
                fea_name = self.fea_label(id)
                save_name = tools.remove_slash(str(fea_name))
                write_lines.append('='*10 + f'{fea_name}({id})' + '='*10)
                arr_points = []
                arr_duration = []
                arr_frequency = []
                arr_avg_value = []
                arr_from_sepsis_time = []
                for s in self._subjects.values():
                    for adm in s.admissions:
                        if id in adm.keys():
                            dur = adm[id][-1,1] - adm[id][0,1]
                            sepsis_time = s.latest_static('sepsis_time', adm[id][0, 1])
                            t_sep = adm[id][0, 1] + adm.admittime - sepsis_time
                            arr_from_sepsis_time.append(t_sep)
                            arr_points.append(adm[id].shape[0])
                            arr_duration.append(dur)
                            if dur > 1e-3: # TODO 只有一个点无法计算
                                arr_frequency.append(arr_points[-1] / arr_duration[-1])
                            else:
                                arr_frequency.append(0)
                            arr_avg_value.append(adm[id][:,0].mean())
                arr_points, arr_duration, arr_frequency, arr_avg_value = \
                    np.asarray(arr_points), np.asarray(arr_duration), np.asarray(arr_frequency), np.asarray(arr_avg_value)
                arr_from_sepsis_time = np.asarray(arr_from_sepsis_time)
                if np.size(arr_points) != 0:
                    write_lines.append(f'average points per admission: {arr_points.mean():.3f}')
                if np.size(arr_duration) != 0:
                    write_lines.append(f'average duration(hour) per admission: {arr_duration.mean():.3f}')
                if np.size(arr_frequency) != 0:
                    write_lines.append(f'average frequency(point/hour) per admission: {arr_frequency.mean():.3f}')
                if np.size(arr_avg_value) != 0:
                    write_lines.append(f'average avg value per admission: {arr_avg_value.mean():.3f}')
                # plot distribution
                titles = ['points', 'duration', 'frequency', 'value', 'from_sepsis']
                arrs = [arr_points, arr_duration, arr_frequency, arr_avg_value, arr_from_sepsis_time]
                for title, arr in zip(titles, arrs):
                    if np.size(arr) != 0:
                        tools.plot_single_dist(
                            data=arr, data_name=f'{title}: {fea_name}', 
                            save_path=os.path.join(dist_dir, title, save_name + '.png'), discrete=False, adapt=True, bins=50)
        if params['static_dist']:
            # static feature explore
            for id in tqdm(self.static_keys, 'generate static feature report'):
                fea_name = self.fea_label(id)
                save_name = tools.remove_slash(str(fea_name))
                write_lines.append('='*10 + f'{fea_name}({id})' + '='*10)
                idx = self.idx_dict[str(id)]
                static_data = self.data[:, idx, 0]
                write_lines.append(f'mean: {static_data.mean():.3f}')
                write_lines.append(f'std: {static_data.std():.3f}')
                write_lines.append(f'max: {np.max(static_data):.3f}')
                write_lines.append(f'min: {np.min(static_data):.3f}')
                tools.plot_single_dist(
                    data=static_data, data_name=f'{fea_name}', 
                    save_path=os.path.join(dist_dir, 'static_value', save_name + '.png'), discrete=False, adapt=True, bins=50)
        # write report
        with open(out_path, 'w', encoding='utf-8') as fp:
            for line in write_lines:
                fp.write(line + '\n')
        logger.info(f'Report generated at {out_path}')