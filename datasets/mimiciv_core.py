import os
import tools
import compress_pickle as pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tools import GLOBAL_CONF_LOADER
from tools import logger
from tqdm import tqdm
from sklearn.model_selection import KFold
from abc import abstractmethod
from datetime import datetime
from random import choice
from .helper import Subject, KFoldIterator, interp

class MIMICIV_Core(Dataset):
    _name = 'mimic-iv-core'

    def __init__(self, dataset_name):
        super().__init__()
        # configs
        self._paths = GLOBAL_CONF_LOADER['paths'][dataset_name]
        self._mimic_dir:str = self._paths['mimic_dir']
        self._loc_conf = tools.Config(self._paths['conf_manual_path'])
        
        # variable for phase 1
        self._extract_result:dict = None
        self._hosp_item:dict = None
        self._icu_item:dict = None
        # variable for phase 2
        self._subjects:dict[int, Subject] = {} # subject_id:Subject
        # preload data
        self._data:np.ndarray = None # to derive other versions
        self._norm_dict:dict = None # key=str(name/id) value={'mean':mean, 'std':std}
        self._static_keys:list[str] = None # list(str)
        self._dynamic_keys:list[str] = None # list(str)
        self._total_keys:list[str] = None
        self._seqs_len:list = None # list(available_len)
        self._idx_dict:dict[int, str] = None
        # mode switch
        self._now_mode:str = None # 'train'/'valid'/'test'/'all'
        self._kf_list:list[dict] = None # list([train_index, valid_index, test_index])
        self._kf_index:int = None # 第几个fold
        self._data_index:list[int] = None # 当前模式下的索引
        # version switch
        self._version_name = None

        self._load_data()

        logger.info(f'{self.name()} inited')

    # read-only access control
    @property
    def norm_dict(self):
        return self._norm_dict
    
    @property
    def idx_dict(self):
        return self._idx_dict
    
    @property
    def total_keys(self):
        return self._total_keys
    
    @property
    def seqs_len(self):
        return self._seqs_len
    
    @property
    def version_name(self):
        return self._version_name
    
    @property
    def static_keys(self):
        return self._static_keys
    
    @property
    def dynamic_keys(self):
        return self._dynamic_keys
    
    @property
    def data(self):
        return self._data

    @property
    def subjects(self):
        return self._subjects

    @classmethod
    def name(cls):
        return cls._name
    
    def check_nan(self, x:np.ndarray):
        assert(not np.any(np.isnan(x)))
    
    def _load_data(self):
        '''Basic load manager. Eliminate unnecessary IO consumption.
        '''
        cache_dir = self._paths['cache_dir']
        if not os.path.exists(cache_dir):
            tools.reinit_dir(cache_dir, build=True)
        suffix = '.pkl' if not self._loc_conf['compress_cache'] else self._loc_conf['compress_suffix']
        
        # create pkl names for each phase
        pkl_paths = [os.path.join(cache_dir, name + suffix) for name in ['1_phase1', '2_phase2', '3_subjects', '4_numeric_subject', '5_norm_dict', '6_table_final']]
        version_files = [os.path.join(cache_dir, f'7_version_{version_name}' + suffix) for version_name in self._loc_conf['version'].keys()] + \
                [os.path.join(cache_dir, f'7_version_{version_name}.npz') for version_name in self._loc_conf['version'].keys()]
        bare_mode = np.all([os.path.exists(p) for p in pkl_paths] + [os.path.exists(p) for p in version_files]) # speed up IO only when all cache files exist
        
        if bare_mode:
            logger.info('MIMIC-IV: Bare Mode Enabled')
            self._preprocess_phase1(pkl_paths[0], dim_files_only=True) # load dim files
            self._preprocess_phase5(pkl_paths[4], load_subject_only=True) # load subjects
        else:
            logger.info('MIMIC-IV: Bare Mode Disabled')
            log_file = os.path.join(self._paths['out_dir'], 'build_dataset.log')
            logger.info('Start logging to ' + log_file)
            tools.reinit_dir(self._paths['out_dir'], build=True) # add logger
            log_id = logger.add(log_file)
            for phase in range(1, 7):
                func = getattr(self, '_preprocess_phase' + str(phase))
                func(pkl_paths[phase-1])
            self._preprocess_phase7()
            logger.remove(log_id)
 
    def _preprocess_phase1(self, pkl_path=None, dim_files_only=False):
        if os.path.exists(pkl_path):
            logger.info(f'load cache for phase 1 from {pkl_path}')
            with open(pkl_path, 'rb') as fp:
                load_dict = pickle.load(fp)
                if not dim_files_only:
                    self._extract_result = load_dict['extract_result']
                self._icu_item = load_dict['icu_item']
                self._hosp_item = load_dict['hosp_item']
                self._ed_item = load_dict['ed_item']
                self._all_items = load_dict['d_item']
            return
    
        logger.info(f'MIMIC-IV: extract dim file')
        
        # 建立hospital lab_item编号映射
        d_hosp_item = pd.read_csv(os.path.join(self._mimic_dir, 'hosp', 'd_labitems.csv'), encoding='utf-8')
        hosp_item = {'id':{}, 'label':{}}
        for _, row in tqdm(d_hosp_item.iterrows(), desc='hosp items'):
            hosp_item['id'][str(row['itemid'])] = {
                'id': str(row['itemid']),
                'label': row['label'], 
                'fluid': row['fluid'], 
                'category': row['category']
            }
            hosp_item['label'][row['label']] = hosp_item['id'][str(row['itemid'])]
        
        # 建立icu lab_item编号映射
        d_icu_item = pd.read_csv(os.path.join(self._mimic_dir, 'icu', 'd_items.csv'), encoding='utf-8')
        icu_item = {'id': {}, 'label': {}}
        for _, row in tqdm(d_icu_item.iterrows(), desc='icu items'):
            icu_item['id'][str(row['itemid'])] = {
                'id': str(row['itemid']),
                'label': row['label'], 
                'category': row['category'], 
                'type': row['param_type'], 
                'low': row['lownormalvalue'], 
                'high': row['highnormalvalue']
            }
            icu_item['label'][row['label']] = icu_item['id'][str(row['itemid'])]

        # 建立ed item的编号映射, 和icu lab_item关联
        ed_item = {'id': {
            'ED_temperature': {'id': 'ED_temperature', 'link_id':'223761', 'label':'(ED) Temperature Fahrenheit'}, 
            'ED_heartrate': {'id': 'ED_heartrate', 'link_id':'220045', 'label':'(ED) Heart Rate'},
            'ED_resprate': {'id': 'ED_resprate', 'link_id':'220210', 'label':'(ED) Respiratory Rate'},
            'ED_o2sat': {'id': 'ED_o2sat', 'link_id':'220277', 'label':'(ED) O2 saturation pulseoxymetry'},
            'ED_sbp': {'id': 'ED_sbp', 'link_id': None, 'label':'(ED) sbp'},
            'ED_dbp': {'id': 'ED_dbp', 'link_id': None, 'label':'(ED) dbp'}
        }}
        ed_item['label'] = {val['label']:val for val in ed_item['id'].values()}

        subject_set, extra_information = self.on_extract_subjects()
        logger.info(f'on_extract_subjects: Loaded {len(subject_set)} unique patients')

        self._hosp_item = hosp_item
        self._icu_item = icu_item
        self._ed_item = ed_item
        self._all_items = {
            'id': {key:val for d in [ed_item, icu_item, hosp_item] for key, val in d['id'].items()},
            'label': {key:val for d in [ed_item, icu_item, hosp_item] for key, val in d['label'].items()}
        }
        self._extract_result = (subject_set, extra_information)
        with open(pkl_path, 'wb') as fp:
            pickle.dump({
                'extract_result': self._extract_result,
                'icu_item': icu_item,
                'hosp_item': self._hosp_item,
                'ed_item': self._ed_item,
                'd_item': self._all_items
                }, fp)
    
    def _preprocess_phase2(self, pkl_path=None):
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as fp:
                load_dict = pickle.load(fp)
                self._subjects = load_dict['subjects']
            logger.info(f'load cache for phase 2 from {pkl_path}')
            return

        logger.info(f'MIMIC-IV: processing subjects and admissions')

        # construct subject
        patients = pd.read_csv(os.path.join(self._mimic_dir, 'hosp', 'patients.csv'), encoding='utf-8')
        for row in tqdm(patients.itertuples(), 'Construct Subjects', total=len(patients), miniters=len(patients)//100):
            s_id = row.subject_id
            if s_id in self._extract_result[0]:
                subject = Subject(row.subject_id, birth_year=row.anchor_year - row.anchor_age)
                self._subjects[s_id] = self.on_build_subject(s_id, subject, row, self._extract_result[0], self._extract_result[1])

        logger.info(f'Extract {len(self._subjects)} patients from {len(self._extract_result[0])} patients in extract_result')

        # add admission
        admissions = pd.read_csv(os.path.join(self._mimic_dir, 'hosp', 'admissions.csv'), encoding='utf-8')
        _total_adm, _extract_adm = 0, 0
        for row in tqdm(admissions.itertuples(), 'Extract Admissions', total=len(admissions), miniters=len(admissions)//100):
            if row.subject_id in self._subjects:
                _total_adm += 1
                _extract_adm += self.on_extract_admission(self._subjects[row.subject_id], source='admission', row=row)
        logger.info(f'Extracted {_extract_adm} admissions from {_total_adm} admissions which belongs to {len(self._subjects)} subjects')

        for path, prefix in zip(
            [os.path.join(self._mimic_dir, 'icu', 'icustays.csv'), os.path.join(self._mimic_dir, 'hosp', 'transfers.csv')], 
            ['icu', 'transfer']
        ):
            table = pd.read_csv(path, encoding='utf-8')
            for row in tqdm(table.itertuples(), desc=f'Extract admission information from {prefix.upper()}', total=len(table), miniters=len(table)//100):
                s_id = row.subject_id
                if s_id in self._subjects:
                    self.on_extract_admission(self._subjects[s_id], source=prefix, row=row)
            del table

        # 患者的基本信息，如身高、体重、血压
        ymd_convertor = tools.TimeConverter(format="%Y-%m-%d", out_unit='hour')
        table_omr = pd.read_csv(os.path.join(self._mimic_dir, 'hosp', 'omr.csv'), encoding='utf-8') 
        # omr: [subject_id,chartdate,seq_num,result_name,result_value]
        for row in tqdm(table_omr.itertuples(), 'Extract patient information from OMR', total=len(table_omr), miniters=len(table_omr)//100):
            s_id = row.subject_id
            if s_id in self._subjects:
                self._subjects[s_id].append_static(ymd_convertor(row.chartdate), row.result_name, row.result_value)
        
        # dump
        with open(pkl_path, 'wb') as fp:
            pickle.dump({'subjects': self._subjects}, fp)
        logger.info(f'Phase 2 dumped at {pkl_path}')

    def _preprocess_phase3(self, pkl_path=None):
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as fp:
                self._subjects = pickle.load(fp)
            logger.info(f'load cache for phase 3 from {pkl_path}')
            return
        logger.info(f'MIMIC-IV: processing dynamic data')
        ymdhms_converter = tools.TimeConverter(format="%Y-%m-%d %H:%M:%S", out_unit='hour')

        # 决定捕捉哪些特征
        collect_icu_set = set([id for id, row in self._icu_item['id'].items() if self.on_select_feature(subject_id=id, row=row, source='icu')])
        collect_ed_set = set([id for id in self._ed_item['id'].keys() if self.on_select_feature(subject_id=id, row=None, source='ed')])
        collect_hosp_set = set([id for id, row in self._hosp_item['id'].items() if self.on_select_feature(subject_id=id, row=row, source='hosp')])
        
        if self._loc_conf['data_linkage']['ed']:
            # 采集ED内的数据
            ed_vitalsign = pd.read_csv(os.path.join(self._mimic_dir, 'ed', 'vitalsign.csv'), 
                                       dtype={'subject_id':int, 'charttime':str, 'temperature':str, "heartrate":str ,"resprate":str ,"o2sat":str, 'sbp':str, 'dbp':str, 'rhythm':str, 'pain':str}, 
                                       encoding='utf-8'
            )
            for row in tqdm(ed_vitalsign.itertuples(), 'Extract vitalsign from MIMIC-IV-ED', total=len(ed_vitalsign), miniters=len(ed_vitalsign)//100):
                s_id = row.subject_id
                for itemid in ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'rhythm', 'pain']:
                    if s_id in self._subjects and 'ED_'+itemid in collect_ed_set:
                        self._subjects[s_id].append_dynamic(
                            charttime=ymdhms_converter(row.charttime),
                            itemid='ED_'+itemid, 
                            value=getattr(row, itemid)
                        )
            del ed_vitalsign

        if self._loc_conf['data_linkage']['hosp']:
            total_size = 10000000 * 12
            hosp_chunksize = 10000000
            hosp_labevents = pd.read_csv(
                os.path.join(self._mimic_dir, 'hosp', 'labevents.csv'), encoding='utf-8', chunksize=hosp_chunksize,
                usecols=['subject_id', 'itemid', 'charttime', 'value', 'valuenum'], engine='c',
                dtype={'subject_id':int, 'itemid':str, 'charttime':str, 'value':str, 'valuenum':str},
                na_filter=False
            )
            for chunk_idx, chunk in tqdm(
                enumerate(hosp_labevents), 'Extract labevent from hosp', 
                total=total_size//hosp_chunksize, 
                miniters=total_size//hosp_chunksize//100
            ):
                for row in tqdm(chunk.itertuples(), f'chunk {chunk_idx}', miniters=len(chunk)//10): # NOTE: reserve dtypes for valuenum
                    s_id, itemid, charttime, valuenum, value = row.subject_id, row.itemid, row.charttime, row.valuenum, row.value
                    if s_id in self._subjects and itemid in collect_hosp_set:
                        charttime = datetime.fromisoformat(charttime).timestamp() / 3600.0 # hour
                        if valuenum == '':
                            valuenum = value
                        self._subjects[s_id].append_dynamic(charttime=charttime, itemid=itemid, value=valuenum)
            del hosp_labevents

        if self._loc_conf['data_linkage']['icu']:
            # 采集icu内的动态数据
            total_size = 10000000 * 32
            icu_events_chunksize = 10000000
            icu_events = pd.read_csv(os.path.join(self._mimic_dir, 'icu', 'chartevents.csv'), 
                    encoding='utf-8', usecols=['subject_id', 'charttime', 'itemid', 'valuenum'], chunksize=icu_events_chunksize, engine='c',
                    dtype={'subject_id':int, 'itemid':str, 'charttime':str, 'valuenum':str}
            )
            
            for chunk_idx, chunk in tqdm(
                enumerate(icu_events), 
                'Extract ICU events', 
                total=total_size // icu_events_chunksize,
                miniters=total_size // icu_events_chunksize // 100
            ):
                for row in tqdm(chunk.itertuples(), f'chunk {chunk_idx}', miniters=len(chunk)//10):
                    s_id, charttime, itemid, valuenum = row.subject_id, row.charttime, row.itemid, row.valuenum # 要和文件头保持相同的顺序
                    if s_id in self._subjects and itemid in collect_icu_set: # do not double check subject id
                        charttime = datetime.fromisoformat(charttime).timestamp() / 3600.0 # hour
                        self._subjects[s_id].append_dynamic(charttime=charttime, itemid=itemid, value=valuenum)
            del icu_events
        
        # 保存subjects
        logger.info('Dump subjects')
        with open(pkl_path, 'wb') as fp:
            pickle.dump(self._subjects, fp)
        logger.info('Dump subjects: Done')

    def _preprocess_phase4(self, p_numeric_subject=None):
        if os.path.exists(p_numeric_subject):
            with open(p_numeric_subject, 'rb') as fp:
                self._subjects = pickle.load(fp)
            logger.info(f'Load numeric subject data from {p_numeric_subject}')
            return

        invalid_record = {}
        for s in tqdm(self._subjects.values(), 'Convert to numeric', miniters=len(self._subjects)//100):
            r = self.on_convert_numeric(s)
            for k,v in r.items():
                if k not in invalid_record:
                    invalid_record[k] = v
                else:
                    invalid_record[k]['count'] += v['count']
                    if len(invalid_record[k]['examples']) < 5:
                        invalid_record[k]['examples'].union(v['examples'])
        for key in invalid_record:
            count, examples = invalid_record[key]['count'], invalid_record[key]['examples']
            logger.debug(f'Invalid: key={key}, count={count}, example={examples}')
        invalid_count = sum([val['count'] for val in invalid_record.values()])
        logger.warning(f'Convert to numeric: find {invalid_count} invalid values in dynamic data')
        
        for s_id in tqdm(self._subjects, desc='update data', miniters=len(self._subjects)//100):
            self._subjects[s_id].update_data()

        self._subjects:dict = self.on_select_admissions(rule=self._loc_conf['remove_rule']['pass1'], subjects=self._subjects)

        logger.info(f'Retain {len(self._subjects)} subjects and {np.sum([len(s.admissions) for s in self._subjects.values()])} admissions')

        value_clip = self._loc_conf['value_clip']
        for id_or_label in value_clip:
            id, label = self.fea_id(id_or_label), self.fea_label(id_or_label)
            clip_count = 0
            for s in self._subjects.values():
                for adm in s.admissions:
                    if id in adm.keys():
                        data = adm[id][:, 0]
                        adm[id][:, 0] = np.clip(data, a_min=value_clip[id_or_label]['min'], a_max=value_clip[id_or_label]['max'])
                        clip_count += 1
            logger.info(f'Value Clipping: clip {label} in {clip_count} admissions')
        
        with open(p_numeric_subject, 'wb') as fp:
            pickle.dump(self._subjects, fp)
        logger.info(f'Numeric subjects dumped at {p_numeric_subject}')
    
    def _verify_subjects(self, subjects:dict[int, Subject]):
        try:
            for s in tqdm(subjects.values(), 'verify subjects', miniters=len(subjects)//100):
                adm = s.admissions[0]
                for v in s.static_data.values():
                    self.check_nan(v)
                for v in adm.dynamic_data.values():
                    self.check_nan(v)
        except Exception as e:
            logger.error('Verify nan failed, error trace:')
            print(e)
            
    def _preprocess_phase5(self, p_norm_dict, load_subject_only=False):
        if os.path.exists(p_norm_dict):
            with open(p_norm_dict, 'rb') as fp:
                result = pickle.load(fp)
                if not load_subject_only:
                    self._dynamic_keys, self._static_keys, self._subjects, self._norm_dict = \
                        result['dynamic_keys'], result['static_keys'], result['subjects'], result['norm_dict']
                else:
                    self._subjects = result['subjects']
            logger.info(f'Load norm dict from {p_norm_dict}')
            return

        # 进一步筛选admission
        self._subjects = self.on_remove_missing_data(self._loc_conf['remove_rule']['pass2'], self._subjects)
        logger.info(f'After remove missing data, retain {len(self._subjects)} subjects and {np.sum([len(s.admissions) for s in self._subjects.values()])} admissions')

        self._verify_subjects(self.subjects)
        # determine static/dyanmic features
        self._static_keys = sorted(np.unique([k for s in self._subjects.values() for k in s.static_data.keys()]))
        self._dynamic_keys = sorted(np.unique([k for s in self._subjects.values() for k in s.admissions[0].keys()]))

        logger.info(f'Static keys: {[self.fea_label(key) for key in self._static_keys]}')
        logger.info(f'Dynamic keys: {[self.fea_label(key) for key in self._dynamic_keys]}')

        norm_dict = {}
        for s in self._subjects.values():
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
            if isinstance(norm_dict[key][0], (float, int)): # no time stamp
                norm_dict[key] = np.asarray(norm_dict[key])
            elif isinstance(norm_dict[key][0], np.ndarray):
                norm_dict[key] = np.concatenate(norm_dict[key], axis=0)[:, 0]
            else:
                assert(0)
        
        for key in norm_dict:
            mean, std = np.mean(norm_dict[key]), np.std(norm_dict[key])
            norm_dict[key] = {'mean':mean, 'std':std}
        
        self._norm_dict = norm_dict
        with open(p_norm_dict, 'wb') as fp:
            pickle.dump({
                'subjects': self._subjects,
                'norm_dict': self._norm_dict,
                'static_keys': self._static_keys,
                'dynamic_keys': self._dynamic_keys
            }, fp)
        logger.info(f'Norm dict dumped at {p_norm_dict}')

    def _preprocess_phase6(self, p_final_table):
        '''
        对每个subject生成时间轴对齐的表, tick(hour)是生成的表的间隔
        '''

        if os.path.exists(p_final_table):
            with open(p_final_table, 'rb') as fp:
                result = pickle.load(fp)
                self._data, self._norm_dict, self._idx_dict, self._seqs_len, self._static_keys, self._dynamic_keys = \
                    result['data'], result['norm_dict'], result['index_dict'], result['seqs_len'], result['static_keys'], result['dynamic_keys']
            logger.info(f'load aligned table from {p_final_table}')
            return
        
        # step1: 插值并生成表格
        tables:list[np.ndarray] = [] # table for all subjects
        collect_keys = set(self._static_keys).union(set(self._dynamic_keys))
        logger.info(f'Detected {len(self._dynamic_keys)} available dynamic features')
        logger.info(f'Detected {len(self._static_keys)} available static features')

        default_missvalue = float(self._loc_conf['generate_table']['default_missing_value'])
        calculate_bin = self._loc_conf['generate_table']['calculate_bin']
        invalid_record = {'length':0}
        for s_id in tqdm(self._subjects.keys(), desc='Generate aligned table', miniters=len(self._subjects)//100):
            s = self._subjects[s_id]
            adm = s.admissions[0]
            
            t_start, t_end = None, None
            if len(self._loc_conf['generate_table']['align_target']) == 0: # no target
                for id in adm.keys():
                    t_start = min(adm[id][0,1], t_start) if t_start is not None else adm[id][0,1]
                    t_end = max(adm[id][-1,1], t_end) if t_end is not None else adm[id][-1,1]
                t_start, t_end = max(t_start, 0), min(t_end, adm.dischtime - adm.admittime)
            else:
                for id in self._loc_conf['generate_table']['align_target']:
                    t_start = max(adm[id][0,1], t_start) if t_start is not None else adm[id][0,1]
                    t_end = min(adm[id][-1,1], t_end) if t_end is not None else adm[id][-1,1]
            t_step = self._loc_conf['generate_table']['delta_t_hour']
            if t_end - t_start < t_step: # NOTE: e.g. all features are in the same tick. We can not made prediction in such cases.
                invalid_record['length'] += 1
                continue

            ticks = np.arange(t_start, t_end, t_step) # 最后一个会确保间隔不变且小于t_end
            # 生成表本身, 缺失值为-1
            individual_table = np.ones((len(collect_keys), ticks.shape[0]), dtype=np.float32) * default_missvalue

            # fulfill static data by nearest value
            static_data = np.ones((len(self._static_keys), ticks.shape[0])) * default_missvalue
            for t_idx, t in enumerate(ticks):
                for idx, key in enumerate(self._static_keys):
                    if key in self._subjects[s_id].static_data:
                        value = self._subjects[s_id].latest_static(key, t+adm.admittime)
                        if value is not None:
                            static_data[idx, t_idx] = self.on_build_table(self._subjects[s_id], key, value, t_start)
                
            individual_table[:len(self._static_keys), :] = static_data
            self.check_nan(individual_table)
            # interpolation of dynamic data
            for idx, key in enumerate(self._dynamic_keys):
                if key in adm.keys() and key not in ['careunit']:
                    self.check_nan(adm[key])
                    if adm[key].shape[0] == 1:
                        individual_table[len(self._static_keys)+idx, :] = adm[key][0, 0]
                    else:
                        individual_table[len(self._static_keys)+idx, :] = interp(
                            fx=adm[key][:, 1], fy=adm[key][:, 0], x_start=ticks[0], 
                            interval=t_step, n_bins=len(ticks), missing=default_missvalue, fill_bin=calculate_bin
                        )
                    self.check_nan(individual_table)
            if individual_table.size > 0:
                self.check_nan(individual_table)
                tables.append(individual_table)
        logger.info(f'Invalid table due to too short timestep: {invalid_record["length"]}')
        logger.info(f'Generated {len(tables)} individual tables. ')
        result = self.on_feature_engineering(tables, self._norm_dict, self._static_keys, self._dynamic_keys) # 特征工程
        tables, self._norm_dict, static_keys, dynamic_keys = result['tables'], result['norm_dict'], result['static_keys'], result['dynamic_keys']
        for table in tables:
            self.check_nan(table)
        total_keys = static_keys + dynamic_keys
        index_dict = {key:val for val, key in enumerate(total_keys)} # used for finding index
        
        # step2: 时间轴长度对齐, 生成seqs_len, 进行某些特征的最后处理
        seqs_len = np.asarray([d.shape[1] for d in tables], dtype=np.int64)
        max_len = max(seqs_len)
        for t_idx in tqdm(range(len(tables)), desc='Padding tables', miniters=len(tables)//100):
            if seqs_len[t_idx] == max_len:
                continue
            padding = -np.ones((len(total_keys), max_len - seqs_len[t_idx]))
            tables[t_idx] = np.concatenate([tables[t_idx], padding], axis=1)
        tables = np.stack(tables, axis=0) # (n_sample, n_fea, seqs_len)
        # verify tables
        self.check_nan(tables)
        self._data = tables
        self._idx_dict = index_dict
        self._seqs_len = seqs_len
        self._static_keys, self._dynamic_keys = static_keys, dynamic_keys  # 特征工程会新增一些key
        with open(p_final_table, 'wb') as fp:
            pickle.dump({
                'data': tables,
                'index_dict': index_dict,
                'norm_dict': self._norm_dict,
                'seqs_len': seqs_len,
                'static_keys': self._static_keys,
                'dynamic_keys': self._dynamic_keys,
                'all_item': self._all_items
            }, fp)
        logger.info(f'Aligned table dumped at {p_final_table}')

    def _preprocess_phase7(self):
        '''生成不同版本的数据集, 不同版本的数据集的样本数量/特征数量都可能不同
        '''
        assert(self._idx_dict is not None)
        version_conf:dict = self._loc_conf['version']
        suffix = '.pkl' if not self._loc_conf['compress_cache'] else self._loc_conf['compress_suffix']

        for version_name in version_conf.keys():
            logger.info(f'Preprocessing version: {version_name}')
            # 检查是否存在pkl
            p_version = os.path.join(self._paths['cache_dir'], f'7_version_{version_name}'+suffix)
            p_version_data = os.path.join(self._paths['cache_dir'], f'7_version_{version_name}.npz')
            if os.path.exists(p_version):
                logger.info(f'Skip preprocess existed version: {version_name}')
                continue
            derived_data_table = self._data.copy() # 深拷贝
            
            # 筛选可用特征
            if len(version_conf[version_name]['feature_limit']) > 0:
                limit_idx = [self._idx_dict[lfea] for lfea in version_conf[version_name]['feature_limit'] if lfea in self._idx_dict]
            else:
                limit_idx = list(self._idx_dict.values())
            
            total_keys = self.static_keys + self.dynamic_keys
            forbidden_idx = set([self.fea_idx(ffea) for ffea in version_conf[version_name]['forbidden_feas'] if ffea in self._idx_dict])
            avail_static_idx = [idx for idx in limit_idx if idx not in forbidden_idx and total_keys[idx] in self.static_keys]
            avail_dynamic_idx = [idx for idx in limit_idx if idx not in forbidden_idx and total_keys[idx] in self.dynamic_keys]
            avail_idx = avail_static_idx + avail_dynamic_idx
            # 特征顺序改变，相关的key list都要修改
            derived_static_keys = [total_keys[idx] for idx in avail_static_idx]
            derived_dynamic_keys = [total_keys[idx] for idx in avail_dynamic_idx]
            derived_total_keys = [total_keys[idx] for idx in avail_idx]
            derived_idx_dict = {key:idx for idx, key in enumerate(derived_total_keys)}
            derived_norm_dict = {key:val for key,val in self._norm_dict.items() if key in derived_total_keys}

            derived_data_table = derived_data_table[:, avail_idx, :]
            if version_conf[version_name].get('fill_missvalue') == 'avg': # 填充缺失值
                logger.info('Fill missing data with average value')
                for key, idx in derived_idx_dict.items():
                    for s_idx in range(derived_data_table.shape[0]): # iter subject
                        arr = derived_data_table[s_idx, idx, :self._seqs_len[s_idx]]
                        derived_data_table[s_idx, idx, :self._seqs_len[s_idx]] = np.where(np.abs(arr + 1) > 1e-4, arr, derived_norm_dict[key]['mean'])

            # 设置k-fold
            derived_kf = KFold(n_splits=GLOBAL_CONF_LOADER['analyzer']['data_container']['n_fold'], \
                shuffle=True, random_state=GLOBAL_CONF_LOADER['analyzer']['data_container']['seed'])

            derived_kf_list = []
            for data_index, test_index in derived_kf.split(X=list(range(derived_data_table.shape[0]))): 
                # encode: train, valid, test
                valid_num = round(len(data_index)*self._loc_conf['validation_proportion'])
                train_index, valid_index = data_index[valid_num:], data_index[:valid_num]
                derived_kf_list.append({'train':train_index, 'valid':valid_index, 'test':test_index})
            
            version_dict = {
                'total_keys': derived_total_keys,
                'static_keys': derived_static_keys,
                'dynamic_keys': derived_dynamic_keys,
                'seqs_len': self._seqs_len,
                'idx_dict': derived_idx_dict,
                'norm_dict': derived_norm_dict,
                'kf': derived_kf_list,
                'all_item': self._all_items
            }
            np.savez_compressed(p_version_data, derived_data_table.astype(np.float32))
            with open(p_version, 'wb') as fp:
                pickle.dump(version_dict, fp)

    def _register_item(self, fea_id:str, fea_label:str):
        self._all_items['id'][fea_id] = {'id':fea_id, 'label':fea_label}
        self._all_items['label'][fea_label] = {'id':fea_id, 'label':fea_label}

    def fea_label(self, x:[int, str]):
        # input must be idx or id
        if isinstance(x, int): # idx
            assert(x < len(self._total_keys))
            id = self._total_keys[x]
            if id in self._all_items['id']:
                return self._all_items['id'][id]['label']
            else:
                return id
        elif x in self._all_items['id']:
            return self._all_items['id'][x]['label']
        else:
            return x

    def fea_id(self, x:[int, str]):
        if isinstance(x, int): # idx
            assert(x < len(self._total_keys))
            return self._total_keys[x] 
        elif x in self._all_items['label']:
            return self._all_items['label'][x]['id']
        else:
            return x
    
    def fea_idx(self, x:str):
        if x in self._all_items['label']:
            id = self._all_items['label'][x]['id']
            assert(id in self._idx_dict)
            return self._idx_dict[id]
        elif x in self._all_items['id'] or x in self._idx_dict.keys():
            assert(x in self._idx_dict)
            return self._idx_dict[x]
        else:
            assert(0)
    
    def register_split(self, train_index, valid_index, test_index):
        self.train_index = train_index
        self.valid_index = valid_index
        self.test_index = test_index

    def mode(self, mode=['train', 'valid', 'test', 'all']):
        '''切换dataset的模式, train/valid/test需要在register_split方法调用后才能使用'''
        if mode == 'train':
            self._data_index = self._kf_list[self._kf_index]['train']
        elif mode =='valid':
            self._data_index = self._kf_list[self._kf_index]['valid']
        elif mode =='test':
            self._data_index = self._kf_list[self._kf_index]['test']
        elif mode == 'all':
            self._data_index = None
        else:
            assert(0)
    

    def load_version(self, version_name):
        '''更新dataset版本'''
        if self._version_name == version_name:
            return
        else:
            self._version_name = version_name
        suffix = '.pkl' if not self._loc_conf['compress_cache'] else self._loc_conf['compress_suffix']
        p_version = os.path.join(self._paths['cache_dir'], f'7_version_{version_name}'+suffix)
        p_version_data = os.path.join(self._paths['cache_dir'], f'7_version_{version_name}.npz')
        assert(os.path.exists(p_version))
        with open(p_version, 'rb') as fp:
            version_dict = pickle.load(fp)
        
        self._static_keys = version_dict['static_keys']
        self._dynamic_keys = version_dict['dynamic_keys']
        self._total_keys = version_dict['total_keys']
        self._seqs_len = version_dict['seqs_len']
        self._idx_dict = version_dict['idx_dict']
        self._norm_dict = version_dict['norm_dict']
        self._kf_list = version_dict['kf']
        self._data = np.load(p_version_data)['arr_0']

    def enumerate_kf(self):
        return KFoldIterator(self, k=len(self._kf_list))

    def set_kf_index(self, kf_index):
        '''设置dataset对应K-fold的一折'''
        self._kf_index = kf_index
        self.train_index = self._kf_list[kf_index]['train']
        self.valid_index = self._kf_list[kf_index]['valid']
        self.test_index = self._kf_list[kf_index]['test']
        # self.mode('all')
        return self.train_index.copy(), self.valid_index.copy(), self.test_index.copy()
    
    def __getitem__(self, idx):
        assert(self._version_name is not None)
        if self._data_index is None: # 'all' mode
            return {'data': self._data[idx, :, :], 'length': self._seqs_len[idx]}
        else:# k-fold mode
            return {'data': self._data[self._data_index[idx], :, :], 'length': self._seqs_len[self._data_index[idx]]}

    def __len__(self):
        if self._data_index is None:
            return self._data.shape[0]
        else:
            return len(self._data_index)

    @abstractmethod
    def on_extract_subjects(self) -> tuple:
        pass

    @abstractmethod
    def on_extract_admission(self, source, row) -> bool:
        pass

    @abstractmethod
    def on_convert_numeric(self, s:Subject) -> Subject:
        pass

    @abstractmethod
    def on_build_subject(self, subject_id:int, subject:Subject, row:dict, patient_set:set, extra_data:object) -> Subject:
        pass

    @abstractmethod
    def on_select_feature(self, subject_id:int, row:dict, source:str=['icu', 'hosp', 'ed']):
        pass

    @abstractmethod
    def on_feature_engineering(self, table:np.ndarray, index_dict:dict, addi_feas:list, data_source:str):
        pass

    @abstractmethod
    def on_select_admissions(self, rule:dict, subjects:dict[int, Subject]):
        pass

    @abstractmethod
    def on_remove_missing_data(self, rule:dict, subjects:dict[int, Subject]):
        pass

    @abstractmethod
    def on_build_table(self, key, value, t_start):
        pass


