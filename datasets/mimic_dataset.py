import os
import tools
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tools import GLOBAL_CONF_LOADER
from tools import logger
from tqdm import tqdm
from sklearn.model_selection import KFold
from .mimic_helper import Subject, Admission, load_sepsis_patients
from scipy.interpolate import interp1d
from abc import abstractmethod

class MIMICIV(Dataset):
    '''
    MIMIC-IV底层抽象, 对源数据进行抽取/合并/移动, 生成中间数据集, 并将中间数据存储到pickle文件中
    这一步的数据是不对齐的, 仅考虑抽取和类型转化问题
    '''
    def __init__(self):
        super().__init__()
        # configs
        self._gbl_conf = GLOBAL_CONF_LOADER['dataset']['mimic-iv']
        self._mimic_dir:str = self._gbl_conf['paths']['mimic_dir']
        self._loc_conf = tools.Config(cache_path=self._gbl_conf['paths']['conf_cache_path'], manual_path=self._gbl_conf['paths']['conf_manual_path'])

        self._procedure_flag = 'init' # 控制标志, 进行不同阶段的cache和dump
        
        # self.align_key_ids = self.loc_conf['dataset']['alignment_key_id']
        # self.report_ids = self.loc_conf['dataset']['make_report']

        # variable for phase 1
        self._extract_result:dict = None
        self._hosp_item:dict = None
        self._icu_item:dict = None
        # variable for phase 2
        self._subjects:dict[int, Subject] = {} # subject_id:Subject
        # preload data
        self._data:np.ndarray = None # to derive other versions
        self.norm_dict:dict = None # key=str(name/id) value={'mean':mean, 'std':std}
        # self.static_keys:list[str] = None # list(str)
        # self.dynamic_keys:list[str] = None # list(str)
        self.total_keys:list[str] = None
        self.seqs_len:list = None # list(available_len)
        self.idx_dict:dict[int, str] = None
        # mode switch
        self.now_mode:str = None # 'train'/'valid'/'test'/'all'
        self._kf_list:list[dict] = None # list([train_index, valid_index, test_index])
        self.kf_index:int = None # 第几个fold
        self.data_index:list[int] = None # 当前模式下的索引
        # version switch
        self.version_name = None

        self._load_data(from_pkl=True)

        logger.info('MIMICIV inited')

    def _load_data(self, from_pkl=True):
        if not os.path.exists(self._gbl_conf['paths']['cache_dir']):
            tools.reinit_dir(self._gbl_conf['paths']['cache_dir'], build=True)
        
        self.preprocess_phase1(from_pkl)
        self.preprocess_phase2(from_pkl)
        self.preprocess_phase3(from_pkl)
        self.preprocess_phase4(from_pkl)
        self.preprocess_phase5(from_pkl)
        self.preprocess_phase6(from_pkl)
        self.preprocess_phase7(from_pkl)
 
    def preprocess_phase1(self, from_pkl=True):
        pkl_path = os.path.join(self._gbl_conf['paths']['cache_dir'], '1_phase1.pkl')
        if from_pkl and os.path.exists(pkl_path):
            logger.info(f'load .pkl for phase 1 from {pkl_path}')
            with open(pkl_path, 'rb') as fp:
                load_dict = pickle.load(fp)
                self._extract_result = load_dict['extract_result']
                self._icu_item = load_dict['icu_item']
                self._hosp_item = load_dict['hosp_item']
            self._procedure_flag = 'phase1'
            return
    
        logger.info(f'MIMIC-IV: processing dim file, flag={self._procedure_flag}')
        # 抽取符合条件的患者id
        
        
        # 建立hospital lab_item编号映射
        d_hosp_item = pd.read_csv(os.path.join(self._mimic_dir, 'hosp', 'd_labitems.csv'), encoding='utf-8')
        hosp_item = {}
        for _,row in tqdm(d_hosp_item.iterrows(), desc='hosp items'):
            hosp_item[str(row['itemid'])] = (row['label'], row['fluid'], row['category'])
        # 建立icu lab_item编号映射
        d_icu_item = pd.read_csv(os.path.join(self._mimic_dir, 'icu', 'd_items.csv'), encoding='utf-8')
        icu_item = {'id': {}, 'label': {}}
        for _,row in tqdm(d_icu_item.iterrows(), desc='icu items'):
            icu_item['id'][str(row['itemid'])] = {
                'id': str(row['itemid']),
                'label': row['label'], 
                'category': row['category'], 
                'type': row['param_type'], 
                'low': row['lownormalvalue'], 
                'high': row['highnormalvalue']
            }
            icu_item['label'][row['label']] = icu_item['id'][str(row['itemid'])] # 可以用名字或id查找
        # 抽取符合条件的患者id
        extract_result = self.on_extract_subjects()
        # 存储cache
        self._hosp_item = hosp_item
        self._icu_item = icu_item
        self._extract_result = extract_result
        with open(pkl_path, 'wb') as fp:
            pickle.dump({
                'extract_result': extract_result,
                'icu_item': icu_item,
                'hosp_item': self._hosp_item
                }, fp)
        self._procedure_flag = 'phase1'
    
    def preprocess_phase2(self, from_pkl=True):
        pkl_path = os.path.join(self._gbl_conf['paths']['cache_dir'], '2_phase2.pkl')
        if from_pkl and os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as fp:
                load_dict = pickle.load(fp)
                self._subjects = load_dict['subjects']
            logger.info(f'load .pkl for phase 2 from {pkl_path}')
            self._procedure_flag = 'phase2'
            return

        logger.info(f'MIMIC-IV: processing subjects and admissions, flag={self._procedure_flag}')
        # 构建subject
        patients = pd.read_csv(os.path.join(self._mimic_dir, 'hosp', 'patients.csv'), encoding='utf-8')
        for _, row in tqdm(patients.iterrows(), 'construct subject', total=len(patients)):
            s_id = row['subject_id']
            if s_id in self._extract_result:
                subject = Subject(row['subject_id'], anchor_year=row['anchor_year'])
                self._subjects[s_id] = self.on_build_subject(s_id, subject, row, self._extract_result)

        logger.info(f'Extract {len(self._subjects)} patients from {len(self._extract_result)} patients in extract_result')

        for path, prefix in zip(
            [os.path.join(self._mimic_dir, 'icu', 'icustays.csv'), os.path.join(self._mimic_dir, 'hosp', 'transfers.csv')], 
            ['icu', 'ed']
        ):
            table = pd.read_csv(path, encoding='utf-8')
            for _, row in tqdm(table.iterrows(), desc=f'Extract admissions from {prefix.upper()}', total=len(table)):
                s_id = int(row['subject_id'])
                if s_id in self._subjects:
                    adm = self.on_extract_admission(source=prefix, row=row)
                    if adm is not None:
                        self._subjects[s_id].append_admission(adm)
            del table

        # 患者的基本信息，如身高、体重、血压
        ymd_convertor = tools.TimeConverter(format="%Y-%m-%d", out_unit='hour')
        table_omr = pd.read_csv(os.path.join(self._mimic_dir, 'hosp', 'omr.csv'), encoding='utf-8') 
        # omr: [subject_id,chartdate,seq_num,result_name,result_value]
        for _, row in tqdm(table_omr.iterrows(), 'Extract patient information from OMR', total=len(table_omr)):
            s_id = int(row['subject_id'])
            if s_id in self._subjects:
                self._subjects[s_id].append_static(ymd_convertor(row['chartdate']), row['result_name'], row['result_value'])
        
        # dump
        with open(pkl_path, 'wb') as fp:
            pickle.dump({'subjects': self._subjects}, fp)
        logger.info(f'Phase 2 dumped at {pkl_path}')
        self._procedure_flag = 'phase2'

    def preprocess_phase3(self, from_pkl=True):
        pkl_path = os.path.join(self._gbl_conf['paths']['cache_dir'], '3_subjects.pkl')
        if from_pkl and os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as fp:
                self._subjects = pickle.load(fp)
            logger.info(f'load pkl for phase 3 from {pkl_path}')
            self._procedure_flag = 'phase3'
            return
        logger.info(f'MIMIC-IV: processing dynamic data, flag={self._procedure_flag}')
        
        # 决定捕捉哪些特征
        collect_id_set = set([id for id, row in self._icu_item['id'].items() if self.on_select_icu_item(id=id, row=row)])
        
        # 采集icu内的动态数据
        out_cache_dir = os.path.join(self._gbl_conf['paths']['cache_dir'], 'icu_events')
        if not os.path.exists(out_cache_dir):
            logger.info(f'Can not find split cache csvs. Run spliting function.')
            tools.split_csv(os.path.join(self._mimic_dir, 'icu', 'chartevents.csv'), out_folder=out_cache_dir)
        ymdhms_converter = tools.TimeConverter(format="%Y-%m-%d %H:%M:%S", out_unit='hour')

        icu_events = None
        p_bar = tqdm(total=len(os.listdir(out_cache_dir)), desc='Processing ICU events')
        for file_name in sorted(os.listdir(out_cache_dir)):
            icu_events = pd.read_csv(os.path.join(out_cache_dir, file_name), encoding='utf-8')[['subject_id', 'itemid', 'charttime', 'valuenum']].to_numpy()
            for idx in range(len(icu_events)):
                s_id, itemid = icu_events[idx, 0], str(icu_events[idx, 1])
                if s_id in self._subjects and itemid in collect_id_set:
                    self._subjects[s_id].append_dynamic(charttime=ymdhms_converter(icu_events[idx, 2]), itemid=itemid, value=icu_events[idx, 3])
            p_bar.update(1)
        
        # 整理admissions的格式
        for s_id in tqdm(self._subjects, desc='update data'):
            self._subjects[s_id].update_data()
        
        self._subjects = self.on_remove_invalid_pass1(rule=self._loc_conf['dataset']['remove_rule']['pass1'], subjects=self._subjects)
        # 保存subjects
        logger.info('Dump subjects')
        with open(pkl_path, 'wb') as fp:
            pickle.dump(self._subjects, fp)
        logger.info('Dump subjects: Done')
        self._procedure_flag = 'phase3'

    def preprocess_phase4(self, from_pkl=True):
        '''
        将所有特征转化为数值型, 并且对于异常值进行处理，最后进行特征筛选
        '''
        p_numeric_subject = os.path.join(self._gbl_conf['paths']['cache_dir'], '4_numeric_subject.pkl')
        if from_pkl and os.path.exists(p_numeric_subject):
            with open(p_numeric_subject, 'rb') as fp:
                self._subjects = pickle.load(fp)
            logger.info(f'Load numeric subject data from {p_numeric_subject}')
            return
        
        for s in self._subjects.values():
            s.static_data = self.on_convert_numeric(s.static_data)
        
        # 进行特征的上下界约束
        # TODO col_abnormal_rate = {}
        value_clip = self._loc_conf['dataset']['value_clip']
        for id_or_label in value_clip:
            id, label = self.get_id_and_label(id_or_label)
            clip_count = 0
            for s in self._subjects.values():
                for adm in s.admissions:
                    if id in adm.keys():
                        data = adm[id][:, 0]
                        adm[id][:, 0] = np.clip(data, a_min=value_clip[id_or_label]['min'], a_max=value_clip[id_or_label]['max'])
                        clip_count += 1
            logger.info(f'Value Clipping: clip {label} in {clip_count} admissions')

        # 计算当前subject中的缺失情况，同时计算行缺失和列缺失
        col_missrate = {}
        for s_id in self._subjects:
            s = self._subjects[s_id]
            for key in s.static_data.keys():
                if key not in col_missrate:
                    col_missrate[key] = 1
                else:
                    col_missrate[key] += 1
            for adm in s.admissions:
                for key in adm.keys():
                    if key not in col_missrate:
                        col_missrate[key] = 1 / len(s.admissions)
                    else:
                        col_missrate[key] += 1 / len(s.admissions)
        
        col_missrate = {key:1-val/len(self._subjects) for key, val in col_missrate.items()}
        n_all_features = len(col_missrate)
        subject_missrate = {}
        for s_id in self._subjects:
            s_keys = set(s.static_data.keys())
            for adm in s.admissions:
                for key in adm.keys():
                    s_keys.add(key)
            subject_missrate[s_id] = 1 - len(s_keys) / n_all_features
        # 进一步筛选admission
        self._subjects = self.on_remove_invalid_pass2(self._loc_conf['dataset']['remove_rule']['pass2'], self._subjects, col_missrate, subject_missrate)
        
        with open(p_numeric_subject, 'wb') as fp:
            pickle.dump(self._subjects, fp)
        logger.info(f'Numeric subjects dumped at {p_numeric_subject}')
        
    def preprocess_phase5(self, from_pkl=True):
        # 提取每个特征的均值和方差，用于归一化和均值填充
        
        p_norm_dict = os.path.join(self._gbl_conf['paths']['cache_dir'], '5_norm_dict.pkl')
        if from_pkl and os.path.exists(p_norm_dict):
            with open(p_norm_dict, 'rb') as fp:
                self.norm_dict = pickle.load(fp)
            logger.info(f'Load norm dict from {p_norm_dict}')
            return
        
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
        
        self.norm_dict = norm_dict
        with open(p_norm_dict, 'wb') as fp:
            pickle.dump(norm_dict, fp)
        logger.info(f'Norm dict dumped at {p_norm_dict}')

    def preprocess_phase6(self, from_pkl=True):
        '''
        对每个subject生成时间轴对齐的表, tick(hour)是生成的表的间隔
        '''  
        p_final_table = os.path.join(self._gbl_conf['paths']['cache_dir'], '6_table_final.pkl')

        if from_pkl and os.path.exists(p_final_table):
            with open(p_final_table, 'rb') as fp:
                result = pickle.load(fp)
                self._data, self.norm_dict, self.idx_dict, self.seqs_len, self.total_keys = \
                    result['data'], result['norm_dict'], result['index_dict'], result['seqs_len'], result['total_keys']
            logger.info(f'load aligned table from {p_final_table}')
            return
        
        # step1: 插值并生成表格
        tables:list[np.ndarray] = [] # table for all subjects
        # collect keys
        static_keys = set()
        dynamic_keys = set()
        for s in self._subjects.values():
            for fea in list(s.static_data.keys()):
                static_keys.add(fea)
            for fea in list(s.admissions[0].keys()):
                dynamic_keys.add(fea)
        static_keys, dynamic_keys = list(static_keys), list(dynamic_keys)
        collect_keys = static_keys + dynamic_keys
        logger.info(f'Detected {len(collect_keys)} available dynamic features')

        for s_id in tqdm(self._subjects.keys(), desc='Generate aligned table'):
            s = self._subjects[s_id]
            adm = s.admissions[0]
            
            t_start, t_end = None, None
            for id in self._loc_conf['dataset']['generate_table']['align_target']:
                t_start = max(adm[id][0,1], t_start) if t_start is not None else adm[id][0,1]
                t_end = min(adm[id][-1,1], t_end) if t_end is not None else adm[id][-1,1]
            t_step = self._loc_conf['dataset']['generate_table']['delta_t_hour']
            ticks = np.arange(t_start, t_end, t_step) # 最后一个会确保间隔不变且小于t_end
            # 生成表本身, 缺失值为-1
            individual_table = -np.ones((len(collect_keys), ticks.shape[0]), dtype=np.float32)

            # 填充static data, 找最近的点
            static_data = np.zeros((len(static_keys)))
            for idx, key in enumerate(static_keys):
                value = self._subjects[s_id].nearest_static(key, t_start)
                static_data[idx] = self.on_build_table(key, value, t_start)
            individual_table[:len(static_keys), :] = static_data[:, None]

            # 插值dynamic data
            for idx, key in enumerate(dynamic_keys):
                if key in adm.keys():
                    interp = interp1d(x=adm[key][:, 1], y=adm[key][:, 0], kind='previous', fill_value="extrapolate") # TODO need test
                    individual_table[len(static_keys)+idx, :] = interp(x=ticks)
            
            tables.append(individual_table)
        result = self.on_feature_engineering(tables, self.norm_dict, collect_keys) # 特征工程
        tables, self.norm_dict, total_keys = result['tables'], result['norm_dict'], result['total_keys']
        
        index_dict = {key:val for val, key in enumerate(total_keys)} # used for finding index
        
        # step2: 时间轴长度对齐, 生成seqs_len, 进行某些特征的最后处理
        seqs_len = [d.shape[1] for d in tables]
        max_len = max(seqs_len)
        for t_idx in tqdm(range(len(tables)), desc='Padding tables'):
            if seqs_len[t_idx] == max_len:
                continue
            padding = -np.ones((len(total_keys), max_len - seqs_len[t_idx]))
            tables[t_idx] = np.concatenate([tables[t_idx], padding], axis=1)
        tables = np.stack(tables, axis=0) # (n_sample, n_fea, seqs_len)

        self._data = tables
        self.idx_dict = index_dict
        self.seqs_len = seqs_len
        self.total_keys = total_keys
        with open(p_final_table, 'wb') as fp:
            pickle.dump({
                'data': tables,
                'index_dict': index_dict,
                'norm_dict': self.norm_dict,
                'seqs_len': seqs_len,
                'total_keys': total_keys
            }, fp)
        logger.info(f'Aligned table dumped at {p_final_table}')

    def preprocess_phase7(self, from_pkl=True):
        '''
        生成不同版本的数据集, 不同版本的数据集的样本数量/特征数量都可能不同
        '''
        assert(self.idx_dict is not None)
        p_version = os.path.join(self._gbl_conf['paths']['cache_dir'], '7_version.pkl')
        version_conf:dict = self._loc_conf['dataset']['version']

        for version_name in version_conf.keys():
            logger.info(f'Preprocessing version: {version_name}')
            # 检查是否存在pkl
            p_version = os.path.join(self._gbl_conf['paths']['cache_dir'], f'8_version_{version_name}.pkl')
            if os.path.exists(p_version):
                logger.info(f'Skip preprocess existed version: {version_name}')
                continue
            version_table = self._data.copy()
            
            # 筛选可用样本
            # if 'data_source' in version_conf[version_name]:
            #     data_source_parts = []
            #     sources = version_conf[version_name]['data_source']
            #     for source in sources:
            #         flag = {'icu':1, 'ed':2}[source]
            #         data_source_parts.append(
            #             version_table[version_table[:, self.idx_dict['data_source'], 0] == flag]
            #         )
            #     version_table = np.concatenate(data_source_parts, axis=0) if len(data_source_parts) > 1 else data_source_parts[0]
            
            # 筛选可用特征
            if len(version_conf[version_name]['feature_limit']) > 0:
                limit_idx = [self.idx_dict[lfea] for lfea in version_conf[version_name]['feature_limit'] if lfea in self.idx_dict]
            else:
                limit_idx = list(self.idx_dict.values())
            
            forbidden_idx = set([self.idx_dict[ffea] for ffea in version_conf[version_name]['forbidden_feas'] if ffea in self.idx_dict])
            avail_idx = [idx for idx in limit_idx if idx not in forbidden_idx]
            
            if version_conf[version_name].get('fill_missvalue') == 'avg': # 填充缺失值
                for key, idx in self.idx_dict.items():
                    for s_idx in range(version_table.shape[0]):
                        arr = version_table[s_idx, idx, :self.seqs_len[s_idx]]
                        version_table[s_idx, idx, :self.seqs_len[s_idx]] = np.where(np.abs(arr + 1) > 1e-4, arr, self.norm_dict[key]['mean'])
            # 更新特征
            version_table = version_table[:, avail_idx, :] # 这里不sort是为了保证PF ratio处于最后一位
            total_keys = [self.total_keys[idx] for idx in avail_idx]
            derived_idx_dict = {key:idx for idx, key in enumerate(total_keys)}
            
            
            # derived_static_keys = total_keys[:np.sum(np.asarray(avail_idx) < len(self.static_keys))]
            # derived_dynamic_keys = total_keys[np.sum(np.asarray(avail_idx) < len(self.static_keys)):]
            
            # 设置k-fold
            kf = KFold(n_splits=GLOBAL_CONF_LOADER['analyzer']['data_container']['n_fold'], \
                shuffle=True, random_state=GLOBAL_CONF_LOADER['analyzer']['data_container']['seed'])

            kf_list = []
            for data_index, test_index in kf.split(X=list(range(version_table.shape[0]))): 
                # encode: train, valid, test
                valid_num = round(len(data_index)*self._loc_conf['dataset']['validation_proportion'])
                train_index, valid_index = data_index[valid_num:], data_index[:valid_num]
                kf_list.append({'train':train_index, 'valid':valid_index, 'test':test_index})
            
            version_dict = {
                'total_keys': total_keys,
                'seqs_len':self.seqs_len,
                'idx_dict':derived_idx_dict,
                'data': version_table,
                'kf': kf_list,
            }
            with open(p_version, 'wb') as fp:
                pickle.dump(version_dict, fp)

    def get_fea_label(self, key_or_idx):
        '''输入key/idx得到关于特征的简短描述, 从icu_item中提取'''
        if isinstance(key_or_idx, int):
            name = self.total_keys[key_or_idx]
        else:
            name = key_or_idx
        if self._icu_item['label'].get(name) is not None:
            return self._icu_item['label'][name]['label']
        elif self._icu_item['id'].get(name) is not None:
            return self._icu_item['id'][name]['label']
        else:
            logger.warning(f'No fea label for {name}, return name')
            return name

    def label2id(self, label):
        result = self._icu_item['label'].get(label)
        if result is not None:
            return result['id']
        else:
            logger.warning(f"label is not in icu item: {label}")
            return None
    
    def id2label(self, id):
        result = self._icu_item['id'].get(id)
        if result is not None:
            return result['label']
        else:
            logger.warning(f"id is not in icu item: {id}")
            return None
    
    def get_id_and_label(self, id_or_label:str):
        if id_or_label in self._icu_item['id']:
            return id_or_label, self.id2label(id_or_label)
        elif id_or_label in self._icu_item['label']:
            return self.label2id(id_or_label), id_or_label
        else:
            return None, None
            
    def register_split(self, train_index, valid_index, test_index):
        self.train_index = train_index
        self.valid_index = valid_index
        self.test_index = test_index

    def mode(self, mode=['train', 'valid', 'test', 'all']):
        '''切换dataset的模式, train/valid/test需要在register_split方法调用后才能使用'''
        if mode == 'train':
            self.data_index = self._kf_list[self.kf_index]['train']
        elif mode =='valid':
            self.data_index = self._kf_list[self.kf_index]['valid']
        elif mode =='test':
            self.data_index = self._kf_list[self.kf_index]['test']
        elif mode == 'all':
            self.data_index = None
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
        if isinstance(name_or_idx, int):
            name_or_idx = self.total_keys[name_or_idx]
        norm = self.norm_dict[name_or_idx]
        if mask is None:
            return data * norm['std'] + norm['mean']
        else:
            assert(mask.shape == data.shape)
            return (data * norm['std'] + norm['mean']) * (mask > 0) + data * (mask <= 0)


    def load_version(self, version_name):
        '''更新dataset版本'''
        if self.version_name == version_name:
            return
        else:
            self.version_name = version_name
        
        p_version = os.path.join(self.gbl_conf['paths']['cache_dir'], f'8_version_{version_name}.pkl')
        assert(os.path.exists(p_version))
        with open(p_version, 'rb') as fp:
            version_dict = pickle.load(fp)
        # self.static_keys = version_dict['static_keys']
        # self.dynamic_keys = version_dict['dynamic_keys']
        self.total_keys = version_dict['total_keys']
        self.seqs_len = version_dict['seqs_len']
        self._data = version_dict['data']
        self.idx_dict = version_dict['idx_dict']
        self._kf_list = version_dict['kf']

    def enumerate_kf(self):
        return KFoldIterator(self, k=len(self._kf_list))

    def set_kf_index(self, kf_index):
        '''设置dataset对应K-fold的一折'''
        self.kf_index = kf_index
        self.train_index = self._kf_list[kf_index]['train']
        self.valid_index = self._kf_list[kf_index]['valid']
        self.test_index = self._kf_list[kf_index]['test']
        # self.mode('all')
        return self.train_index.copy(), self.valid_index.copy(), self.test_index.copy()
    
    def __getitem__(self, idx):
        assert(self.version_name is not None)
        if self.data_index is None:
            return {'data': self._data[idx, :, :], 'length': self.seqs_len[idx]}
        else:
            return {'data': self._data[self.data_index[idx], :, :], 'length': self.seqs_len[self.data_index[idx]]}

    def __len__(self):
        if self.data_index is None:
            return self._data.shape[0]
        else:
            return len(self.data_index)

    @abstractmethod
    def on_extract_subjects(self) -> dict:
        pass

    @abstractmethod
    def on_extract_admission(self, source, row):
        pass

    @abstractmethod
    def on_convert_numeric(self, static_data:dict[list]) -> dict[np.ndarray]:
        pass

    @abstractmethod
    def on_build_subject(self, id:int, subject:Subject, row:dict, _extract_result:dict) -> dict:
        pass

    @abstractmethod
    def on_select_icu_item(self, id:int, row:dict):
        pass

    @abstractmethod
    def on_feature_engineering(self, table:np.ndarray, index_dict:dict, addi_feas:list, data_source:str):
        pass

    @abstractmethod
    def on_remove_invalid_pass1(self, rule:dict, subjects:dict[int, Subject]):
        pass

    @abstractmethod
    def on_remove_invalid_pass2(self, rule:dict, subjects:dict[int, Subject]):
        pass

    @abstractmethod
    def on_build_table(self, key, value, t_start):
        pass

    

class MIMICIVDataset(MIMICIV):
    '''
    MIMIC-IV上层抽象, 从中间文件读取数据, 进行处理, 得到最终数据集
    (batch, n_fea, seq_len)
    有效接口: norm_dict(key可能是多余的), idx_dict, static_keys, dynamic_keys, total_keys
    '''
    __name = 'mimic-iv'

    @classmethod
    def name(cls):
        return cls.__name
    
    def __init__(self):
        super().__init__()
        # self.mimiciv = MIMICIV()
        # self._subjects = self.mimiciv._subjects
        # self.gbl_conf = self.mimiciv._gbl_conf
        # self.loc_conf = self.mimiciv._loc_conf
        # self.additional_feas = self.loc_conf['dataset']['additional_features']
        # hit table
        # self._hit_table = None
        # miss_table
        # self._global_misstable = None
        
        # logger.info(f'Dynamic keys={len(self.dynamic_keys)}, static_keys={len(self.static_keys)}')

    @abstractmethod
    def on_remove_invalid_pass1(self, rule:dict, subjects:dict[int, Subject]):
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

    def on_remove_invalid_pass2(self, rule:dict, subjects: dict[int, Subject], col_missrate:dict[str, float], subject_missrate: dict[int, float]) -> dict[int, Subject]:
        '''
        按照传入的配置去除无效特征
        '''
        adm_select_strategy = rule['choose_admission_in_subject']
        subject_pop_list = []
        for s_id, s in subjects.items():
            if subject_missrate[s_id] > rule['max_subject_missrate']: # invalid subject
                subject_pop_list.append(s_id)
                continue
            if adm_select_strategy == 'random':
                retain_adm = np.random.randint(0, len(s.admissions))
            else:
                retain_adm = 0 # default strategy is keeping the first admission
            s.admissions = s.admissions[retain_adm]
            adm:Admission = s.admissions[0]
            for key in adm.keys():
                if col_missrate[key] > rule['max_col_missrate']:
                    adm.pop_dynamic(key)
            subjects[s_id].admissions[0] = adm

        # 删除空的admission和subjects
        pop_list = []
        for s_id in subjects:
            subjects[s_id].del_empty_admission() # 删除空的admission
            if subjects[s_id].empty():
                pop_list.append(s_id)
                
        for s_id in pop_list:
            subjects.pop(s_id)
        
        logger.info(f'remove_pass2: Deleted {len(pop_list)}/{len(pop_list)+len(self._subjects)} subjects')
        return subjects

    def on_extract_subjects(self) -> dict:
        sepsis_patient_path = self._gbl_conf['paths']['sepsis_patient_path']
        sepsis_result = load_sepsis_patients(sepsis_patient_path)
        return sepsis_result
    
    def on_build_subject(self, id:int, subject:Subject, row:dict, _extract_result:dict) -> dict:
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
        for ele_dict in _extract_result[id]: # dict(list(dict))
            sepsis_time = ele_dict['sepsis_time']

            # self._subjects[s_id].append_static(sepsis_time, 'age', row['anchor_age']) 每次入院的年龄是有可能变化的
            subject.append_static(sepsis_time, 'gender', row['gender'])
            if row['dod'] is not None and isinstance(row['dod'], str):
                subject.append_static(sepsis_time, 'dod', ymd_convertor(row['dod']))
            subject.append_static(sepsis_time, 'sepsis_time', sepsis_time)
            subject.append_static(sepsis_time, 'sofa_score', ele_dict['sofa_score'])
            subject.append_static(sepsis_time, 'respiration', ele_dict['respiration'])
            subject.append_static(sepsis_time, 'liver', ele_dict['liver'])
            subject.append_static(sepsis_time, 'cardiovascular', ele_dict['cardiovascular'])
            subject.append_static(sepsis_time, 'cns', ele_dict['cns'])
            subject.append_static(sepsis_time, 'renal', ele_dict['renal'])
        return subject
    
    def on_extract_admission(self, source, row):
        ymdhms_converter = tools.TimeConverter(format="%Y-%m-%d %H:%M:%S", out_unit='hour')
        if source == 'icu':
            return Admission(
                unique_id=int(row['hadm_id']*1e8+row['stay_id']),
                admittime=ymdhms_converter(row['intime']), 
                dischtime=ymdhms_converter(row['outtime']),
                label='icu',
            )
        elif source == 'ed':
            if row['careunit'] != 'Emergency Department':
                return None
            else:
                if not np.isnan(row['hadm_id']):
                    unique_id = int(row['hadm_id']*1e8+row['transfer_id'])
                else:
                    unique_id = int(row['transfer_id']*1e8+row['transfer_id']) # transfer中某些情况没有分配admission
                return Admission(
                    unique_id=unique_id,
                    admittime=ymdhms_converter(row['intime']), 
                    dischtime=ymdhms_converter(row['outtime']),
                    label='ed',
                )
        else:
            assert(0)

    def on_select_icu_item(self, id:int, row:dict):
        if row['type'] in ['Numeric', 'Numeric with tag'] and row['category'] != 'Alarms':
            return True # select
        else:
            return False # not select
    
    def on_convert_numeric(self, static_data:dict[list]) -> dict[np.ndarray]:
        '''
        1. 对特定格式的特征进行转换(血压)
        2. 检测不能转换为float的静态特征
        '''
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
                    except Exception as e:
                        logger.warning(f'Invalid value [{v}] for {key}')
                    static_data[key][idx] = (v,t)
                    valid_idx.append(idx)
                static_data[key] = np.asarray(static_data[key])[valid_idx, :]
        return static_data

    def on_build_table(self, key, value, t_start):
        if key == 'sepsis_time' or key == 'dod': # sepsis time 基准变为表格的起始点
            return value - t_start
      
    def on_feature_engineering(self, tables:list[np.ndarray], norm_dict:dict, collect_keys:list):
        '''
        特征工程, 增加某些计算得到的特征
        '''
        addi_feas = ['shock_index', 'MAP', 'PPD', 'PF_ratio']
        total_keys = collect_keys + addi_feas
        index_dict = {key:val for val, key in enumerate(collect_keys)} # used for finding index
        norm_data = []
        for t_idx, table in enumerate(tables):
            addi_table = np.zeros((len(addi_feas), table.shape[1]))
            for idx, name in enumerate(addi_feas):
                if name == 'PF_ratio':
                    addi_table[idx, :] = np.clip(table[index_dict['220224'], :] / (table[index_dict['223835'], :]*0.01), 0, 500)
                elif name == 'shock_index':
                    if np.all(table[index_dict['systolic pressure'], :] > 0):
                        addi_table[idx, :] = table[index_dict['220045'], :] / table[index_dict['systolic pressure'], :]
                    else:
                        addi_table[idx, :] = -1 # missing value
                        logger.warning('feature_engineering: skip shock_index with zero sbp')
                elif name == 'MAP':
                    addi_table[idx, :] = (table[index_dict['systolic pressure'], :] + 2*table[index_dict['diastolic pressure'], :]) / 3
                elif name == 'PPD':
                    addi_table[idx, :] = table[index_dict['systolic pressure'], :] - table[index_dict['diastolic pressure'], :]
                else:
                    logger.error(f'Invalid feature name:{name}')
                    assert(0)
            tables[t_idx] = np.concatenate([table, addi_table], axis=0)
            norm_data.append(addi_table)
        # update norm dict
        norm_data = np.concatenate(norm_data, axis=-1)
        for idx, key in enumerate(addi_feas):
            mean, std = np.mean(norm_data[idx, :]), np.std(norm_data[idx, :])
            norm_dict[key] = {'mean': mean, 'std': std}
        return {
            'tables': tables,
            'norm_dict': norm_dict,
            'total_keys': total_keys
        }
        
    
    
    def make_report(self, version_name, params:dict):
        '''进行数据集的信息统计'''
        # switch version
        self.load_version(version_name)
        self.mode('all')
        out_path = os.path.join(self._gbl_conf['paths']['out_dir'], f'dataset_report_{version_name}.txt')
        dist_dir = os.path.join(self._gbl_conf['paths']['out_dir'], 'report_dist')
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
            write_lines.append(f'subjects:{len(self)}')
            write_lines.append(f'Static feature: {[self.get_fea_label(id) for id in self.static_keys]}')
            write_lines.append(f'Dynamic feature: {[self.get_fea_label(id) for id in self.dynamic_keys]}')
        if params['dynamic_dist']:
            # dynamic feature explore
            for id in self.dynamic_keys:
                fea_name = self.get_fea_label(id)
                save_name = tools.remove_slash(str(fea_name))
                write_lines.append('='*10 + f'{fea_name}({id})' + '='*10)
                arr_points = []
                arr_duration = []
                arr_frequency = []
                arr_avg_value = []
                arr_from_sepsis_time = []
                for s in tqdm(self._subjects.values(), desc=f'id={id}'):
                    for adm in s.admissions:
                        if id in adm.keys():
                            dur = adm[id][-1,1] - adm[id][0,1]
                            
                            sepsis_time = s.nearest_static('sepsis_time', adm[id][0, 1])
                            t_sep = adm[id][0, 1] + adm.admittime - sepsis_time
                            arr_from_sepsis_time.append(t_sep)
                            arr_points.append(adm[id].shape[0])
                            arr_duration.append(dur)
                            if dur > 1e-3: # TODO 只有一个点无法计算
                                arr_frequency.append(arr_points[-1] / arr_duration[-1])
                            else:
                                arr_frequency.append(0)
                            arr_avg_value.append(adm[id][:,0].mean())
                arr_points, arr_duration, arr_frequency, arr_avg_value = np.asarray(arr_points), np.asarray(arr_duration), np.asarray   (arr_frequency), np.asarray(arr_avg_value)
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
                for title, arr in tqdm(zip(titles, arrs),'plot'):
                    if np.size(arr) != 0:
                        tools.plot_single_dist(
                            data=arr, data_name=f'{title}: {fea_name}', 
                            save_path=os.path.join(dist_dir, title, save_name + '.png'), discrete=False, adapt=True)
        if params['static_dist']:
            # static feature explore
            for id in tqdm(self.static_keys, 'generate static feature report'):
                fea_name = self.get_fea_label(id)
                save_name = tools.remove_slash(str(fea_name))
                write_lines.append('='*10 + f'{fea_name}({id})' + '='*10)
                idx = self.idx_dict[str(id)]
                static_data = self._data[:, idx, 0]
                write_lines.append(f'mean: {static_data.mean():.3f}')
                write_lines.append(f'std: {static_data.std():.3f}')
                write_lines.append(f'max: {np.max(static_data):.3f}')
                write_lines.append(f'min: {np.min(static_data):.3f}')
                tools.plot_single_dist(
                    data=static_data, data_name=f'{fea_name}', 
                    save_path=os.path.join(dist_dir, 'static_value', save_name + '.png'), discrete=False, adapt=True)
        if params['hit_table']['enabled']:
            # write hit table
            hit_table = self.hit_table()
            key_list = sorted(hit_table.keys(), key= lambda key:hit_table[key], reverse=True)
            write_lines.append('\n='*10 + 'Feature hit table' + '='*10)
            for key in key_list:
                fea_name = self.get_fea_label(key)
                value = hit_table[key]
                if value < params['hit_table']['min_thres']:
                    continue
                write_lines.append(f"{value:.2f}\t({key}){fea_name} ")
        if params['global_missrate']:
            # write global miss rate
            write_lines.append('\n='*10 + 'Global miss rate' + '='*10)
            gbl_msr = self.global_misstable()
            for idx, key in enumerate(self.total_keys):
                write_lines.append(f'{gbl_msr[idx]:.3f} \t ({key}){self.get_fea_label(key)}')
        # write report
        with open(out_path, 'w', encoding='utf-8') as fp:
            for line in write_lines:
                fp.write(line + '\n')
        logger.info(f'Report generated at {out_path}')


class KFoldIterator:
    def __init__(self, dataset:MIMICIV, k):
        self.current = -1
        self.k = k
        self.dataset = dataset

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current < self.k:
            return self.dataset.set_kf_index(self.current)
        else:
            raise StopIteration


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/chenyt/sepsis_data_processing/data_processing')
    dataset = MIMICIVDataset()
    