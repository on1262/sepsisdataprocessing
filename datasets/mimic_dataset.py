import os, sys
sys.path.insert(0, os.getcwd())
import tools
import pickle
import numpy as np
import pandas as pd
from tools import GLOBAL_CONF_LOADER
from tools import logger
from tqdm import tqdm
from sklearn.model_selection import KFold


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

    def __getitem__(self, idx): # TODO error 220074
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
            assert(time is not None)
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
        self.gbl_conf = GLOBAL_CONF_LOADER['dataset']['mimic-iv']
        # paths
        self.mimic_dir = self.gbl_conf['paths']['mimic_dir']
        # configs
        self.loc_conf = tools.Config(cache_path=self.gbl_conf['paths']['conf_cache_path'], manual_path=self.gbl_conf['paths']['conf_manual_path'])
        self.procedure_flag = 'init' # 控制标志, 进行不同阶段的cache和dump
        self.converter = tools.TimeConverter(format="%Y-%m-%d %H:%M:%S", out_unit='hour')
        self.target_icu_ids = self.loc_conf['dataset']['target_icu_id']
        # self.report_ids = self.loc_conf['dataset']['make_report']
        # variable for phase 1
        self.sepsis_result = None
        self.hosp_item = None
        self.icu_item = None
        # variable for phase 2
        self.subjects = {} # subject_id:Subject
        self.preprocess()
        # post process
        self.remove_invalid_data(rules=self.loc_conf['dataset']['remove_rule'])
        logger.info('MIMICIV inited')

    def preprocess(self, from_pkl=True):
        self.preprocess_phase1(from_pkl)
        self.preprocess_phase2(from_pkl)
        self.preprocess_phase3(from_pkl)
 
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
        # 抽取符合条件的患者id
        sepsis_result = load_sepsis_patients(self.gbl_conf['paths']['sepsis_patient_path'])
        logger.info(f'Extracted {len(sepsis_result.keys())} sepsis subjects')
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
                '''
                NOTE: sepsis time的处理方式
                sepsis time被看作一个静态特征添加到subject下, 一个subject可以有多个sepsis time
                如果一个admission没有sepsis time对应, 那么这个admission无效
                在最终的三维数据上, sepsis_time会变为距离起始点t_start的相对值(sep-t_start)
                由于起始点设为max(sepsis, t_start), 所以sepsis_time只会是负数或者0
                当sepsis_time<0的时候, 表明sepsis发生得早, 对于一些模型, sepsis time不能太小, 可以用来筛选数据
                '''
                for element in self.sepsis_result[s_id]:
                    sepsis_time, _, sofa_score, respiration, liver, cardiovascular, cns, renal = element
                    self.subjects[s_id] = Subject(row['subject_id'], anchor_year=row['anchor_year'])
                    self.subjects[s_id].append_static(sepsis_time, 'age', row['anchor_age'])
                    self.subjects[s_id].append_static(sepsis_time, 'gender', row['gender'])
                    self.subjects[s_id].append_static(sepsis_time, 'sepsis_time', sepsis_time)
                    self.subjects[s_id].append_static(sepsis_time, 'sofa_score', sofa_score)
                    self.subjects[s_id].append_static(sepsis_time, 'respiration', respiration)
                    self.subjects[s_id].append_static(sepsis_time, 'liver', liver)
                    self.subjects[s_id].append_static(sepsis_time, 'cardiovascular', cardiovascular)
                    self.subjects[s_id].append_static(sepsis_time, 'cns', cns)
                    self.subjects[s_id].append_static(sepsis_time, 'renal', renal)
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


    def preprocess_phase3(self, from_pkl=True):
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
        if not os.path.exists(out_cache_dir):
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
                            if dur > max(rules['min_duration'], 0) and dur < rules['max_duration']:
                                points = max(np.sum((adm[pao2_id][:,1] >= start_time) * (adm[pao2_id][:,1] <= end_time)), \
                                    np.sum((adm[fio2_id][:,1] >= start_time) * (adm[fio2_id][:,1] <= end_time)))
                                if points >= rules['min_points'] and dur/points <= rules['max_avg_interval']:
                                    flag *= 1
                                else:
                                    flag = 0
                            else:
                                flag = 0
                            if flag != 0:
                                # 检查sepsis time, 必须要有一个sepsis time和这次admission对应上
                                sepsis_time = self.subjects[s_id].nearest_static('sepsis_time', adm.admittime+start_time)[0][0]
                                if not (-10 < adm.admittime+start_time-sepsis_time < 30):
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


class MIMICDataset():
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
        self.mimiciv = MIMICIV()
        self.subjects = self.mimiciv.subjects
        self.gbl_conf = self.mimiciv.gbl_conf
        self.loc_conf = self.mimiciv.loc_conf
        self.target_name = self.loc_conf['dataset']['target_label']
        self._additional_feas = self.loc_conf['dataset']['additional_features']
        # hit table
        self._hit_table = None
        # preload data
        self.data_table = None # to derive other versions
        self.data = None # ndarray(samples, n_fea, ticks)
        self.norm_dict = None # key=str(name/id) value={'mean':mean, 'std':std}
        self.static_keys = None # list(str)
        self.dynamic_keys = None # list(str)
        self.seqs_len = None # list(available_len)
        self.idx_dict = None
        self.target_idx = None
        # mode switch
        self.now_mode = None # 'train'/'valid'/'test'/'all'
        self.kf_list = None # list([train_index, valid_index, test_index])
        self.kf_index = None # int
        self.data_index = None # 当前模式下的索引
        self.train_index = None
        self.valid_index = None
        self.test_index = None
        # version switch
        self.version_name = None

        self.preprocess()
        self.mimiciv.subjects = self.subjects # update
        
        logger.info(f'Dynamic keys={len(self.dynamic_keys)}, static_keys={len(self.static_keys)}')

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

    # def register_split(self, train_index, valid_index, test_index):
    #     self.train_index = train_index
    #     self.valid_index = valid_index
    #     self.test_index = test_index

    def mode(self, mode=['train', 'valid', 'test', 'all']):
        '''切换dataset的模式, train/valid/test需要在register_split方法调用后才能使用'''
        if mode == 'train':
            self.data_index = self.kf_list[self.kf_index]['train']
        elif mode =='valid':
            self.data_index = self.kf_list[self.kf_index]['valid']
        elif mode =='test':
            self.data_index = self.kf_list[self.kf_index]['test']
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

    def _available_dyn_id(self, min_cover_rate=0.5) -> list:
        '''
        根据hit table生成符合最小覆盖率的动态特征id
        return: list(id)
        '''
        hit_table = self.hit_table()
        result = []
        for key in hit_table.keys():
            if hit_table[key] > min_cover_rate:
                result.append(key)
        return result

    def preprocess(self, from_pkl=True):
        p_numeric_subject = os.path.join(self.gbl_conf['paths']['cache_dir'], '4_numeric_subject.pkl')
        p_norm_dict = os.path.join(self.gbl_conf['paths']['cache_dir'], '5_norm_dict.pkl')
        # preprocessing
        self.static_feas = set()
        self.target_label = self.loc_conf['dataset']['target_label']
        self.dynamic_ids = self._available_dyn_id()
        logger.info(f'Detected {len(self.dynamic_ids)} available dynamic features')

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
        # init
        self.static_feas, self.dynamic_ids = None, None # not use these attributes
        self.target_idx = self.data_table.shape[1] - 1
        self.idx_dict = dict( # idx_dict: fea_key->idx
            {str(key):val for val, key in enumerate(self.static_keys)}, \
                **{str(key):val+len(self.static_keys) for val, key in enumerate(self.dynamic_keys)})
        # preprocess version
        self.proprocess_version()

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
                elif isinstance(s.static_data[key], list):
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
            
            # pao2和fio2的上下界约束
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

    def global_miss_rate(self) -> dict:
        '''
        该方法在preprocess之后调用
        给出基于table_origin的全特征缺失率表
        返回dict(key:value)其中key是str(id), value是该特征对应的列缺失率
        '''
        p_origin_table = os.path.join(self.gbl_conf['paths']['cache_dir'], '6_table_origin.pkl')
        miss_rate = None
        with open(p_origin_table, 'rb') as fp:
            origin_table,_, _,_ = pickle.load(fp)
        for idx, table in enumerate(origin_table):
            if idx == 0:
                miss_rate = np.mean(table == -1, dim=1)
            else:
                miss_rate += np.mean(table == -1, dim=1)
        miss_rate /= len(origin_table)
        return {key:miss_rate[self.idx_dict[key]] for key in self.idx_dict.keys()}
        

    def preprocess_table(self, from_pkl=True, t_step=0.5):
        '''
        对每个subject生成时间轴对齐的表, tick(hour)是生成的表的间隔
        origin_table:  list(ndarray(n_fea, lens)) 其中lens不对齐, 缺失值为-1
        '''
        p_origin_table = os.path.join(self.gbl_conf['paths']['cache_dir'], '6_table_origin.pkl')
        # p_normed_table = os.path.join(self.gbl_conf['paths']['cache_dir'], '7_table_norm.pkl')
        p_final_table = os.path.join(self.gbl_conf['paths']['cache_dir'], '7_table_final.pkl')
        # step1: 插值并生成表格
        if from_pkl and os.path.exists(p_origin_table):
            with open(p_origin_table, 'rb') as fp:
                self.data_table, self.norm_dict, self.static_keys, self.dynamic_keys = pickle.load(fp)
            logger.info(f'load original aligned table from {p_origin_table}')
        else:
            data_table = []
            align_id = self.loc_conf['dataset']['align_target_id'] # 用来确认对齐的基准时间
            static_keys = list(self.static_feas)
            dynamic_keys = self.dynamic_ids + self._additional_feas
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
                    sepsis_time = self.subjects[s_id].nearest_static('sepsis_time', t_start+adm.admittime) - adm.admittime
                    if sepsis_time >= (t_end-self.loc_conf['dataset']['remove_rule']['min_duration']):
                        logger.warning('Invalid admission: no matching sepsis time')
                        continue
                    else:
                        t_start = max(sepsis_time, t_start)
                    ticks = np.arange(t_start, t_end, t_step) # 最后一个会确保间隔不变且小于t_end
                    # 生成表本身, 缺失值为-1
                    table = -np.ones((len(static_keys) + len(dynamic_keys), ticks.shape[0]), dtype=np.float32)
                    # 填充static data, 找最近的点
                    static_data = np.zeros((len(static_keys)))
                    for idx, key in enumerate(static_keys):
                        static_data[idx] = self.subjects[s_id].nearest_static(key, adm[align_id][0, 1] + adm.admittime)
                        if key == 'sepsis_time': # sepsis time 基准变为表格的起始点
                            static_data[idx] = static_data[idx] - adm.admittime - t_start
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
                    data_table.append(table)
            # 计算特征工程新增特征的norm_dict
            additional_vals = {key:np.concatenate(val, axis=0) for key, val in additional_vals.items()}
            additional_vals = {key:{'mean':val.mean(), 'std':val.std()} for key,val in additional_vals.items()}
            for key in additional_vals:
                self.norm_dict[key] = additional_vals[key]
            self.data_table = data_table
            self.static_keys = static_keys
            self.dynamic_keys = dynamic_keys
            with open(p_origin_table, 'wb') as fp:
                pickle.dump([self.data_table, self.norm_dict, static_keys, dynamic_keys], fp)
            logger.info(f'data table dumped at {p_origin_table}')
        
        # step2: 时间轴长度对齐, 生成seqs_len, 进行某些特征的最后处理
        if from_pkl and os.path.exists(p_final_table):
            with open(p_final_table, 'rb') as fp:
                seqs_len, self.static_keys, self.data_table = pickle.load(fp)
                self.seqs_len = seqs_len
            logger.info(f'load length aligned table from {p_final_table}')
        else:
            seqs_len = [d.shape[1] for d in self.data_table]
            self.seqs_len = seqs_len
            max_len = max(seqs_len)
            n_fea = len(self.static_keys) + len(self.dynamic_keys)
            for t_idx in tqdm(range(len(self.data_table)), desc='length alignment'):
                if seqs_len[t_idx] == max_len:
                    continue
                new_table = -np.ones((n_fea, max_len - seqs_len[t_idx]))
                self.data_table[t_idx] = np.concatenate([self.data_table[t_idx], new_table], axis=1)
            self.data_table = np.stack(self.data_table, axis=0) # (n_sample, n_fea, seqs_len)
            with open(p_final_table, 'wb') as fp:
                pickle.dump((seqs_len, self.static_keys, self.data_table), fp)
            logger.info(f'length aligned table dumped at {p_final_table}')

    def load_version(self, version_name):
        '''更新dataset版本'''
        # 检查是否以及装入
        if self.version_name == version_name:
            return
        else:
            self.version_name = version_name
        
        p_version = os.path.join(self.gbl_conf['paths']['cache_dir'], f'8_version_{version_name}.pkl')
        assert(os.path.exists(p_version))
        with open(p_version, 'rb') as fp:
            version_dict = pickle.load(fp)
        self.static_keys = version_dict['static_keys']
        self.dynamic_keys = version_dict['dynamic_keys']
        self.total_keys = self.static_keys + self.dynamic_keys
        self.data = version_dict['data']
        self.idx_dict = version_dict['idx_dict']
        self.kf_list = version_dict['kf']

    def proprocess_version(self):
        '''
        生成不同版本的数据集, 不同版本的数据集的样本数量/特征数量都可能不同
        '''
        assert(self.idx_dict is not None)
        p_final_table = os.path.join(self.gbl_conf['paths']['cache_dir'], '7_table_final.pkl')
        with open(p_final_table, 'rb') as fp:
            seqs_len, self.static_keys, self.data_table = pickle.load(fp)
        version_params = self.loc_conf['dataset']['version']
        for version_name in version_params.keys():
            logger.info(f'Preprocessing version: {version_name}')
            # 检查是否存在pkl
            p_version = os.path.join(self.gbl_conf['paths']['cache_dir'], f'8_version_{version_name}.pkl')
            if os.path.exists(p_version):
                logger.info(f'Skip preprocess existed version: {version_name}')
                continue
            version_table = self.data_table.copy()
            # 筛选可用特征
            if len(version_params[version_name]['feature_limit']) > 0:
                limit_idx = [self.idx_dict[key] for key in version_params[version_name]['feature_limit']]
            else:
                limit_idx = list(self.idx_dict.values())
            avail_idx = []
            forbidden_idx = set([self.idx_dict[key] for key in version_params[version_name]['forbidden_feas']])
            for idx in limit_idx:
                if idx not in forbidden_idx:
                    avail_idx.append(idx)
            avail_idx = sorted(avail_idx)
            # 原本的缺失用均值填充
            for key, idx in self.idx_dict.items():
                for s_idx in range(version_table.shape[0]):
                    arr = version_table[s_idx, idx, :self.seqs_len[s_idx]]
                    version_table[s_idx, idx, :self.seqs_len[s_idx]] = np.where(np.abs(arr + 1) < 1e-4, arr, self.norm_dict[key]['mean'])
            # 更新特征
            version_table = version_table[:, avail_idx, :]
            derived_idx_dict = {}
            idx_converter = {idx:new_idx for new_idx, idx in enumerate(avail_idx)}
            avail_keys = {}
            for key, idx in self.idx_dict.items():
                avail_keys[idx] = key
                if idx in avail_idx:
                    derived_idx_dict[key] = idx_converter[idx]
            avail_keys = [avail_keys[idx] for idx in avail_idx]
            derived_static_keys = avail_keys[:np.sum(np.asarray(avail_idx) < len(self.static_keys))]
            derived_dynamic_keys = avail_keys[np.sum(np.asarray(avail_idx) < len(self.static_keys)):]
            derived_seqs_len = seqs_len
            # 设置k-fold
            kf = KFold(n_splits=GLOBAL_CONF_LOADER['analyzer']['data_container']['n_fold'], \
                shuffle=True, random_state=GLOBAL_CONF_LOADER['analyzer']['data_container']['seed'])
            kf_list = []
            for data_index, test_index in kf.split(X=list(range(version_table.shape[0]))): 
                # encode: train, valid, test
                valid_num = round(len(data_index)*0.15)
                train_index, valid_index = data_index[valid_num:], data_index[:valid_num]
                kf_list.append({'train':train_index, 'valid':valid_index, 'test':test_index})
            version_dict = {
                'static_keys':derived_static_keys,
                'dynamic_keys':derived_dynamic_keys,
                'seqs_len':derived_seqs_len,
                'idx_dict':derived_idx_dict,
                'data': version_table,
                'kf': kf_list,
            }
            with open(p_version, 'wb') as fp:
                pickle.dump(version_dict, fp)
    
    def enumerate_kf(self):
        return KFoldIterator(self, k=len(self.kf_list))

    def set_kf_index(self, kf_index):
        '''设置dataset对应K-fold的一折'''
        self.kf_index = kf_index
        self.train_index = self.kf_list[kf_index]['train']
        self.valid_index = self.kf_list[kf_index]['valid']
        self.test_index = self.kf_list[kf_index]['test']
        # self.mode('all')
        return self.train_index.copy(), self.valid_index.copy(), self.test_index.copy()

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
                table[t_idx, :] = np.clip(table[index_dict['pao2'], :] / table[index_dict['fio2'], :], 0, 500)
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

    def hit_table(self):
        '''
        生成特征在subject粒度的覆盖率
        return: dict(key=feature_id, val=cover_rate)
        '''
        if self._hit_table is None:
            hit_table = {}
            adm_count = np.sum([len(s.admissions) for s in self.subjects.values()])
            for sub in self.subjects.values():
                for adm in sub.admissions:
                    for id in adm.keys():
                        if hit_table.get(id) is None:
                            hit_table[id] = 1
                        else:
                            hit_table[id] += 1
            self._hit_table = {key:val / adm_count for key,val in hit_table.items()}
        return self._hit_table.copy()
    
    def make_report(self):
        '''进行数据集的信息统计'''
        out_path = os.path.join(self.gbl_conf['paths']['out_dir'], 'dataset_report.txt')
        dist_dir = os.path.join(self.gbl_conf['paths']['out_dir'], 'report_dist')
        dir_names = ['points', 'duration', 'frequency', 'value', 'from_sepsis', 'static_value']
        tools.reinit_dir(dist_dir, build=True)
        for name in dir_names:
            os.makedirs(os.path.join(dist_dir, name))
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
            for s in tqdm(self.subjects.values(), desc=f'id={id}'):
                for adm in s.admissions:
                    if id in adm.keys():
                        dur = adm[id][-1,1] - adm[id][0,1]
                        if dur < 0.1:
                            continue
                        sepsis_time = s.nearest_static('sepsis_time', adm[id][0, 1])
                        t_sep = adm[id][0, 1] + adm.admittime - sepsis_time
                        if np.abs(t_sep) < 72:
                            arr_from_sepsis_time.append(t_sep)
                        arr_points.append(adm[id].shape[0])
                        arr_duration.append(dur)
                        arr_frequency.append(arr_points[-1] / arr_duration[-1])
                        arr_avg_value.append(adm[id][:,0].mean())
                        assert(arr_duration[-1] > 0)
            arr_points, arr_duration, arr_frequency, arr_avg_value = np.asarray(arr_points), np.asarray(arr_duration), np.asarray(arr_frequency), np.asarray(arr_avg_value)
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
        # static feature explore
        for id in tqdm(self.static_keys, 'generate static feature report'):
            fea_name = self.get_fea_label(id)
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
                save_path=os.path.join(dist_dir, 'static_value', save_name + '.png'), discrete=False, adapt=True)

        # write hit table
        hit_table = self.hit_table()
        key_list = sorted(hit_table.keys(), key= lambda key:hit_table[key], reverse=True)
        write_lines.append('='*10 + 'Feature hit table(>0.5)' + '='*10)
        for key in key_list:
            fea_name = self.icu_item[key][0]
            value = hit_table[key]
            if value < 0.5:
                continue
            write_lines.append(f"{value:.2f}\t({key}){fea_name} ")
        # write report
        with open(out_path, 'w', encoding='utf-8') as fp:
            for line in write_lines:
                fp.write(line + '\n')
        logger.info(f'Report generated at {out_path}')

    def __getitem__(self, idx):
        assert(self.version_name is not None)
        if self.data_index is None:
            return {'data': self.data[idx, :, :], 'length': self.seqs_len[idx]}
        else:
            return {'data': self.data[self.data_index[idx], :, :], 'length': self.seqs_len[self.data_index[idx]]}

    def __len__(self):
        if self.data_index is None:
            return self.data.shape[0]
        else:
            return len(self.data_index)

class KFoldIterator:
    def __init__(self, dataset:MIMICDataset, k):
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

def load_sepsis_patients(csv_path:str) -> dict:
    '''
    从csv中读取按照mimic-iv pipeline生成的sepsis3.0表格
    sepsis_dict: dict(int(subject_id):list(element))
        element: [sepsis_time(float), stay_id, sofa_score, respiration, liver, cardiovascular, cns, renal]
        多次出现会记录多个sepsis_time
    '''
    converter = tools.TimeConverter(format="%Y-%m-%d %H:%M:%S", out_unit='hour')
    sepsis_dict = {}
    def extract_time(row): # 提取sepsis发生时间, 返回float
        return min(converter(row['antibiotic_time']), converter(row['culture_time']))
    def build_dict(row): # 提取sepsis dict
        id = row['subject_id']
        sepsis_time = row['sepsis_time']
        element = [sepsis_time, int(row['stay_id']), row['sofa_score'], \
            row['respiration'], row['liver'], row['cardiovascular'], row['cns'], row['renal']]
        if id in sepsis_dict:
            sepsis_dict[id].append(element)
        else:
            sepsis_dict[id] = [element]
        
    df = pd.read_csv(csv_path, encoding='utf-8')
    df['sepsis_time'] = df.apply(extract_time, axis=1)
    df.apply(build_dict, axis=1)
    logger.info(f'Load {len(sepsis_dict.keys())} sepsis patients based on sepsis3.csv')
    return sepsis_dict



if __name__ == '__main__':
    dataset = MIMICDataset()
    dataset.load_version('lite')
    logger.debug('Dataset Initialization Done')
    
    # dataset.make_report()
    