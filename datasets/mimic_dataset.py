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
import math


class Admission:
    '''
    代表一段连续的、环境较稳定的住院经历，原subject/admission/stay/transfer的四级结构被精简到subject/admission的二级结构
    label: 代表急诊室数据或ICU数据
    admittime: 起始时间
    dischtime: 结束时间
    '''
    def __init__(self, unique_id:int, admittime:float, dischtime:float, label=['ed', 'icu']) -> None:
        self.dynamic_data = {} # dict(fea_name:ndarray(value, time))
        assert(admittime < dischtime)
        self.label = label # ed or icu
        self.unique_id = unique_id # hadm_id+stay_id or hadm_id+transfer_id, 16 digits
        self.admittime = admittime
        self.dischtime = dischtime
        self.data_updated = False
    
    def append_dynamic(self, itemid, time:float, value):
        assert(not self.data_updated)
        if self.dynamic_data.get(itemid) is None:
            self.dynamic_data[itemid] = [(value, time)]
        else:
            self.dynamic_data[itemid].append((value, time))

    def update_data(self):
        '''绝对时间变为相对时间，更改动态特征的格式'''
        if not self.data_updated:
            self.data_updated = True
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
    static data: dict(feature name: (value, charttime))
    dyanmic data: admissions->(id, chart time, value)
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
    
    def nearest_static(self, key, time=None):
        '''返回与输入时间最接近的静态特征取值'''
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
        '''添加一个动态特征到合适的admission中'''
        for adm in self.admissions: # search admission by charttime
            if adm.admittime < charttime and charttime < adm.dischtime:
                adm.append_dynamic(itemid, charttime, value)

    def update_data(self):
        '''将数据整理成连续形式'''
        for adm in self.admissions:
            adm.update_data()

    def del_empty_admission(self):
        # 删除空的admission
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
        self.ymdhms_convertor = tools.TimeConverter(format="%Y-%m-%d %H:%M:%S", out_unit='hour')
        self.ymd_convertor = tools.TimeConverter(format="%Y-%m-%d", out_unit='hour')
        self.align_key_ids = self.loc_conf['dataset']['alignment_key_id']
        # self.report_ids = self.loc_conf['dataset']['make_report']

        # variable for phase 1
        self.sepsis_result = None
        self.hosp_item = None
        self.icu_item = None
        # variable for phase 2
        self.subjects = {} # subject_id:Subject

        self.preprocess()
        # post process
        self.global_missrate = self._global_miss_rate()
        self.remove_invalid_data(rule=self.loc_conf['dataset']['remove_rule']['version2'])
        logger.info('MIMICIV inited')

    def preprocess(self, from_pkl=True):
        if not os.path.exists(self.gbl_conf['paths']['cache_dir']):
            tools.reinit_dir(self.gbl_conf['paths']['cache_dir'], build=True)
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
        sepsis_patient_path = self.gbl_conf['paths']['sepsis_patient_path']
        sepsis_result = load_sepsis_patients(sepsis_patient_path)
        # 建立hospital lab_item编号映射
        d_hosp_item = pd.read_csv(os.path.join(self.mimic_dir, 'hosp', 'd_labitems.csv'), encoding='utf-8')
        hosp_item = {}
        for _,row in tqdm(d_hosp_item.iterrows(), desc='hosp items'):
            hosp_item[str(row['itemid'])] = (row['label'], row['fluid'], row['category'])
        # 建立icu lab_item编号映射
        d_icu_item = pd.read_csv(os.path.join(self.mimic_dir, 'icu', 'd_items.csv'), encoding='utf-8')
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
                sepsis time被看作一个静态特征添加到subject下, 一个subject可以有多个sepsis time, 这里假设sepsis time都被stay覆盖
                如果一个admission没有sepsis time对应, 那么这个admission无效
                在最终的三维数据上, sepsis_time会变为距离起始点t_start的相对值(sep-t_start)
                由于起始点设为max(sepsis, t_start), 所以sepsis_time只会是负数或者0
                当sepsis_time<0的时候, 表明sepsis发生得早, 对于一些模型, sepsis time不能太小, 可以用来筛选数据
                '''
                for ele_dict in self.sepsis_result[s_id]: # dict(list(dict))
                    sepsis_time = ele_dict['sepsis_time']

                    self.subjects[s_id] = Subject(row['subject_id'], anchor_year=row['anchor_year'])
                    # self.subjects[s_id].append_static(sepsis_time, 'age', row['anchor_age']) 每次入院的年龄是有可能变化的
                    self.subjects[s_id].append_static(sepsis_time, 'gender', row['gender'])

                    if row['dod'] is not None and isinstance(row['dod'], str):
                        self.subjects[s_id].append_static(sepsis_time, 'dod', self.ymd_convertor(row['dod']))

                    self.subjects[s_id].append_static(sepsis_time, 'sepsis_time', sepsis_time)
                    self.subjects[s_id].append_static(sepsis_time, 'sofa_score', ele_dict['sofa_score'])
                    self.subjects[s_id].append_static(sepsis_time, 'respiration', ele_dict['respiration'])
                    self.subjects[s_id].append_static(sepsis_time, 'liver', ele_dict['liver'])
                    self.subjects[s_id].append_static(sepsis_time, 'cardiovascular', ele_dict['cardiovascular'])
                    self.subjects[s_id].append_static(sepsis_time, 'cns', ele_dict['cns'])
                    self.subjects[s_id].append_static(sepsis_time, 'renal', ele_dict['renal'])
        # 更新sepsis_result, 去除不存在的s_id
        logger.info(f'Extract {len(self.subjects)} from {len(self.sepsis_result)} patients in sepsis3 results')
        self.sepsis_result = {key:self.sepsis_result[key] for key in self.subjects.keys()}
        # 从icu stays中抽取stay id
        table_icustays = pd.read_csv(os.path.join(self.mimic_dir, 'icu', 'icustays.csv'), encoding='utf-8')
        for _,row in tqdm(table_icustays.iterrows(), desc='extract admission from ICU', total=len(table_icustays)):
            s_id = int(row['subject_id'])
            if s_id not in self.subjects:
                continue
            adm = Admission(
                unique_id=int(row['hadm_id']*1e8+row['stay_id']),
                admittime=self.ymdhms_convertor(row['intime']), 
                dischtime=self.ymdhms_convertor(row['outtime']),
                label='icu',
            )
            self.subjects[s_id].append_admission(adm)
        del table_icustays
        # 从transfer中抽取Emergency Department的transfer
        table_transfer = pd.read_csv(os.path.join(self.mimic_dir, 'hosp', 'transfers.csv'), encoding='utf-8')
        for _,row in tqdm(table_transfer.iterrows(), desc='extract admission from ED', total=len(table_transfer)):
            s_id = int(row['subject_id'])
            if row['careunit'] != 'Emergency Department' or s_id not in self.subjects:
                continue
            else:
                if not np.isnan(row['hadm_id']):
                    unique_id = int(row['hadm_id']*1e8+row['transfer_id'])
                else:
                    unique_id = int(row['transfer_id']*1e8+row['transfer_id']) # transfer中某些情况没有分配admission
                adm = Admission(
                    unique_id=unique_id,
                    admittime=self.ymdhms_convertor(row['intime']), 
                    dischtime=self.ymdhms_convertor(row['outtime']),
                    label='ed',
                )
                self.subjects[s_id].append_admission(adm)
        # 若要补充admission表中的有效信息，如insurance、race，从这里插入

        # 患者的基本信息，如身高、体重、血压
        omr = pd.read_csv(os.path.join(self.mimic_dir, 'hosp', 'omr.csv'), encoding='utf-8') # [subject_id,chartdate,seq_num,result_name,result_value]
        omr = omr.to_numpy()
        for idx in tqdm(range(omr.shape[0]), 'extract omr'):
            s_id = omr[idx, 0]
            # TODO 是否需要seq num>1的数据？
            if s_id in self.subjects and int(omr[idx, 2]) == 1: # sequence num=1, fist element only
                self.subjects[s_id].append_static(self.ymd_convertor(omr[idx, 1]), omr[idx, 3], omr[idx, 4])
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

        collect_id_set = set() # 只有数值型的特征是需要捕捉的
        for id, data in self.icu_item['id'].items():
            if data['type'] in ['Numeric', 'Numeric with tag'] and data['category'] != 'Alarms':
                collect_id_set.add(id)
        # 采集icu内的动态数据
        out_cache_dir = os.path.join(self.gbl_conf['paths']['cache_dir'], 'icu_events')
        if not os.path.exists(out_cache_dir):
            logger.info(f'Can not find split cache csvs. Run spliting function')
            tools.split_csv(os.path.join(self.mimic_dir, 'icu', 'chartevents.csv'), out_folder=out_cache_dir)
        icu_events = None
        logger.info('Loading icu events')
        p_bar = tqdm(total=len(os.listdir(out_cache_dir)))
        for file_name in sorted(os.listdir(out_cache_dir)):
            icu_events = pd.read_csv(os.path.join(out_cache_dir, file_name), encoding='utf-8')[['subject_id', 'itemid', 'charttime', 'valuenum']].to_numpy()
            for idx in range(len(icu_events)):
                s_id, itemid = icu_events[idx, 0], str(icu_events[idx, 1])
                if s_id in self.subjects and itemid in collect_id_set:
                    self.subjects[s_id].append_dynamic(charttime=self.ymdhms_convertor(icu_events[idx, 2]), itemid=itemid, value=icu_events[idx, 3])
            p_bar.update(1)
        # 整理admissions的格式
        for s_id in tqdm(self.subjects, desc='update data'):
            self.subjects[s_id].update_data()
        # 删去空的subject和空的admission
        self.remove_invalid_data(rule=self.loc_conf['dataset']['remove_rule']['version1'])
        # 保存subjects
        logger.info('Dump subjects')
        with open(pkl_path, 'wb') as fp:
            pickle.dump(self.subjects, fp)
        logger.info('Dump subjects: Done')
        self.procedure_flag = 'phase3'

    def _global_miss_rate(self) -> dict:
        '''
        返回dict(key:value)其中key是str(id), value是该特征对应的列缺失率
        NOTE: miss rate和hit table的区别: miss rate在MIMIC-IV执行remove invalid subjects之前进行,
        基于所有sepsis患者计算缺失率, 而hit table基于筛选后的患者计算缺失率.
        '''
        miss_dict = {}
        count = 0
        for s_id in self.subjects:
            s = self.subjects[s_id]
            if not s.empty():
                for key in s.static_data.keys():
                    if key not in miss_dict:
                        miss_dict[key] = 1
                    else:
                        miss_dict[key] += 1
                count += 1
                for adm in s.admissions:
                    for key in adm.keys():
                        if key not in miss_dict:
                            miss_dict[key] = 1
                        else:
                            miss_dict[key] += 1
        miss_dict = {key:1-val/count for key, val in miss_dict.items()}
        return miss_dict

    def remove_invalid_data(self, rule:dict):
        '''按照传入的配置去除无效特征'''
        for s_id in self.subjects:
            if not self.subjects[s_id].empty():
                retain_adms = []
                for idx, adm in enumerate(self.subjects[s_id].admissions):
                    flag = 1
                    # 检查duration, points, interval
                    if 'target_id' in rule and flag != 0 and len(rule['target_id']) > 0:
                        for target_id in rule['target_id']:
                            if target_id not in adm.keys():
                                flag = 0
                        if flag == 0:
                            continue
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
                        if 'check_sepsis_time' in rule:
                            # 检查sepsis time, 必须要有一个sepsis time和这次admission对应上
                            st_min, st_max = rule['check_sepsis_time']
                            sepsis_time = self.subjects[s_id].nearest_static('sepsis_time', adm.admittime+start_time)[0][0]
                            if not (st_min < adm.admittime+start_time-sepsis_time < st_max):
                                continue
                    
                    retain_adms.append(idx)
                self.subjects[s_id].admissions = [self.subjects[s_id].admissions[idx] for idx in retain_adms]
        if 'remove_empty' in rule: # 删除空的admission和subjects
            pop_list = []
            for s_id in self.subjects:
                if self.subjects[s_id].empty():
                    pop_list.append(s_id)
                else:
                    self.subjects[s_id].del_empty_admission()
            for s_id in pop_list:
                self.subjects.pop(s_id)
        else:
            pop_list = []
        logger.info(f'del_invalid_subjects: Deleted {len(pop_list)}/{len(pop_list)+len(self.subjects)} subjects')


class MIMICIVDataset(Dataset):
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
        self.additional_feas = self.loc_conf['dataset']['additional_features']
        # hit table
        self._hit_table = None
        # miss_table
        self._global_misstable = None
        # preload data
        self.data_table = None # to derive other versions
        self.data = None # ndarray(samples, n_fea, ticks)
        self.norm_dict = None # key=str(name/id) value={'mean':mean, 'std':std}
        self.static_keys = None # list(str)
        self.dynamic_keys = None # list(str)
        self.total_keys = None
        self.seqs_len = None # list(available_len)
        self.idx_dict = None
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

    def _update_miss_table(self):
        '''
        将特征工程构造的特征与原始特征联系, 得到缺失情况. 该方法对整个数据集有效，不受样本筛选的影响
        NOTE 从subject计算得到的缺失值并不包括新增特征
        '''
        construct_dict = GLOBAL_CONF_LOADER["dataset"]['mimic-iv']['miss_dict']
        miss_dict = self.mimiciv.global_missrate
        for fea_id, _ in self.idx_dict.items():
            if fea_id not in miss_dict.keys() and fea_id in construct_dict.keys():
                miss_dict[fea_id] = np.max([miss_dict[f] for f in construct_dict[fea_id]])
        result = [(idx, miss_dict[id]) for id, idx in self.idx_dict.items()]
        result = sorted(result, key=lambda x:x[0])
        self._global_misstable = np.asarray([r[1] for r in result])

    def global_misstable(self):
        return self._global_misstable

    def get_fea_label(self, key_or_idx):
        '''输入key/idx得到关于特征的简短描述, 从icu_item中提取'''
        if isinstance(key_or_idx, int):
            name = self.total_keys[key_or_idx]
        else:
            name = key_or_idx
        if self.mimiciv.icu_item['label'].get(name) is not None:
            return self.mimiciv.icu_item['label'][name]['label']
        elif self.mimiciv.icu_item['id'].get(name) is not None:
            return self.mimiciv.icu_item['id'][name]['label']
        else:
            logger.warning(f'No fea label for {name}, return name')
            return name

    def label2id(self, label):
        result = self.mimiciv.icu_item['label'].get(label)
        if result is not None:
            return result['id']
        else:
            logger.warning(f"label is not in icu item: {label}")
            return None
    
    def id2label(self, id):
        result = self.mimiciv.icu_item['id'].get(id)
        if result is not None:
            return result['label']
        else:
            logger.warning(f"id is not in icu item: {id}")
            return None
    
    def get_id_and_label(self, id_or_label:str):
        if id_or_label in self.mimiciv.icu_item['id']:
            return id_or_label, self.id2label(id_or_label)
        elif id_or_label in self.mimiciv.icu_item['label']:
            return self.label2id(id_or_label), id_or_label
        else:
            return None, None
            
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

    def _available_dyn_id(self, min_cover_rate) -> list:
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
        # 这一步按照hit table对特征进行筛选
        self.dynamic_ids = self._available_dyn_id(min_cover_rate=self.loc_conf['dataset']['remove_rule']['min_cover_rate'])
        logger.info(f'Detected {len(self.dynamic_ids)} available dynamic features')

        # 将所有记录变成数值型
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
        
        for s in self.subjects.values(): # 添加静态特征的key
            for key in s.static_data.keys():
                self.static_feas.add(key)
        
        # 得到每个特征的归一化参数
        if from_pkl and os.path.exists(p_norm_dict):
            with open(p_norm_dict, 'rb') as fp:
                self.norm_dict = pickle.load(fp)
            logger.info(f'Load norm dict from {p_norm_dict}')
        else:
            self.norm_dict = self.preprocess_norm()
            with open(p_norm_dict, 'wb') as fp:
                pickle.dump(self.norm_dict, fp)
            logger.info(f'Norm dict dumped at {p_norm_dict}')
        
        # 将数据集进行线性插值，得到3维矩阵
        self.preprocess_table(from_pkl=from_pkl)

        # 不同版本的数据集需要进行不同的初始化
        self.static_feas, self.dynamic_ids = None, None
        self.icu_item = self.mimiciv.icu_item
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
                    # female: 0, male: 1
                    s.static_data[key] = 0 if s.static_data[key][0][0] == 'F' else 1
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
            
        # 进行特征的上下界约束
        value_clip = self.loc_conf['dataset']['value_clip']

            
        for id_or_label in value_clip:
            id, label = self.get_id_and_label(id_or_label)
            clip_count = 0
            for s in self.subjects.values():
                for adm in s.admissions:
                    if id in adm.keys():
                        data = adm[id][:, 0]
                        adm[id][:, 0] = np.clip(data, a_min=value_clip[id_or_label]['min'], a_max=value_clip[id_or_label]['max'])
                        clip_count += 1
            logger.info(f'Value Clipping: clip {label} in {clip_count} admissions')
            
            # PaO2 异常峰检测进行硬平滑
            # logger.info('Anomaly peak detection')
            # for adm in s.admissions:
            #     if "220224" in adm.keys():
            #         adm["220224"][:, 0] = reduce_peak(adm["220224"][:, 0])
    
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
            if isinstance(norm_dict[key][0], (float, int)): # no time stamp
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
        '''
        对每个subject生成时间轴对齐的表, tick(hour)是生成的表的间隔
        origin_table:  list(ndarray(n_fea, lens)) 其中lens不对齐, 缺失值为-1
        '''
        p_origin_table = os.path.join(self.gbl_conf['paths']['cache_dir'], '6_table_origin.pkl')
        p_final_table = os.path.join(self.gbl_conf['paths']['cache_dir'], '7_table_final.pkl')

        # step1: 插值并生成表格
        if from_pkl and os.path.exists(p_origin_table):
            with open(p_origin_table, 'rb') as fp:
                self.data_table, self.norm_dict, self.static_keys, self.dynamic_keys = pickle.load(fp)
            logger.info(f'load original aligned table from {p_origin_table}')
        else:
            data_table = []
            align_ids = [self.get_id_and_label(id_or_label)[0] for id_or_label in self.loc_conf['dataset']['alignment_key_id']] # 用来确认对齐的基准时间
            static_keys = list(self.static_feas)
            dynamic_keys = self.dynamic_ids + self.additional_feas
            # 添加所需feature对应的id
            index_dict = dict( # idx_dict: fea_key->idx
            {str(key):val for val, key in enumerate(static_keys)}, \
                **{str(key):val+len(static_keys) for val, key in enumerate(dynamic_keys)})
            
            additional_vals = {key:[] for key in self.additional_feas} # 记录特征工程生成的特征, 用于更新norm_dict
            for s_id in tqdm(self.subjects.keys(), desc='Generate aligned table'):
                for adm in self.subjects[s_id].admissions:
                    # skip bad admission
                    skip_flag = False
                    for align_id in align_ids:
                        if align_id not in adm.keys():
                            logger.warning('Invalid admission: no align id')
                            skip_flag = True
                            break
                    if skip_flag:
                        continue
                    
                    t_start, t_end = None, None
                    for id in self.loc_conf['dataset']['alignment_key_id']:
                        t_start = max(adm[id][0,1], t_start) if t_start is not None else adm[id][0,1]
                        t_end = min(adm[id][-1,1], t_end) if t_end is not None else adm[id][-1,1]
                    # sepsis_time = self.subjects[s_id].nearest_static('sepsis_time', adm.admittime) - t_start
                    ticks = np.arange(t_start, t_end, t_step) # 最后一个会确保间隔不变且小于t_end
                    # 生成表本身, 缺失值为-1
                    table = -np.ones((len(static_keys) + len(dynamic_keys), ticks.shape[0]), dtype=np.float32)
                    # 填充static data, 找最近的点
                    static_data = np.zeros((len(static_keys)))
                    for idx, key in enumerate(static_keys):
                        static_data[idx] = self.subjects[s_id].nearest_static(key, adm.admittime)
                        if key == 'sepsis_time' or key == 'dod': # sepsis time 基准变为表格的起始点
                            static_data[idx] = static_data[idx] - t_start
                    table[:len(static_keys), :] = static_data[:, None]
                    # 插值dynamic data
                    for idx, key in enumerate(dynamic_keys[:-1]):
                        if key not in adm.keys():
                            continue
                        table[static_data.shape[0]+idx, :] = np.interp(x=ticks, xp=adm[key][:, 1], fp=adm[key][:, 0])
                    
                    self._feature_engineering(table, index_dict, self.additional_feas, adm.label) # 特征工程
                    for idx, key in enumerate(reversed(self.additional_feas)): # 记录附加信息
                        additional_vals[key].append(table[-(idx+1), :])
                    data_table.append(table)
            # 计算特征工程新增特征的norm_dict
            additional_vals = {key:np.concatenate(val, axis=0) for key, val in additional_vals.items()}
            additional_vals = {key:{'mean':val.mean(), 'std':val.std()} for key,val in additional_vals.items()}
            for key in additional_vals:
                self.norm_dict[key] = additional_vals[key]
            # 存储数据
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
                padding = -np.ones((n_fea, max_len - seqs_len[t_idx]))
                self.data_table[t_idx] = np.concatenate([self.data_table[t_idx], padding], axis=1)
            self.data_table = np.stack(self.data_table, axis=0) # (n_sample, n_fea, seqs_len)
            with open(p_final_table, 'wb') as fp:
                pickle.dump((seqs_len, self.static_keys, self.data_table), fp)
            logger.info(f'length aligned table dumped at {p_final_table}')

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
        self.static_keys = version_dict['static_keys']
        self.dynamic_keys = version_dict['dynamic_keys']
        self.total_keys = self.static_keys + self.dynamic_keys
        self.seqs_len = version_dict['seqs_len']
        self.data = version_dict['data']
        self.idx_dict = version_dict['idx_dict']
        self.kf_list = version_dict['kf']
        self._update_miss_table()

    def proprocess_version(self):
        '''
        生成不同版本的数据集, 不同版本的数据集的样本数量/特征数量都可能不同
        '''
        assert(self.idx_dict is not None)
        p_final_table = os.path.join(self.gbl_conf['paths']['cache_dir'], '7_table_final.pkl')
        with open(p_final_table, 'rb') as fp:
            seqs_len, self.static_keys, self.data_table = pickle.load(fp)
        version_conf = self.loc_conf['dataset']['version']
        for version_name in version_conf.keys():
            logger.info(f'Preprocessing version: {version_name}')
            # 检查是否存在pkl
            p_version = os.path.join(self.gbl_conf['paths']['cache_dir'], f'8_version_{version_name}.pkl')
            if os.path.exists(p_version):
                logger.info(f'Skip preprocess existed version: {version_name}')
                continue
            version_table = self.data_table.copy()
            # 筛选可用样本
            if 'data_source' in version_conf[version_name]:
                data_source_parts = []
                sources = version_conf[version_name]['data_source']
                for source in sources:
                    flag = {'icu':1, 'ed':2}[source]
                    data_source_parts.append(
                        version_table[version_table[:, self.idx_dict['data_source'], 0] == flag]
                    )
                version_table = np.concatenate(data_source_parts, axis=0) if len(data_source_parts) > 1 else data_source_parts[0]
            # 筛选可用特征
            if len(version_conf[version_name]['feature_limit']) > 0:
                limit_idx = []
                for lfea in version_conf[version_name]['feature_limit']:
                    if lfea in self.idx_dict:
                        limit_idx.append(self.idx_dict[lfea])
            else:
                limit_idx = list(self.idx_dict.values())
            avail_idx = []
            forbidden_idx = set()
            for ffea in version_conf[version_name]['forbidden_feas']:
                if ffea in self.idx_dict:
                    forbidden_idx.add(self.idx_dict[ffea])
            
            for idx in limit_idx:
                if idx not in forbidden_idx:
                    avail_idx.append(idx)
            
            if version_conf[version_name].get('fill_missvalue') == 'avg': # 填充缺失值
                for key, idx in self.idx_dict.items():
                    for s_idx in range(version_table.shape[0]):
                        arr = version_table[s_idx, idx, :seqs_len[s_idx]]
                        version_table[s_idx, idx, :seqs_len[s_idx]] = np.where(np.abs(arr + 1) > 1e-4, arr, self.norm_dict[key]['mean'])
            # 更新特征
            version_table = version_table[:, avail_idx, :] # 这里不sort是为了保证PF ratio处于最后一位
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

    def _feature_engineering(self, table:np.ndarray, index_dict:dict, addi_feas:list, data_source:str):
        '''
        特征工程, 增加某些计算得到的特征
        table: (n_feas, ticks) 单个subject的table
        index_dict: dict(str:int(index)) 特征名字对应的位置, 已经加上了len(static_keys)
        addi_feas: list(str) 需要计算的特征名字
            默认addi_feas在table的末尾, 载入的顺序和addi_feas相同
        data_source: admission来自于icu或ed
            1: icu, 2: ed
        '''
        for idx, name in enumerate(reversed(addi_feas)):
            t_idx = -(idx+1)
            if name == 'PF_ratio':
                table[t_idx, :] = np.clip(table[index_dict['220224'], :] / (table[index_dict['223835'], :]*0.01), 0, 500)
            elif name == 'shock_index':
                if np.all(table[index_dict['systolic pressure'], :] > 0):
                    table[t_idx, :] = table[index_dict['220045'], :] / table[index_dict['systolic pressure'], :]
                else:
                    table[t_idx, :] = 0
                    logger.warning('feature_engineering: skip shock_index with zero sbp')
            elif name == 'MAP':
                table[t_idx, :] = (table[index_dict['systolic pressure'], :] + 2*table[index_dict['diastolic pressure'], :]) / 3
            elif name == 'PPD':
                table[t_idx, :] = table[index_dict['systolic pressure'], :] - table[index_dict['diastolic pressure'], :]
            elif name == 'data_source':
                table[t_idx, :] = 1 if data_source == 'icu' else 2
            else:
                logger.error(f'Invalid feature name:{name}')
                assert(0)

    def hit_table(self):
        '''
        生成特征在admission粒度的覆盖率
        return: dict(key=feature_id, val=cover_rate)
        '''
        if self._hit_table is None:
            hit_table = {}
            adm_count = np.sum([len(s.admissions) for s in self.mimiciv.subjects.values()])
            for sub in self.mimiciv.subjects.values():
                for adm in sub.admissions:
                    for id in adm.keys():
                        if hit_table.get(id) is None:
                            hit_table[id] = 1
                        else:
                            hit_table[id] += 1
            self._hit_table = {key:val / adm_count for key,val in hit_table.items()}
        return self._hit_table.copy()
    
    def make_report(self, version_name, params:dict):
        '''进行数据集的信息统计'''
        # switch version
        self.load_version(version_name)
        self.mode('all')
        out_path = os.path.join(self.gbl_conf['paths']['out_dir'], f'dataset_report_{version_name}.txt')
        dist_dir = os.path.join(self.gbl_conf['paths']['out_dir'], 'report_dist')
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
                for s in tqdm(self.subjects.values(), desc=f'id={id}'):
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
                static_data = self.data[:, idx, 0]
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
    def __init__(self, dataset:MIMICIVDataset, k):
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
    sepsis_dict: dict(int(subject_id):list(occur count, elements))
        element: [sepsis_time(float), stay_id, sofa_score, respiration, liver, cardiovascular, cns, renal]
        多次出现会记录多个sepsis_time
    '''
    converter = tools.TimeConverter(format="%Y-%m-%d %H:%M:%S", out_unit='hour')
    sepsis_dict = {}

    def extract_time(row): # 提取sepsis发生时间, 返回float，注意这里包含了对sepsis发生时间的定义
        return min(converter(row['antibiotic_time']), converter(row['culture_time']))
    
    def build_dict(row): # 提取sepsis dict
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
    return sepsis_dict

if __name__ == '__main__':
    dataset = MIMICIVDataset()
    