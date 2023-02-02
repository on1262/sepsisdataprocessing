#-*- coding: utf-8 -*-
import pandas as pd
import pandas_profiling as ppf
import json, os
import matplotlib.pyplot as plt
import numpy as np
import tools
from tools.colorful_logging import logger
import pickle
from tqdm import tqdm


class DynamicSepsisDataset():
    def __init__(self, from_pkl=False):
        self.conf_loader = tools.GLOBAL_CONF_LOADER["dataset_dynamic"]['paths']
        self.csv_path = self.conf_loader['csv_origin_path'] # origin data
        self.conf_cache_path = self.conf_loader['conf_cache_path']
        self.conf_manual_path = self.conf_loader['conf_manual_path']
        self.profile_conf_path = self.conf_loader['profile_conf_path']
        self.profile_save_path = self.conf_loader['profile_save_path']
        self.output_cleaned_path = self.conf_loader['output_cleaned_path']
        self.dataframe_save_path = self.conf_loader['dataframe_save_path']
        self.fea_manager_save_path = self.conf_loader['fea_manager_save_path']
        self.out_path = self.conf_loader['out_dir']
        self.k = None
        self.slice_dict = None
        self.target_time_dict = None

        self.fea_manager = tools.FeatureManager()

        # update combined csv
        logger.info("updating combined csv")
        if not from_pkl:
            tools.scripts_combine_and_select_samples()

        with open(self.csv_path, 'r', encoding='utf-8') as f:
            self.data_pd = pd.read_csv(f, encoding='utf-8')


        self._load_configs()
        self.target_fea = self.configs['target_fea']

        if from_pkl and os.path.exists(self.dataframe_save_path) and os.path.exists(self.fea_manager_save_path):
            if self.configs['origin_md5'] == tools.cal_file_md5(self.csv_path):
                logger.info(f'loading data from {self.dataframe_save_path}')
                self.data_pd = pd.read_pickle(self.dataframe_save_path)
                self.fea_manager = pd.read_pickle(self.fea_manager_save_path)
                self._init_time_arr()
                with open(self.conf_loader['slice_result_save_path'], 'rb') as f:
                    tmp = pickle.load(f)
                    self.k = tmp[0]
                    self.slice_dict = tmp[1]
                    self.target_time_dict = tmp[2]
                self.print_features()
                return
            else:
                logger.warning('MD5 validation failed, change from_pkl=false')
        
        self._feature_select()
        self.data_pd = tools.one_hot_decoding(
            self.data_pd, cluster_dict=self.configs['static']['one_hot_decoding'], fea_manager=self.fea_manager
        )
        self.plot_na('bar')
        # TODO 这里有个小bug, select_na在fill_default之前运行, 阈值会把还没fill的视为0
        # 但是也可以解释: 如果非0值很少, fill_default之后也没什么用
        self.data_pd = tools.select_na(self.data_pd,
            col_thres=self.configs['select_na']['1st_col_thres'],
            row_thres=self.configs['select_na']['1st_row_thres'],
            fea_manager=self.fea_manager
        )

        self.type_dict = tools.check_fea_types(self.data_pd)
        self.interval_dict = self._expand_normal_interval(self.configs['normal_interval']) # 不需要更新configs
        self.data_pd = tools.remove_invalid_rows(
            self.data_pd, self.type_dict, self.interval_dict, self.fea_manager.get_expanded_fea(self.target_fea))
        self.data_pd.reset_index(drop=True, inplace=True)

        self.category_dict = tools.detect_category_fea(self.data_pd, self.type_dict, \
            cluster_perc=self.configs['static']['cluster_perc'])
        self.configs['type_dict'] = self.type_dict
        self.configs['category_dict'] = self.category_dict

        # data cleaning
        self.data_pd = tools.apply_category_fea(self.data_pd, self.category_dict)
        tools.fill_default(self.data_pd,
            self.configs['static']['fill_default'], self.configs['dynamic']['fill_default'], self.fea_manager)
        self.data_pd = tools.select_na(self.data_pd,
            col_thres=self.configs['select_na']['2nd_col_thres'],
            row_thres=self.configs['select_na']['2nd_row_thres'],
            fea_manager=self.fea_manager
        )
        self.plot_na('matrix')
        self.plot_na('sample')
        self._plot_correlation()
        tools.plot_category_dist(data=self.data_pd, type_dict=self.type_dict,
            output_dir=os.path.join(self.out_path, 'category_dist'))
        
        self.print_features()
        
        # prepare slice dataset
        self._init_time_arr()
        time_k = 1
        slice_target_time = self._make_slice(mode='target_time', k=time_k)
        slice_k = self._make_slice(mode='k_slice', k=time_k)
        slice_k['data'].to_csv(self.conf_loader['csv_slice_path'], index=False)
        slice_target_time['data'].to_csv(self.conf_loader['csv_target_time_path'], index=False)

        with open(self.conf_loader['slice_result_save_path'], 'wb') as f:
            pickle.dump([time_k, slice_k, slice_target_time], f)

        # data should write before configs: to avoid md5 validation failure.
        self.data_pd.to_csv(self.output_cleaned_path, index=False)
        self.data_pd.to_pickle(self.dataframe_save_path)
        
        with open(self.fea_manager_save_path, 'wb') as fp:
            pickle.dump(obj=self.fea_manager, file=fp)
        self.configs['origin_md5'] = tools.cal_file_md5(self.csv_path)
        self._dump_configs()

    def _init_time_arr(self):
        # init time arr
        self.target_time_arr = tools.cal_available_time(
            data=self.data_pd,
            expanded_target=self.fea_manager.get_expanded_fea(self.target_fea)
        ) # [:,0]=start_time, [:,1]=duration
    
    # 将开始和持续时间变成整数
    def get_time_target_idx(self):
        times = [val[0] for val in self.fea_manager.get_expanded_fea(self.target_fea)]
        step = times[1] - times[0]
        start_idx = np.round(self.target_time_arr[:,0] / step).astype(np.int32)
        dur_len = np.round(self.target_time_arr[:,1] / step).astype(np.int32)
        return start_idx, dur_len
    '''
    生成动态模型所需的时间切片
    mode:
        'target_time' 适用于只看目标历史数据的方法
        'k_slice' 适用于只用T-k天预测第T天数据的方法
    '''
    def _make_slice(self, mode:str='target_time', k=None)->dict:
        if self.k is not None and k == self.k:
            logger.info(f'Load slice dataset from pkl')
            if mode == 'target_time':
                return self.target_time_dict
            elif mode == 'k_slice':
                return self.slice_dict
        else:
            logger.info('Creating slice dataset. This procedure may be time consuming.')
        start_idx, dur_len = self.get_time_target_idx()
        
        if mode == 'target_time':
            expanded = self.fea_manager.get_expanded_fea(self.target_fea)
            names = [val[1] for val in expanded]
            times = np.asarray([val[0] for val in expanded])
            result = self.data_pd[names]
            assert(start_idx.shape[0] == dur_len.shape[0] and start_idx.shape[0] == result.shape[0])
            return {'data':result, 'start_idx': start_idx, 'dur_len':dur_len} # dict{key:ndarray}
        elif mode == 'k_slice':
            assert(k is not None)
            dynamic_target_name = tools.GLOBAL_CONF_LOADER["dynamic_analyzer"]['dynamic_target_name']
            data = self.data_pd.loc[dur_len > k]
            dur_len = dur_len[dur_len > k]
            start_idx = start_idx[dur_len > k]
            data.reset_index(drop=True, inplace=True)
            sta_names = self.fea_manager.get_names(sta=True)
            dyn_names = self.fea_manager.get_names(dyn=True)
            dyn_dict = {key:[val[1] for val in self.fea_manager.get_expanded_fea(key)] for key in dyn_names} # old name
            result = pd.DataFrame(columns=sta_names + dyn_names + [dynamic_target_name]) # target_fea 包括在内
            map_table = []
            tmp = self._make_slice(mode='target_time', k=k)
            for r_idx in tqdm(range(len(data)), desc='Mapping slice dataset'):
                for delta in range(dur_len[r_idx] - k): # duration=N, k=1, delta=0,1,2,...,N-2
                    new_row = {}
                    for name in sta_names:
                        new_row[name] = data.at[r_idx, name]
                    for name in dyn_names:
                        new_row[name] = data.at[r_idx, dyn_dict[name][start_idx[r_idx] + delta]]
                    new_row[dynamic_target_name] = data.at[r_idx, dyn_dict[self.target_fea][start_idx[r_idx] + delta + k]]
                    map_table.append([r_idx, start_idx[r_idx] + delta + k])
                    result.loc[len(result)] = new_row
            map_table = np.asarray(map_table, dtype=np.int32)
            logger.info(f'Extended Datasets size={len(result)}, with {len(result.columns)} features')
            # generate new type dict
            new_type_dict = self._generate_dyn_type_dict()
            for name in result.columns:
                if name == dynamic_target_name:
                    new_type_dict[name] = float
                elif name not in dyn_dict.keys():
                    new_type_dict[name] = self.type_dict[name]
            return {'data':result, 'type_dict':new_type_dict, 'gt_table': tmp['data'], 
                'start_idx':start_idx, 'dur_len':dur_len, 'map_table':map_table}
        else:
            logger.error('make slice: unknown mode')
            assert(0)
    
    def _generate_dyn_type_dict(self) -> dict:
        dyn_names = set(self.fea_manager.get_names(dyn=True))
        result = {}
        for name in dyn_names:
            result[name] = self.type_dict[self.fea_manager.get_expanded_fea(name)[0][1]]
        return result

    def get_type_dict(self):
        return self.configs['type_dict'].copy()
    
    # 把动态特征的名字扩展到时间轴, 便于处理
    def _expand_normal_interval(self, interval_dict:dict):
        interval_dict = interval_dict.copy()
        dyn_names = set(self.fea_manager.get_names(dyn=True))
        for key in list(interval_dict.keys()):
            if key in dyn_names:
                ep_fea = [val[1] for val in self.fea_manager.get_expanded_fea(key)]
                for name in ep_fea:
                    interval_dict[name] = interval_dict[key]
                interval_dict.pop(key)
        return interval_dict
                

    def _feature_select(self):
        static_features_1 = self.configs['static']['features'].copy()
        static_features_2 = self.configs['dynamic']['invariant_features'].copy()
        dynamic_features = set(self.configs['dynamic']['variant_features'].copy())
        time_prefix= self.configs['dynamic']['time_prefix']
        prefix = self.configs['prefix'] # sta_ or dyn_
        # static features can be directly matched with column name
        static_features = set([prefix[0] + fea for fea in static_features_1] + static_features_2)
        for col in self.data_pd.columns:
            if col in static_features:
                self.fea_manager.add_sta(col)
            else:
                # match prefix
                for p in time_prefix:
                    result = tools.match_dyn_feature(col, p, dynamic_features, signs={"[day]", "[period]"})
                    if result is not None:
                        self.fea_manager.add_dyn(col, prefix[1] + result[0], float(result[1]))
                        break
        # drop unused features
        self.data_pd = self.data_pd.loc[:, self.fea_manager.get_names(sta=True, dyn=True, expand_dyn=True)]
        disp_names = self.fea_manager.get_names(sta=True, dyn=True, expand_dyn=False)
        logger.info(f'{len(disp_names)} features selected')
        logger.info('Feature selection OK')


    def _plot_correlation(self):
        tools.reinit_dir(os.path.join(self.out_path, 'correlation'))
        # 所有动态特征都展开, 构建一个新的表
        # 静态特征按照原先的方法输出
        sta_fea = self.fea_manager.get_names(sta=True)
        data_static = self.data_pd[sta_fea].copy()
        # 抽取目标值
        target_expanded = []
        for val in self.fea_manager.get_expanded_fea(self.target_fea):
                target_expanded.append(val[1])
        
        target_expanded = self.data_pd[target_expanded].to_numpy()
        # 计算最低值
        target_min = target_expanded.min(axis=1)
        # 统计数值型静态特征
        numeric_static = []
        for fea in self.fea_manager.get_names(sta=True):
            if self.configs['type_dict'][fea] != str:
                numeric_static.append(fea)
        # 静态特征输出
        logger.debug("Plotting static features' correlation")
        tools.plot_reg_correlation(
            data_static[numeric_static].to_numpy(), numeric_static, target_min, self.target_fea, restrict_area=True,
            write_dir_path=os.path.join(self.out_path, 'correlation', 'static_feature'))
        # 动态特征进行时间轴的合并
        # 按列合并, 相同时间点的所有患者紧邻
        target_expanded = target_expanded.T.reshape((target_expanded.shape[0]*target_expanded.shape[1]))
        target_time = [val[0] for val in self.fea_manager.get_expanded_fea(self.target_fea)]

        dyn_names = self.fea_manager.get_names(dyn=True)
        len_rows = len(self.data_pd)
        assert(self.target_fea in dyn_names)
        dyn_names.remove(self.target_fea) # 去除target本身

        dyn_expanded = self.data_pd.loc[:, self.fea_manager.get_names(dyn=True, expand_dyn=True)]
        dyn_matrix = np.empty((len_rows*len(target_time), len(dyn_names)))
        for idx, t in enumerate(target_time):
            for c_idx, dyn_name in enumerate(dyn_names):
                nearest_fea = self.fea_manager.get_nearest_fea(dyn_name, t)
                dyn_matrix[idx*len_rows:(idx+1)*len_rows, c_idx] = \
                    dyn_expanded.loc[:, nearest_fea].to_numpy()
        logger.debug("Plotting dynamic features' correlation")
        tools.plot_reg_correlation(
            dyn_matrix, dyn_names, target_expanded, self.target_fea, restrict_area=True,
            write_dir_path=os.path.join(self.out_path, 'correlation', 'dynamic_feature'))
        logger.info("Plot correlation: Done")


    def plot_na(self, mode='matrix', disp=False):
        tools.plot_na(data=self.data_pd, save_path=os.path.join(self.out_path, f'missing_{mode}.png'), mode=mode, disp=disp)

    def print_features(self):
        sta_names = self.fea_manager.get_names(sta=True)
        dyn_names = self.fea_manager.get_names(dyn=True)
        sta_all = len(self.configs['static']['features']) + len(self.configs['dynamic']['invariant_features'])
        dyn_all = len(self.configs['dynamic']['variant_features'])
        logger.info("="*10 + f" Used {len(sta_names)}/{sta_all} Static Features " + "="*10)
        logger.info(str(sta_names))
        logger.info("="*10 + f" Used {len(dyn_names)}/{dyn_all} Dynamic Features " + "="*10)
        logger.info(str(dyn_names))
    
    def profile(self):
        profile = ppf.profile_report.ProfileReport(
            df = self.data_pd,
            # config_file=self.profile_conf_path
        )
        profile.to_file(self.profile_save_path)
    
    def _dump_configs(self):
        # some item needs deep copy, because type_dict of self.configs will change if using shallow copy
        # It is not needed when items not change in dumping procedure.
        configs = {}
        for key in self.configs.keys():
            if key == 'one_hot_decoding':
                configs['one_hot_decoding'] = {}
                for name in self.configs['one_hot_decoding'].keys():
                    configs['one_hot_decoding'][name] = list(self.configs['one_hot_decoding'][name])
            elif key == 'type_dict':
                configs['type_dict'] = {}
                for name in self.configs['type_dict'].keys():
                    configs['type_dict'][name] = self.configs['type_dict'][name].__name__
            else:
                configs[key] = self.configs[key]

        with open(self.conf_cache_path, 'w', encoding='utf-8') as f:
            json.dump(configs, f, ensure_ascii=False) # allow utf-8 characters not converted into \uXXX
        logger.debug(f'Data configs dumped at {self.conf_cache_path}')


    def _load_configs(self):
        try:
            with open(self.conf_cache_path,'r', encoding='utf-8') as f:
                self.configs = json.load(f)
        except Exception as e:
            logger.error(f'open json error: {self.conf_cache_path}')
            self.configs = {}
        try:
            with open(self.conf_manual_path, 'r', encoding='utf-8') as fc:
                manual_dict = json.load(fc)
                for key in manual_dict.keys():
                    self.configs[key] = manual_dict[key] # 手动指定的内容覆盖自动生成的内容
            if self.configs.get('type_dict') is not None:
                for name in self.configs['type_dict'].keys():
                    self.configs['type_dict'][name] = eval(self.configs['type_dict'][name])
            if self.configs.get('one_hot_decoding') is not None:
                for name in self.configs['one_hot_decoding'].keys():
                    self.configs['one_hot_decoding'][name] = list(self.configs['one_hot_decoding'][name])
        except Exception as e:
            logger.error('Config load error')
            self.configs = {}


if __name__ == '__main__':
    tools.set_chinese_font()
    dataset = DynamicSepsisDataset(from_pkl=False)
