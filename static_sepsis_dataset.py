#-*- coding: utf-8 -*-
import pandas as pd
import pandas_profiling as ppf
import json, os
import matplotlib.pyplot as plt
import numpy as np
import tools
from tools.colorful_logging import logger



class StaticSepsisDataset():
    def __init__(self, from_pkl=False):
        self.conf_loader = tools.GLOBAL_CONF_LOADER["dataset_static"]['paths']
        self.csv_path = self.conf_loader['csv_origin_path'] # origin data
        self.conf_cache_path = self.conf_loader['conf_cache_path']
        self.conf_manual_path = self.conf_loader['conf_manual_path']
        self.profile_conf_path = self.conf_loader['profile_conf_path']
        self.profile_save_path = self.conf_loader['profile_save_path']
        self.output_cleaned_path = self.conf_loader['output_cleaned_path']
        self.apriori_df_path = self.conf_loader['apriori_df_path']
        self.dataframe_save_path = self.conf_loader['dataframe_save_path']
        self.out_path = self.conf_loader['out_dir']

        with open(self.csv_path, 'r', encoding='utf-8') as f:
            self.data_pd = pd.read_csv(f, encoding='utf-8')

        self.load_configs()
        self.target_fea = 'ARDS'

        if from_pkl and os.path.exists(self.dataframe_save_path) and os.path.exists(self.apriori_df_path):
            if self.configs['origin_md5'] == tools.cal_file_md5(self.csv_path):
                logger.info(f'loading data from {self.dataframe_save_path}')
                self.data_pd = pd.read_pickle(self.dataframe_save_path)
                self.apriori_pd = pd.read_pickle(self.apriori_df_path)
                return
            else:
                logger.warning('MD5 validation failed, change from_pkl=false')

        self.create_death_label()
        self.feature_check()
        self.data_pd = tools.one_hot_decoding(self.data_pd, cluster_dict=self.configs['one_hot_decoding'])
        self.plot_na('bar')
        self.data_pd = tools.select_na(self.data_pd,
            col_thres=self.configs['select_na']['1st_col_thres'], 
            row_thres=self.configs['select_na']['1st_row_thres'])

        self.type_dict = tools.check_fea_types(self.data_pd)
        self.data_pd = tools.remove_invalid_rows(self.data_pd, self.type_dict)
        self.data_pd.reset_index(drop=True, inplace=True)
        self.apriori_pd = self.data_pd.copy(deep=True)

        self.category_dict = tools.detect_category_fea(self.data_pd, self.type_dict, cluster_perc=self.configs['cluster_perc'])
        self.configs['type_dict'] = self.type_dict
        self.configs['category_dict'] = self.category_dict

        # data cleaning
        self.data_pd = tools.apply_category_fea(self.data_pd, self.category_dict)
        tools.fill_default(self.data_pd, self.configs['fill_default'])
        self.data_pd = tools.select_na(self.data_pd,
            col_thres=self.configs['select_na']['2nd_col_thres'],
            row_thres=self.configs['select_na']['2nd_row_thres'])
        self.plot_na('matrix')
        self.plot_na('sample')
        self.plot_correlation()
        tools.plot_category_dist(data=self.data_pd, type_dict=self.type_dict,
            output_dir=os.path.join(self.out_path, 'category_dist'))
        

        # data should write before configs: to avoid md5 validation failure.
        self.data_pd.to_csv(self.output_cleaned_path, index=False)
        self.data_pd.to_pickle(self.dataframe_save_path)
        self.apriori_pd.to_pickle(self.apriori_df_path)
        self.configs['origin_md5'] = tools.cal_file_md5(self.csv_path)
        self.dump_configs()


        
    def get_numeric_feas(self):
        fea_list = []
        idx_list = []
        for idx, col in enumerate(self.data_pd.columns):
            if (self.configs['type_dict'][col] != str) \
                and (col != self.target_fea) and col != self.get_death_label():
                fea_list.append(col)
                idx_list.append(idx)
        return fea_list, idx_list

        
    def get_category_feas(self):
        fea_list = []
        idx_list = []
        for idx, col in enumerate(self.data_pd.columns):
            if self.configs['type_dict'][col] == str:
                fea_list.append(col)
                idx_list.append(idx)
        return fea_list, idx_list

    def get_fea_names(self):
        return None

    def get_type_dict(self):
        return self.configs['type_dict'].copy()
    
    def get_death_label(self):
        return 'death_label'
    
    def create_death_label(self):
        death_col = [0 for _ in range(len(self.data_pd))]
        cols = self.data_pd[self.configs['death_label']]
        for idx in range(len(self.data_pd)):
            row = cols.loc[idx].to_numpy(dtype=bool)
            if any(row):
                death_col[idx] = 1
        self.data_pd['death_label'] = death_col

    def feature_check(self):
        basic_fea = []
        days_fea = {idx:[] for idx in range(1, 15,1)}
        drop_days_fea = []
        for col in self.data_pd.columns:
            n_str = str(col).split('D')[-1]
            if 'D' in col and n_str.isdigit():
                days_fea[int(n_str)].append(str(col)[:len(col)-len(n_str)-1])
                drop_days_fea.append(col)
            else:
                basic_fea.append(col)
        template_fea_names = None
        for idx in list(days_fea.keys()):
            if len(days_fea[idx]) == 0:
                days_fea.pop(idx)
            else:
                days_fea[idx] = sorted(days_fea[idx])
                if template_fea_names is None:
                    template_fea_names = days_fea[idx]
        for idx in days_fea.keys():
            for name in template_fea_names:
                if name not in days_fea[idx]:
                    logger.warning(f'feature {name} not in day {idx}')
                    assert(0)
        self.configs['basic_fea'] = basic_fea
        self.configs['days_fea'] = template_fea_names
        # remove days feature
        logger.info('dropping feature')
        self.data_pd = self.data_pd.loc[:,self.configs['used_feature']]
        self.data_pd.drop(labels=self.configs['high_correlation'], axis=1, inplace=True)
        logger.info(f'{len(self.data_pd.columns)} feature used')
        logger.info(self.data_pd.columns)
        logger.info('Feature check OK')


    def plot_correlation(self):
        data_pd = self.data_pd.copy(deep=True)
        tools.plot_dis_correlation(
            X=data_pd.to_numpy()[:,1:],
            Y=data_pd['ARDS'].to_numpy(),
            target_name='ARDS',
            fea_names=list(data_pd.columns)[1:],
            write_dir_path=os.path.join(self.out_path, 'dis_corr')
        )

    def plot_na(self, mode='matrix', disp=False):
        tools.plot_na(data=self.data_pd, save_path=os.path.join(self.out_path, f'missing_{mode}.png'), mode=mode, disp=disp)

    def profile(self):
        profile = ppf.profile_report.ProfileReport(
            df = self.data_pd,
            # config_file=self.profile_conf_path
        )
        profile.to_file(self.profile_save_path)
    
    def dump_configs(self):
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
        logger.debug('Data configs dumped at', self.conf_cache_path)


    def load_configs(self):
        try:
            with open(self.conf_cache_path,'r', encoding='utf-8') as f:
                self.configs = json.load(f)
        except Exception as e:
            logger.error('open json error')
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
    dataset = StaticSepsisDataset(from_pkl=False)
