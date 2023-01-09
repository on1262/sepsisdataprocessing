#-*- coding: utf-8 -*-
import pandas as pd
import pandas_profiling as ppf
import json, os
import matplotlib.pyplot as plt
import numpy as np
from tools import *


class TimeSepsisDataset():
    def __init__(self, data_path, from_pkl=False):
        self.data_path = data_path
        self.csv_path = os.path.join(data_path, 'sepsis.csv')
        self.conf_cache_path = os.path.join(data_path, 'configs_cache.json')
        self.conf_manual_path = os.path.join(data_path, 'configs_manual.json')
        self.profile_conf_path = os.path.join(data_path, 'profile_conf.yaml')
        self.profile_save_path = os.path.join(data_path, 'profile.html')
        self.output_cleaned_path = os.path.join(data_path, 'cleaned.csv')
        self.dataframe_save_path = os.path.join(data_path, 'dataframe.pkl')

        with open(self.csv_path, 'r', encoding='utf-8') as f:
            self.data_pd = pd.read_csv(f, encoding='utf-8')

        self.load_configs()

        if from_pkl and os.path.exists(self.dataframe_save_path):
            print(f'loading data from {self.dataframe_save_path}')
            self.data_pd = pd.read_pickle(self.dataframe_save_path)
        else:
            self.create_death_label()
            self.feature_check()

            self.data_pd = one_hot_decoding(self.data_pd, cluster_dict=self.configs['one_hot_decoding'])
            self.plot_na('bar')
            self.data_pd = select_na(self.data_pd, col_thres=0.5, row_thres=0.7)

            self.type_dict = check_fea_types(self.data_pd)
            self.data_pd = remove_invalid_rows(self.data_pd, self.type_dict)

            self.category_dict = detect_category_fea(self.data_pd, self.type_dict, cluster_perc=0.01)
            self.configs['type_dict'] = self.type_dict
            self.configs['category_dict'] = self.category_dict

            # data cleaning
            self.data_pd = apply_category_fea(self.data_pd, self.category_dict)
            fill_default(self.data_pd, self.configs['fill_default'])
            self.data_pd = select_na(self.data_pd, col_thres=0.5, row_thres=0.87)
            self.plot_na('matrix')
            self.plot_na('sample')
            plot_category_dist(data=self.data_pd, type_dict=self.type_dict,
                output_dir=os.path.join(self.data_path, 'category_dist'))
            
            self.dump_configs()
            self.data_pd.to_csv(self.output_cleaned_path, index=False)
            self.data_pd.to_pickle(self.dataframe_save_path)
            self.profile(self.profile_save_path)

        self.target_fea = 'ARDS'
        
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
                    print(f'Warning: feature {name} not in day {idx}')
                    assert(0)
        self.configs['basic_fea'] = basic_fea
        self.configs['days_fea'] = template_fea_names
        # remove days feature
        print('dropping feature')
        self.data_pd = self.data_pd.loc[:,self.configs['used_feature']]
        self.data_pd.drop(labels=self.configs['high_correlation'], axis=1, inplace=True)
        # self.data_pd.drop(labels='df_index', axis=1, inplace=True)
        print(f'{len(self.data_pd.columns)} feature used')
        print(self.data_pd.columns)
        print('Feature check OK')


    def plot_correlation(self):
        plot_dis_correlation(
            X=self.data_pd.to_numpy()[:,1:],
            Y=self.data_pd['ARDS'].to_numpy(),
            target_name='ARDS',
            fea_names=list(self.data_pd.columns)[1:],
            write_dir_path=os.path.join(self.data_path, 'dis_corr')
        )

    def plot_na(self, mode='matrix', disp=False):
        plot_na(data=self.data_pd, save_path=os.path.join(self.data_path, f'missing_{mode}.png'), mode=mode, disp=disp)

    def profile(self, output_path):
        profile = ppf.profile_report.ProfileReport(
            df = self.data_pd,
            # config_file=self.profile_conf_path
        )
        profile.to_file(output_path)
    
    def dump_configs(self):
        configs = self.configs
        if configs.get('one_hot_decoding') is not None:
            for name in configs['one_hot_decoding'].keys():
                configs['one_hot_decoding'][name] = list(configs['one_hot_decoding'][name])
        if configs.get('type_dict') is not None:
            for name in configs['type_dict'].keys():
                configs['type_dict'][name] = configs['type_dict'][name].__name__

        with open(self.conf_cache_path, 'w', encoding='utf-8') as f:
            json.dump(configs, f, ensure_ascii=False) # allow utf-8 characters not converted into \uXXX
        print('Data configs dumped at', self.conf_cache_path)


    def load_configs(self):
        try:
            with open(self.conf_cache_path,'r', encoding='utf-8') as f:
                self.configs = json.load(f)
        except Exception as e:
            print('open json error')
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
            print('Config load error')
            self.configs = {}


if __name__ == '__main__':
    set_chinese_font()
    # dataset = TimeSepsisDataset('F:\\Project\\DiplomaProj\\new_data', from_pkl=False)
    pd_a = pd.read_csv('F:\\Project\\DiplomaProj\\new_data\\data_raw_old.csv', encoding='utf-8')
    pd_b = pd.read_csv('F:\\Project\\DiplomaProj\\new_data\\data_raw_new.csv', encoding='utf-8')
    pd_combine = combine_and_select_samples(pd_a, pd_b, rename_prefix=['oldsys_', 'newsys_'])
    pd_combine.to_csv('F:\\Project\\DiplomaProj\\new_data\\data_combined.csv', encoding='utf-8')
