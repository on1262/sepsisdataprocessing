import tools
from tools.logging import logger
from ..container import DataContainer
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
import os
from os.path import join as osjoin
import pandas as pd
from datasets.derived_raw_dataset import MIMICIV_Raw_Dataset
import yaml
from scipy.signal import convolve2d


class RawFeatureExplorer:
    def __init__(self, params:dict, container:DataContainer) -> None:
        self.params = params
        self.container = container
        self.dataset = MIMICIV_Raw_Dataset()
        self.dataset.load_version(params['dataset_version'])
        self.gbl_conf = container._conf
        self.data = self.dataset.data

    def run(self):
        '''输出mimic-iv数据集的统计特征, 独立于模型和研究方法'''
        logger.info('Analyzer: Feature explore')
        dataset_version = self.params['dataset_version']
        out_dir = osjoin(self.params['paths']['out_dir'], f'feature_explore[{dataset_version}]')
        tools.reinit_dir(out_dir, build=True)
        # random plot sample time series
        if self.params['plot_admission_dist']:
            self.plot_admission_dist(out_dir=out_dir)
        if self.params['plot_chart_vis']['enabled']:
            self.plot_chart_vis(out_dir=osjoin(out_dir, 'chart_vis'))
        if self.params['plot_samples']['enabled']:
            n_sample = self.params['plot_samples']['n_sample']
            id_list = [self.dataset.fea_id(x) for x in self.params['plot_samples']['features']]
            id_names = [self.dataset.fea_label(x) for x in self.params['plot_samples']['features']]
            self.plot_samples(num=n_sample, id_list=id_list, id_names=id_names, out_dir=os.path.join(out_dir, 'samples'))
        if self.params['correlation']:
            self.correlation(out_dir)
        if self.params['abnormal_dist']['enabled']:
            self.abnormal_dist(out_dir)
        if self.params['miss_mat']:
            self.miss_mat(out_dir)
        if self.params['feature_count']:
            self.feature_count(out_dir)
    
    def plot_admission_dist(self, out_dir):
        out_path = osjoin(out_dir, 'admission_dist.png')
        admission_path = osjoin(self.params['paths']['mimic-iv-ards']['mimic_dir'], 'hosp', 'admissions.csv')
        admissions = pd.read_csv(admission_path, encoding='utf-8')
        subject_dict = {}
        for row in tqdm(admissions.itertuples(), 'Collect Admission info', total=len(admissions)):
            subject_dict[row.subject_id] = 1 if row.subject_id not in subject_dict else subject_dict[row.subject_id] + 1
        n_adm = np.asarray(list(subject_dict.values()))
        logger.info(f'Admission: mean={n_adm.mean():.3f}')

        logger.info(f'Retain {100*len(n_adm)/np.sum(n_adm):.3f}% admissions if we only choose the first admission.')
        tools.plot_single_dist(n_adm, 'Number of Admission', save_path=out_path, discrete=True, adapt=True, label=True, shrink=0.9, edgecolor=None)
    
    def plot_chart_vis(self, out_dir):
        tools.reinit_dir(out_dir)
        plot_keys = self.params['plot_chart_vis']['collect_list']
        record = {}
        for plot_key in plot_keys:
            if plot_key == 'transfer':
                key_record = {}
                transfer_path = osjoin(self.params['paths']['mimic-iv-ards']['mimic_dir'], 'hosp', 'transfers.csv')
                table = pd.read_csv(transfer_path, engine='c', encoding='utf-8')
                for row in tqdm(table.itertuples(), 'plot category distribution: transfers'):
                    r = 'empty' if not isinstance(row.careunit, str) else row.careunit
                    key_record[r] = 1 if r not in key_record else key_record[r] + 1
                record[plot_key] = key_record
            elif plot_key == 'admission':
                names = ['insurance', 'language', 'marital_status', 'race']
                for name in names:
                    record[name] = {}
                admission_path = osjoin(self.params['paths']['mimic-iv-ards']['mimic_dir'], 'hosp', 'admissions.csv')
                table = pd.read_csv(admission_path, engine='c', encoding='utf-8')
                for row in tqdm(table.itertuples(), 'plot chcategory distribution: admissions'):
                    for name in names:
                        r = 'empty' if not isinstance(getattr(row, name), str) else getattr(row, name)
                        record[name][r] = 1 if r not in record[name] else record[name][r] + 1

        # sort and plot
        plot_in = {}
        for key, key_record in record.items():
            x = sorted(key_record.values(), reverse=True)
            y = sorted(list(key_record.keys()), key=lambda x:key_record[x], reverse=True)
            total = sum(key_record.values())
            x = [xi/total for xi in x]
            plot_in[key] = [x, y]
            tools.plot_bar_with_label(data=np.asarray(x), labels=y, title=f'Category distribution for {key}', sort=False, out_path=osjoin(out_dir, f'category_{key}.png'))
        with open(osjoin(out_dir, 'categories.yml'), 'w', encoding='utf-8') as fp:
            out_dict = {}
            for key in plot_in:
                out_dict[key] = {name:idx+1 for idx, name in enumerate(plot_in[key][1]) if sum(plot_in[key][0][idx:]) > 0.02}
                out_dict[key].update({'Default':0})
            yaml.dump(out_dict, fp)
        
        tools.plot_stack_proportion(plot_in, out_path=os.path.join(out_dir, f"stack_percentage.png"))


    def abnormal_dist(self, out_dir):
        limit:dict = self.params['abnormal_dist']['value_limitation']
        limit_names = list(limit.keys())
        abnormal_table = np.zeros((len(limit_names), 2)) # True=miss
        idx_dict = {name:idx for idx, name in enumerate(limit_names)}
        for s_id in self.dataset.subjects:
            adm = self.dataset.subjects[s_id].admissions[0]
            for key in adm.dynamic_data:
                name = self.dataset.fea_label(key)
                if name in limit.keys():
                    is_abnormal = np.any(
                        np.logical_or(adm.dynamic_data[key][:, 0] < limit[name]['min'], adm.dynamic_data[key][:, 0] > limit[name]['max'])
                    )
                    abnormal_table[idx_dict[name], 0] += (1 if is_abnormal else 0)
                    abnormal_table[idx_dict[name], 1] += 1
        abnormal_table = abnormal_table[:, 0] / np.maximum(abnormal_table[:, 1], 1e-3) # avoid nan
        # sort table
        limit_names = sorted(limit_names, key=lambda x:abnormal_table[limit_names.index(x)], reverse=True)
        abnormal_table = np.sort(abnormal_table)[::-1]
        # bar plot
        plt.figure(figsize=(10,10))
        ax = sns.barplot(x=np.asarray(limit_names), y=abnormal_table)
        ax.set_xticklabels(limit_names, rotation=45, ha='right')
        ax.bar_label(ax.containers[0], fontsize=10, fmt=lambda x:f'{x:.4f}')
        plt.subplots_adjust(bottom=0.3)
        plt.savefig(osjoin(out_dir, 'abnormal.png'))
        plt.close()
    
    def correlation(self, out_dir):
        # plot correlation matrix
        labels = [self.dataset.fea_label(id) for id in self.dataset._total_keys]
        logger.info('plot correlation')
        tools.plot_correlation_matrix(self.data[:, :, 0], labels, save_path=os.path.join(out_dir, 'correlation_matrix'), corr_thres=0.8)
    
    def miss_mat(self, out_dir):
        '''计算行列缺失分布并输出'''
        na_table = np.ones((len(self.dataset.subjects), len(self.dataset._dynamic_keys)), dtype=bool) # True=miss
        for r_id, s_id in enumerate(self.dataset.subjects):
            for adm in self.dataset.subjects[s_id].admissions:
                # TODO 替换dynamic keys到total keys
                adm_key = set(adm.keys())
                for c_id, key in enumerate(self.dataset._dynamic_keys):
                    if key in adm_key:
                        na_table[r_id, c_id] = False
        
        row_nas = na_table.mean(axis=1)
        col_nas = na_table.mean(axis=0)
        tools.plot_single_dist(row_nas, f"Row miss rate", os.path.join(out_dir, "row_miss_rate.png"), discrete=False, adapt=True)
        tools.plot_single_dist(col_nas, f"Column miss rate", os.path.join(out_dir, "col_miss_rate.png"), discrete=False, adapt=True)
        # save raw/col miss rate to file
        tools.save_pkl(row_nas, os.path.join(out_dir, "row_missrate.pkl"))
        tools.save_pkl(col_nas, os.path.join(out_dir, "col_missrate.pkl"))

        # plot matrix
        row_idx = sorted(list(range(row_nas.shape[0])), key=lambda x:row_nas[x])
        col_idx = sorted(list(range(col_nas.shape[0])), key=lambda x:col_nas[x])
        na_table = na_table[row_idx, :][:, col_idx] # (n_subjects, n_feature)
        # apply conv to get density
        conv_kernel = np.ones((5,5)) / 25
        na_table = np.clip(convolve2d(na_table, conv_kernel, boundary='symm'), 0, 1.0)
        tools.plot_density_matrix(1.0-na_table, 'Missing distribution for subjects and features [miss=white]', xlabel='features', ylabel='subjects',
                               aspect='auto', save_path=os.path.join(out_dir, "miss_mat.png"))

    def feature_count(self, out_dir):
        '''打印vital_sig中特征出现的次数和最短间隔排序'''
        adms = [adm for s in self.dataset.subjects.values() for adm in s.admissions]
        count_hist = {}
        for adm in adms:
            for key in adm.keys():
                if key not in count_hist.keys():
                    count_hist[key] = {'num':0, 'count':0, 'interval':0}
                count_hist[key]['num'] += 1
                count_hist[key]['count'] += adm[key].shape[0]
                count_hist[key]['interval'] += ((adm[key][-1, 1] - adm[key][0, 1]) / adm[key].shape[0])
        for key in count_hist.keys():
            count_hist[key]['count'] /= count_hist[key]['num']
            count_hist[key]['interval'] /= count_hist[key]['num']
        key_list = list(count_hist.keys())
        key_list = sorted(key_list, key=lambda x:count_hist[x]['count'])
        key_list = key_list[-40:] # 最多80, 否则vital_sig可能不准
        with open(os.path.join(out_dir, 'interval.txt'), 'w') as fp:
            for key in key_list:
                interval = count_hist[key]['interval']
                fp.write(f'\"{key}\", {self.dataset.fea_label(key)} mean interval={interval:.1f}\n')
        vital_sig = {"220045", "220210", "220277", "220181", "220179", "220180", "223761", "223762", "224685", "224684", "224686", "228640", "224417"}
        med_ind = {key for key in key_list} - vital_sig
        for name in ['vital_sig', 'med_ind']:
            subset = vital_sig if name == 'vital_sig' else med_ind
            new_list = []
            for key in key_list:
                if key in subset:
                    new_list.append(key)
            counts = np.asarray([count_hist[key]['count'] for key in new_list])
            intervals = np.asarray([count_hist[key]['interval'] for key in new_list])
            labels = [self.dataset.fea_label(key) for key in new_list]
            tools.plot_bar_with_label(counts, labels, f'{name} Count', out_path=os.path.join(out_dir, f"{name}_feature_count.png"))
            tools.plot_bar_with_label(intervals, labels, f'{name} Interval', out_path=os.path.join(out_dir, f"{name}_feature_interval.png"))

    def plot_samples(self, num, id_list:list, id_names:list, out_dir):
        '''随机抽取num个样本生成id_list中特征的时间序列, 在非对齐的时间刻度下表示'''
        tools.reinit_dir(out_dir, build=True)
        count = 0
        nrow = len(id_list)
        assert(nrow <= 5) # 太多会导致subplot拥挤
        bar = tqdm(desc='plot samples', total=num)
        for s_id, s in self.dataset.subjects.items():
            for adm in s.admissions:
                if count >= num:
                    return
                plt.figure(figsize = (6, nrow*3))
                # register xlim
                xmin, xmax = np.inf,-np.inf
                for idx, id in enumerate(id_list):
                    if id in adm.keys():
                        xmin = min(xmin, np.min(adm[id][:,1]))
                        xmax = max(xmax, np.max(adm[id][:,1]))
                for idx, id in enumerate(id_list):
                    if id in adm.keys():
                        plt.subplot(nrow, 1, idx+1)
                        plt.plot(adm[id][:,1], adm[id][:,0], '-o', label=id_names[idx])
                        plt.gca().set_xlim([xmin, xmax])
                        plt.legend()
                plt.suptitle(f'subject={s_id}')
                plt.savefig(os.path.join(out_dir, f'{count}.png'))
                plt.close()
                bar.update(1)
                count += 1