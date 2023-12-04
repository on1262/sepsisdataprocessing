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
import yaml
from datasets.derived_vent_dataset import MIMICIV_Vent_Dataset
from scipy.signal import convolve2d

class VentFeatureExplorer:
    def __init__(self, params:dict, container:DataContainer) -> None:
        self.params = params
        self.container = container
        self.dataset = MIMICIV_Vent_Dataset()
        self.dataset.load_version(params['dataset_version'])
        self.gbl_conf = container._conf
        self.data = self.dataset.data

    def run(self):
        '''输出mimic-iv数据集的统计特征, 独立于模型和研究方法'''
        logger.info('Vent dataset Explore')
        dataset_version = self.params['dataset_version']
        out_dir = osjoin(self.params['paths']['out_dir'], f'feature_explore[{dataset_version}]')
        tools.reinit_dir(out_dir, build=True)
        # random plot sample time series
        if self.params['correlation']['enabled']:
            self.correlation(out_dir, self.params['correlation']['target'])
        if self.params['miss_mat']:
            self.miss_mat(out_dir)
        if self.params['vent_statistics']:
            self.vent_statistics(out_dir)
        if self.params['vent_sample']['enabled']:
            self.vent_sample(out_dir)
    
    def correlation(self, out_dir, target_id_or_label):
        # plot correlation matrix
        target_id, target_label = self.dataset.fea_id(target_id_or_label), self.dataset.fea_label(target_id_or_label)
        target_index = self.dataset.idx_dict[target_id]
        labels = [self.dataset.fea_label(id) for id in self.dataset._total_keys]
        corr_mat = tools.plot_correlation_matrix(self.data[:, :, 0], labels, save_path=os.path.join(out_dir, 'correlation_matrix'))
        correlations = []
        for idx in range(corr_mat.shape[1]):
            correlations.append([corr_mat[target_index, idx], labels[idx]]) # list[(correlation coeff, label)]
        correlations = sorted(correlations, key=lambda x:np.abs(x[0]), reverse=True)
        with open(os.path.join(out_dir, 'correlation.txt'), 'w') as fp:
            fp.write(f"Target feature: {target_label}\n")
            for idx in range(corr_mat.shape[1]):
                fp.write(f'Correlation with target: {correlations[idx][0]} \t{correlations[idx][1]}\n')
    
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
        conv_kernel = np.ones((3,3)) / 9
        na_table = np.clip(convolve2d(na_table, conv_kernel, boundary='symm'), 0, 1.0)
        tools.plot_density_matrix(1.0-na_table, 'Missing distribution for subjects and features [miss=white]', xlabel='features', ylabel='subjects',
                               aspect='auto', save_path=os.path.join(out_dir, "miss_mat.png"))

    def vent_statistics(self, out_dir):
        result = {'no_vent':0, 'non-invasive_vent':0, 'invasive_vent':0}
        non_invasive_vent_times = []
        invasive_vent_times = []
        for s_id in self.dataset.subjects:
            subject = self.dataset.subjects[s_id]
            if 'ventilation_num' not in subject.static_data:
                result['no_vent'] += 1
                continue
            vent_num = subject.static_data['ventilation_num']
            max_vent = int(np.max(vent_num[:, 0]))
            if max_vent == 0:
                result['no_vent'] += 1
            elif max_vent == 1:
                result['non-invasive_vent'] += 1
                for idx in range(vent_num.shape[0]):
                    if int(vent_num[idx, 0]) == 1 and (vent_num[idx, 1] >= subject.admissions[0].admittime) and (vent_num[idx, 1] < subject.admissions[0].dischtime):
                        non_invasive_vent_times.append(vent_num[idx, 1] - subject.admissions[0].admittime)
                        break
            elif max_vent == 2:
                result['invasive_vent'] += 1
                for idx in range(vent_num.shape[0]):
                    if int(vent_num[idx, 0]) == 2 and (vent_num[idx, 1] >= subject.admissions[0].admittime) and (vent_num[idx, 1] < subject.admissions[0].dischtime):
                        invasive_vent_times.append(vent_num[idx, 1] - subject.admissions[0].admittime)
                        break
            else:
                assert(0)
        logger.info(f'All subjects: {len(self.dataset.subjects)}')
        logger.info(f'Vent status (max vent for each sequence): {result}')
        # plot distribution of first ventilation time
        tools.plot_single_dist(np.asanyarray(non_invasive_vent_times), 'non-invasive first ventilation time', osjoin(out_dir, 'non_invasive_dist.png'), bins=72)
        tools.plot_single_dist(np.asanyarray(invasive_vent_times), 'invasive first ventilation time', osjoin(out_dir, 'invasive_dist.png'), bins=72)

    def vent_sample(self, out_dir):
        n_plot = self.params['vent_sample']['n_plot']
        n_idx = 1
        plt.figure(figsize = (6, n_plot*2))
        for s_id in self.dataset.subjects:
            subject = self.dataset.subjects[s_id]
            adm = subject.admissions[0]
            if 'ventilation_num' not in subject.static_data:
                continue
            else:
                vent_status = subject.static_data['ventilation_num'][:, 0]
                vent_start = subject.static_data['ventilation_start'][:, 0]
                vent_end = subject.static_data['ventilation_end'][:, 0]
                mask = np.logical_and(vent_start >= adm.admittime, vent_start < adm.dischtime)
                vent_status = vent_status[mask]
                if not np.any(mask):
                    continue
                status_list = np.unique(vent_status).astype(int).tolist()
                if 2 not in status_list or 1 not in status_list:
                    continue
                vent_start = vent_start[mask]
                vent_end = vent_end[mask]
                plt.subplot(n_plot, 1, n_idx)
                for idx in range(vent_status.shape[0]):
                    plt.plot([vent_start[idx] - adm.admittime, vent_end[idx] - adm.admittime], [vent_status[idx], vent_status[idx]], '-o')
                n_idx += 1
                if n_idx > n_plot:
                    break
        plt.savefig(osjoin(out_dir, 'vent_sample.png'))
        plt.close()
        