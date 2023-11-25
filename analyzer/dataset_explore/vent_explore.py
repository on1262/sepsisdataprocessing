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
from datasets.derived_ards_dataset import MIMICIV_ARDS_Dataset


class VentFeatureExplorer:
    def __init__(self, params:dict, container:DataContainer) -> None:
        self.params = params
        self.container = container
        self.dataset = MIMICIV_ARDS_Dataset()
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
            fp.write(f"Target feature: {target_label}")
            for idx in range(corr_mat.shape[1]):
                fp.write(f'Correlation with target: {correlations[idx][0]} \t{correlations[idx][1]}\n')
    
    def miss_mat(self, out_dir):
        '''计算行列缺失分布并输出'''
        na_table = np.ones((len(self.dataset.subjects), len(self.dataset._dynamic_keys)), dtype=bool)
        for r_id, s_id in enumerate(self.dataset.subjects):
            for adm in self.dataset.subjects[s_id].admissions:
                # TODO 替换dynamic keys到total keys
                adm_key = set(adm.keys())
                for c_id, key in enumerate(self.dataset._dynamic_keys):
                    if key in adm_key:
                        na_table[r_id, c_id] = False
        # 行缺失
        row_nas = na_table.mean(axis=1)
        col_nas = na_table.mean(axis=0)
        tools.plot_single_dist(row_nas, f"Row miss rate", os.path.join(out_dir, "row_miss_rate.png"), discrete=False, adapt=True)
        tools.plot_single_dist(col_nas, f"Column miss rate", os.path.join(out_dir, "col_miss_rate.png"), discrete=False, adapt=True)
        # save raw/col miss rate to file
        tools.save_pkl(row_nas, os.path.join(out_dir, "row_missrate.pkl"))
        tools.save_pkl(col_nas, os.path.join(out_dir, "col_missrate.pkl"))