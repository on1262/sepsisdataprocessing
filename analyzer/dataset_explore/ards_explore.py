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
from scipy.signal import convolve2d
from tools.data import DynamicDataGenerator, label_func_min, LabelGenerator_cls, cal_label_weight



class ArdsFeatureExplorer:
    def __init__(self, params:dict, container:DataContainer) -> None:
        self.params = params
        self.container = container
        self.dataset = MIMICIV_ARDS_Dataset()
        self.dataset.load_version(params['dataset_version'])
        self.gbl_conf = container._conf
        self.data = self.dataset.data
        self.dataset.mode('all')
        self.target_idx = self.dataset.fea_idx('PF_ratio')

    def run(self):
        '''输出mimic-iv数据集的统计特征, 独立于模型和研究方法'''
        logger.info('Analyzer: Feature explore')
        dataset_version = self.params['dataset_version']
        out_dir = osjoin(self.params['paths']['out_dir'], f'feature_explore[{dataset_version}]')
        tools.reinit_dir(out_dir, build=True)
        # random plot sample time series
        if self.params['coverrate']['enabled']:
            self.plot_cover_rate(self.params['coverrate']['class_names'])
        if self.params['plot_samples']['enabled']:
            n_sample = self.params['plot_samples']['n_sample']
            id_list = [self.dataset.fea_id(x) for x in self.params['plot_samples']['features']]
            id_names = [self.dataset.fea_label(x) for x in self.params['plot_samples']['features']]
            self.plot_samples(num=n_sample, id_list=id_list, id_names=id_names, out_dir=os.path.join(out_dir, 'samples'))
        if self.params['plot_time_series']['enabled']:
            n_sample = self.params['plot_time_series']['n_sample']
            n_per_plots = self.params['plot_time_series']['n_per_plots']
            for name in self.params['plot_time_series']["names"]:
                self.plot_time_series_samples(name, n_sample=n_sample, n_per_plots=n_per_plots, write_dir=os.path.join(out_dir, f"time_series_{name}"))
        if self.params['correlation']['enabled']:
            self.correlation(out_dir, self.params['correlation']['target'])
        if self.params['miss_mat']:
            self.miss_mat(out_dir)
        if self.params['first_ards_time']:
            self.first_ards_time(out_dir)

    def first_ards_time(self, out_dir):
        '''打印首次呼衰出现的时间分布'''
        times = []
        counts = [] # 产生呼衰的次数
        ards_count = 0
        adms = [adm for s in self.dataset.subjects.values() for adm in s.admissions]
        pao2_id, fio2_id =  "220224", "223835"
        for adm in adms:
            count = 0
            ticks = adm[pao2_id][:, 1]
            fio2 = np.interp(x=ticks, xp=adm[fio2_id][:, 1], fp=adm[fio2_id][:, 0])
            pao2 = adm[pao2_id][:, 0]
            pf = pao2 / fio2
            for idx in range(pf.shape[0]):
                if pf[idx] < self.container.ards_threshold:
                    times.append(adm[pao2_id][idx, 1])
                    count += 1
            if count != 0:
                ards_count += 1
                counts.append(count) 
        tools.plot_single_dist(np.asarray(times), f"First ARDS time(hour)", os.path.join(out_dir, "first_ards_time.png"), adapt=True)
        tools.plot_single_dist(np.asarray(counts), f"ARDS Count", os.path.join(out_dir, "ards_count.png"), adapt=True)
        logger.info(f"ARDS patients count={ards_count}")

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
        conv_kernel = np.ones((5,5)) / 25
        na_table = np.clip(convolve2d(na_table, conv_kernel, boundary='symm'), 0, 1.0)
        tools.plot_density_matrix(1.0-na_table, 'Missing distribution for subjects and features [miss=white]', xlabel='features', ylabel='subjects',
                               aspect='auto', save_path=os.path.join(out_dir, "miss_mat.png"))

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
                plt.figure(figsize = (6, nrow*2))
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
                        plt.gca().set_xlim([xmin-1, xmax+1])
                        plt.legend()
                plt.suptitle(f'subject={s_id}')
                plt.savefig(os.path.join(out_dir, f'{count}.png'))
                plt.close()
                bar.update(1)
                count += 1

    def plot_time_series_samples(self, fea_name:str, n_sample:int=100, n_per_plots:int=10, write_dir=None):
        '''
        fea_name: total_keys中的项, 例如"220224"
        '''
        if write_dir is not None:
            tools.reinit_dir(write_dir)
        n_sample = min(n_sample, self.data.shape[0])
        n_plot = int(np.ceil(n_sample / n_per_plots))
        fea_idx = self.dataset._idx_dict[fea_name]
        start_idx = 0
        label = self.dataset.fea_label(fea_name)
        for p_idx in range(n_plot):
            stop_idx = min(start_idx + n_per_plots, n_sample)
            mat = self.data[start_idx:stop_idx, fea_idx, :]
            for idx in range(stop_idx-start_idx):
                series = mat[idx, :]
                plt.plot(series[series > 0], alpha=0.3)
            plt.title(f"Time series sample of {label}")
            plt.xlabel("time tick=0.5 hour")
            plt.xlim(left=0, right=72)
            plt.ylabel(label)
            start_idx = stop_idx
            if write_dir is None:
                plt.show()
            else:
                plt.savefig(os.path.join(write_dir, f"plot_{p_idx}.png"))
            plt.close()


    def plot_cover_rate(self, class_names):
        generator = DynamicDataGenerator(
            window_points=self.params['coverrate']['window'],
            n_fea=len(self.dataset.total_keys),
            label_generator=LabelGenerator_cls(
                centers=self.params['coverrate']['centers']
            ),
            label_func=label_func_min,
            target_idx=self.target_idx,
            limit_idx=[self.target_idx],
            forbidden_idx=[]
        )
        result = generator(self.dataset.data, self.dataset.seqs_len)
        label, mask = result['label'], result['mask']
        weight = cal_label_weight(4, label[mask, :])
        rw = 1 / weight
        rw = rw / np.sum(rw)
        logger.info(f'Slice label proportion: {rw}')
        label = np.argmax(label, axis=-1)
        label[np.logical_not(mask)] = 4
        cls_label = [np.sum(np.min(label, axis=-1) == idx) for idx in range(4)]
        cls_label = np.asarray(cls_label)
        n_sum = np.sum(cls_label)
        for idx, name in enumerate(class_names):
            logger.info(f'{name}: n={cls_label[idx]}, proportion={cls_label[idx] / n_sum:.3f}')