import tools
from tools.colorful_logging import logger
from .container import DataContainer
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
import os
from os.path import join as osjoin
import pandas as pd
from datasets.mimic_raw_dataset import MIMICIV_Raw_Dataset
from scipy.signal import convolve2d


class FeatureExplorerRaw:
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
        if self.params['generate_report']:
            self.dataset.make_report(version_name=dataset_version, params=self.params['report_params'])
        if self.params['plot_chart_vis']['enabled']:
            self.plot_chart_vis(out_dir=osjoin(out_dir, 'chart_vis'))
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
        if self.params['correlation']:
            self.correlation(out_dir)
        if self.params['abnormal_dist']['enabled']:
            self.abnormal_dist(out_dir)
        if self.params['miss_mat']:
            self.miss_mat(out_dir)
        if self.params['feature_count']:
            self.feature_count(out_dir)
    
    def plot_chart_vis(self, out_dir):
        tools.reinit_dir(out_dir)
        if self.params['plot_chart_vis']['plot_transfer_careunit']:
            transfer_path = osjoin(self.params['paths']['mimic-iv']['mimic_dir'], 'hosp', 'transfers.csv')
            table = pd.read_csv(transfer_path, engine='c', encoding='utf-8')
            record = {}
            for row in tqdm(table.itertuples(), 'plot chart: transfers'):
                r = 'empty' if not isinstance(row.careunit, str) else row.careunit
                if not r in record:
                    record[r] = 1
                else:
                    record[r] += 1
            # sort and plot careunit types
            x = sorted(list(record.keys()), key=lambda x:record[x], reverse=True)
            y = np.asarray([record[k] for k in x])
            tools.plot_bar_with_label(y, x, f'hosp/transfer.careunit Count', out_path=os.path.join(out_dir, f"transfer_careunit.png"))

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
                    count_hist[key] = {'count':0, 'interval':0}
                count_hist[key]['count'] += adm[key].shape[0]
                count_hist[key]['interval'] += ((adm[key][-1, 1] - adm[key][0, 1]) / adm[key].shape[0])
        for key in count_hist.keys():
            count_hist[key]['count'] /= len(adms)
            count_hist[key]['interval'] /= len(adms)
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


def plot_cover_rate(class_names, labels, mask, out_dir):
    '''
    二分类/多分类问题的覆盖率探究
    labels: (sample, seq_lens, n_cls)
    mask: (sample, seq_lens)
    out_dir: 输出文件夹
    '''
    assert(mask.shape == labels.shape[:-1])
    assert(len(class_names) == labels.shape[-1])
    mask_sum = np.sum(mask, axis=1)
    valid = (mask_sum > 0) # 有效的行, 极少样本无效
    logger.debug(f'sum valid: {valid.sum()}')
    label_class = (np.argmax(labels, axis=-1) + 1) * mask # 被mask去掉的是0,第一个class从1开始
    if len(class_names) == 2:
        cover_rate = np.sum(label_class==2, axis=1)[valid] / mask_sum[valid] # ->(sample,)
        tools.plot_single_dist(data=cover_rate, 
            data_name=f'{class_names[1]} cover rate (per sample)', 
            save_path=os.path.join(out_dir, 'cover_rate.png'), discrete=False, adapt=False,bins=10)
    else:
        cover_rate = []
        names = []
        for idx, name in enumerate(class_names):
            arr = np.sum(label_class==idx+1, axis=1)[valid] / mask_sum[valid] # ->(sample,)
            arr = arr[arr > 0]
            names += [name for _ in range(len(arr))]
            cover_rate += [arr]
        cover_rate = np.concatenate(cover_rate, axis=0)
        df = pd.DataFrame(data={'coverrate':cover_rate, 'class':names})
        sns.histplot(
            df,
            x="coverrate", hue='class',
            multiple="stack",
            palette=sns.light_palette("#79C", reverse=True, n_colors=4),
            edgecolor=".3",
            linewidth=.5,
            log_scale=False,
        )
        plt.savefig(os.path.join(out_dir, f'coverrate_4cls.png'))
        plt.close()

