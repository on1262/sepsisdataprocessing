import tools
from tools.colorful_logging import logger
from .container import DataContainer
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
import os
import pandas as pd

class FeatureExplorer:
    def __init__(self, params:dict, container:DataContainer) -> None:
        self.params = params
        self.container = container
        self.gbl_conf = container._conf
        self.dataset = container.dataset
        self.data = self.dataset.data

    def run(self):
        '''输出mimic-iv数据集的统计特征, 独立于模型和研究方法'''
        logger.info('Analyzer: Feature explore')
        dataset_version = self.params['dataset_version']
        out_dir = os.path.join(tools.GLOBAL_CONF_LOADER['analyzer'][self.container.dataset.name()]['paths']['out_dir'], f'explore_{dataset_version}')
        tools.reinit_dir(out_dir, build=True)
        # random plot sample time series
        if self.params['generate_report']:
            self.dataset.make_report(version_name=dataset_version, params=self.params['report_params'])
        if self.params['plot_samples']:
            self.plot_samples(num=50, id_list=["220224", "223835"], id_names=['PaO2', 'FiO2'], out_dir=os.path.join(out_dir, 'samples'))
        if self.params['plot_time_series']:
            self.plot_time_series_samples(self.container.target_name, n_sample=400, n_per_plots=40, write_dir=os.path.join(out_dir, "target_plot"))
            self.plot_time_series_samples("220224", n_sample=400, n_per_plots=40, write_dir=os.path.join(out_dir, "pao2_plot"))
            self.plot_time_series_samples("223835", n_sample=400, n_per_plots=40, write_dir=os.path.join(out_dir, "fio2_plot"))
        if self.params['correlation']:
            self.correlation(out_dir)
        if self.params['miss_mat']:
            self.miss_mat(out_dir)
        if self.params['first_ards_time']:
            self.first_ards_time(out_dir)
        if self.params['feature_count']:
            self.feature_count(out_dir)
    
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

    def correlation(self, out_dir):
        # plot correlation matrix
        labels = [self.container.dataset.get_fea_label(id) for id in self.container.dataset.total_keys]
        corr_mat = tools.plot_correlation_matrix(self.data[:, :, 0], labels, save_path=os.path.join(out_dir, 'correlation _matrix'))
        correlations = []
        for idx in range(corr_mat.shape[1]):
            correlations.append([corr_mat[-1, idx], labels[idx]]) # list[(correlation coeff, label)]
        correlations = sorted(correlations, key=lambda x:np.abs(x[0]), reverse=True)
        with open(os.path.join(out_dir, 'correlation.txt'), 'w') as fp:
            for idx in range(corr_mat.shape[1]):
                fp.write(f'Correlation with target: {correlations[idx][0]} \t{correlations[idx][1]}\n')
        
    def miss_mat(self, out_dir):
        '''计算行列缺失分布并输出'''
        na_table = np.zeros((len(self.dataset.subjects), len(self.dataset.dynamic_keys)), dtype=bool)
        for r_id, s_id in enumerate(self.dataset.subjects):
            adm_key = set(self.dataset.subjects[s_id].admissions[0].keys())
            for c_id, key in enumerate(self.dataset.dynamic_keys):
                if key in adm_key:
                    na_table[r_id, c_id] = True
        # 行缺失
        row_nas = na_table.mean(axis=1)
        col_nas = na_table.mean(axis=0)
        tools.plot_single_dist(row_nas, f"Row miss rate", os.path.join(out_dir, "row_miss_rate.png"), discrete=False, adapt=True)
        tools.plot_single_dist(col_nas, f"Column miss rate", os.path.join(out_dir, "col_miss_rate.png"), discrete=False, adapt=True)

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
        for key in key_list:
            interval = count_hist[key]['interval']
            logger.info(f'\"{key}\", {self.dataset.get_fea_label(key)} interval={interval:.1f}')
        vital_sig = {"220045", "220210", "220277", "220181", "220179", "220180", "223761", "223762", "224685", "224684", "224686", "228640",    "224417"}
        med_ind = {key for key in key_list} - vital_sig
        for name in ['vital_sig', 'med_ind']:
            subset = vital_sig if name == 'vital_sig' else med_ind
            new_list = []
            for key in key_list:
                if key in subset:
                    new_list.append(key)
            counts = np.asarray([count_hist[key]['count'] for key in new_list])
            intervals = np.asarray([count_hist[key]['interval'] for key in new_list])
            labels = [self.dataset.get_fea_label(key) for key in new_list]
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
        fea_idx = self.dataset.idx_dict[fea_name]
        start_idx = 0
        label = self.dataset.get_fea_label(fea_name)
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

