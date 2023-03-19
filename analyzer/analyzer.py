import torch
import torchinfo
import numpy as np
from datasets.mimic_dataset import MIMICDataset, Subject, Admission, Config # 这个未使用的import是pickle的bug
import mimic_model.lstm_reg as lstmreg
import mimic_model.lstm_cls as lstmcls
from mimic_model.baseline import BaselineNearest
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import tools
import os
from tqdm import tqdm
from tools import logger as logger


# 鉴于mimic数据集提取后的大小和数量都和之前的数据集在同一规模, 所以这里决定写一个接口(继承)
# 直接复用dynamic_analyzer的代码

class Analyzer():
    def __init__(self, dataset:MIMICDataset):
        self.dataset = dataset
        self.data = dataset.data
        self.gbl_conf = tools.GLOBAL_CONF_LOADER['mimic_analyzer']
        self.loc_conf = Config(cache_path=self.gbl_conf['paths']['conf_cache_path'], manual_path=self.gbl_conf['paths']['conf_manual_path'])
        self.target_name = dataset.target_name # only one name
        self.target_idx = dataset.target_idx
        self.n_fold = 4
        self.ards_threshold = self.gbl_conf['ards_threshold']
        # for feature importance
        self.register_values = {}

    def feature_explore(self):
        '''输出mimic-iv数据集的统计特征'''
        logger.info('Analyzer: Feature explore')
        out_dir = self.gbl_conf['paths']['out_dir']
        
        # random plot sample time series
        self._plot_time_series_samples(self.target_name, n_sample=400, n_per_plots=40, write_dir=os.path.join(out_dir, "target_plot"))
        self._plot_time_series_samples("220224", n_sample=400, n_per_plots=40, write_dir=os.path.join(out_dir, "pao2_plot"))
        self._plot_time_series_samples("223835", n_sample=400, n_per_plots=40, write_dir=os.path.join(out_dir, "fio2_plot"))
        self._plot_samples(num=50, id_list=["220224", "223835"], id_names=['PaO2', 'FiO2'], out_dir=os.path.join(out_dir, 'samples'))
        # plot fourier tranform result
        # 不等长的话结果的合并很麻烦
        # self._plot_fourier_transform(self.target_name, self.dataset.target_time_arr[:,1], save_dir=out_dir)

        # self._miss_mat()
        # self._first_ards_time()
        #self._feature_count()

    def _detect_adm_data(self, id:str):
        '''直接打印某个id的输出'''
        for s_id, s in self.dataset.subjects.items():
            for adm in s.admissions:
                logger.info(adm[id][:,0])
                input()
    
    def _first_ards_time(self):
        '''打印首次呼衰出现的时间分布'''
        out_dir = self.gbl_conf['paths']['out_dir']
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
                if pf[idx] < self.ards_threshold:
                    times.append(adm[pao2_id][idx, 1])
                    count += 1
            if count != 0:
                ards_count += 1
                counts.append(count) 
        tools.plot_single_dist(np.asarray(times), f"First ARDS time(hour)", os.path.join(out_dir, "first_ards_time.png"), restrict_area=True)
        tools.plot_single_dist(np.asarray(counts), f"ARDS Count", os.path.join(out_dir, "ards_count.png"), restrict_area=True)
        logger.info(f"ARDS patients count={ards_count}")

    def _miss_mat(self):
        '''计算行列缺失分布并输出'''
        out_dir = self.gbl_conf['paths']['out_dir']
        na_table = np.zeros((len(self.dataset.subjects), len(self.dataset.dynamic_keys)), dtype=bool)
        for r_id, s_id in enumerate(self.dataset.subjects):
            adm_key = set(self.dataset.subjects[s_id].admissions[0].keys())
            for c_id, key in enumerate(self.dataset.dynamic_keys):
                if key in adm_key:
                    na_table[r_id, c_id] = True
        # 行缺失
        row_nas = na_table.mean(axis=1)
        col_nas = na_table.mean(axis=0)
        tools.plot_single_dist(row_nas, f"Row miss rate", os.path.join(out_dir, "row_miss_rate.png"), discrete=False, restrict_area=True)
        tools.plot_single_dist(col_nas, f"Column miss rate", os.path.join(out_dir, "col_miss_rate.png"), discrete=False, restrict_area=True)

    def _feature_count(self):
        '''打印vital_sig中特征出现的次数和最短间隔排序'''
        out_dir = self.gbl_conf['paths']['out_dir']
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
            labels = [self.dataset.get_fea_label(key) for key in new_list]
            tools.plot_bar_with_label(counts, labels, f'{name} Count', out_path=os.path.join(out_dir, f"{name}_feature_count.png"))
            tools.plot_bar_with_label(intervals, labels, f'{name} Interval', out_path=os.path.join(out_dir, f"{name}_feature_interval.png"))
    
    def _plot_samples(self, num, id_list:list, id_names:list, out_dir):
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

    def _plot_time_series_samples(self, fea_name:str, n_sample:int=100, n_per_plots:int=10, write_dir=None):
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

    def lstm_auto_reg(self):
        '''自回归模型'''
        params = self.loc_conf['lstm_autoreg']
        kf = KFold(n_splits=self.n_fold, shuffle=True)
        self.register_values.clear()
        model_name = 'LSTM_autoreg'
        tools.reinit_dir(os.path.join(self.gbl_conf['paths']['out_dir'], model_name), build=True)
        start_points = params['start_points']
        log_paths = [os.path.join(self.gbl_conf['paths']['out_dir'], model_name, f'result_{sp}.log') for sp in start_points]
        pred_point = params['pred_point']
        # datasets and params
        metrics = {key:tools.DynamicPredictionMetric(target_name=self.target_name, out_dir=self.gbl_conf['paths']['out_dir']) for key in start_points}
        params['in_channels'] = self.data.shape[1]
        params['target_idx'] = self.target_idx
        params['target_std'] = self.dataset.norm_dict[self.target_name]['std']
        params['fea_names'] = self.dataset.total_keys
        params['cache_path'] = self.gbl_conf['paths']['lstm_autoreg_cache']
        params['norm_arr'] = self.dataset.get_norm_array()
        logger.info(f'lstm_model: in channels=' + str(params['in_channels']))
        assert(np.min(params['norm_arr'][:, 1]) > 1e-4) # std can not be zero
        # loss logger
        loss_logger = tools.LossLogger()
        # 训练集划分
        for idx, (data_index, test_index) in enumerate(kf.split(X=self.dataset)):
            valid_num = round(len(data_index)*0.15)
            train_index, valid_index = data_index[valid_num:], data_index[:valid_num]
            self.dataset.register_split(train_index, valid_index, test_index)
            trainer = lstmreg.LSTMAutoRegTrainer(params, self.dataset)
            if idx == 0:
                trainer.summary()
            trainer.train()
            loss_logger.add_loss(trainer.get_loss())
            Y_gt = self.data[test_index, self.target_idx, :]
            Y_pred = trainer.predict(mode='test', start_points=start_points, pred_point=pred_point)
            Y_pred = np.asarray(Y_pred)
            # 生成对齐后的start_idx, 并不是原来的start_idx
            start_idx = np.zeros((len(test_index),), dtype=np.int32)
            duration = pred_point*np.ones((len(test_index),), dtype=np.int32)
            for idx, key in enumerate(start_points):
                yp, gt = Y_pred[idx, :, :], Y_gt[:, key+1:key+pred_point+1]
                # yp = self.dataset.restore_norm(self.target_name, Y_pred[idx, :, :]) # 恢复norm
                # gt = self.dataset.restore_norm(self.target_name, Y_gt[:, key+1:key+pred_point+1])
                metrics[key].add_prediction(prediction=yp , gt=gt, start_idx=start_idx, duration=duration)
            self.dataset.mode('all') # 恢复原本状态
        loss_logger.plot(std_bar=False, log_loss=False, title='Loss for LSTM autoreg Model', 
            out_path=os.path.join(self.gbl_conf['paths']['out_dir'], tools.remove_slash(model_name), 'loss.png'))
        for idx, key in enumerate(start_points):
            metrics[key].write_result(model_name+'_sp_'+str(key), log_path=log_paths[idx])
            metrics[key].plot(model_name+'_sp_'+str(key))
        # self.create_final_result()

    def label_explore(self, labels, mask, out_dir):
        '''
        生成标签的统计信息
        labels, mask: (sample, seq_lens)
        out_dir: 输出文件夹
        '''
        cover_rate = np.sum(labels, axis=1) / np.sum(mask, axis=1)
        tools.plot_single_dist(
            data=cover_rate, data_name='ARDS cover rate (per sample)', save_path=os.path.join(out_dir, 'cover_rate.png'), discrete=False, restrict_area=True)

    def explore_cls_result(self, Y_pred, Y_gt, mask, out_dir, cmt):
        '''
        输出误差和其他特征的统计关系, 观察误差大的样本是否存在特殊的分布
        Y_pred, Y_gt: (batch, seq_lens), 值域只能是[0,1]
        '''
        delta = np.abs(Y_pred - Y_gt)
        cover = (Y_gt > 0) * (Y_gt < self.ards_threshold) * mask
        diffs = np.diff(cover.astype(int), axis=1)
        # count the number of flips
        num_flips = np.count_nonzero(diffs, axis=1)
        num_flips = np.repeat(num_flips[:, None], Y_pred.shape[1], axis=1)
        mask = mask[:]
        num_flips = num_flips[:][mask]
        delta = delta[:][mask][:, None]
        tools.plot_reg_correlation(
            X=delta, fea_names=['Prediction Abs Error'], Y=num_flips, target_name='Num flips', restrict_area=True, write_dir_path=out_dir, plot_dash=False, comment=cmt)


    def lstm_cls(self):
        '''预测窗口内是否发生ARDS的分类器'''
        params = self.loc_conf['lstm_cls']
        params['cache_path'] = self.gbl_conf['paths']['lstm_cls_cache']
        params['in_channels'] = self.data.shape[1]
        kf = KFold(n_splits=self.n_fold, shuffle=True, random_state=100)
        self.register_values.clear()
        model_name = 'LSTM_cls'
        out_dir = os.path.join(self.gbl_conf['paths']['out_dir'], model_name)
        tools.reinit_dir(out_dir, build=True)
        metric = tools.DichotomyMetric()
        # loss logger
        loss_logger = tools.LossLogger()
        # 制作ARDS标签
        self.dataset.mode('all') # 恢复原本状态
        logger.info('Generating ARDS label')
        generator = lstmcls.LabelGenerator(window=params['window'], threshold=self.ards_threshold)
        mask = lstmcls.make_mask((self.data.shape[0], self.data.shape[2]), self.dataset.seqs_len) # -> (batch, seq_lens)
        # 手动去掉最后一格
        mask[:, -1] = False
        labels = generator(self.data[:, self.target_idx, :], mask)
        self.label_explore(labels, mask, out_dir)
        positive_proportion = np.sum(labels[:]) / np.sum(mask)
        logger.info(f'Positive label={positive_proportion:.2f}')
        # 训练集划分
        for idx, (data_index, test_index) in enumerate(kf.split(X=self.dataset)):
            valid_num = round(len(data_index)*0.15)
            train_index, valid_index = data_index[valid_num:], data_index[:valid_num]
            self.dataset.register_split(train_index, valid_index, test_index)
            trainer = lstmcls.Trainer(params, self.dataset)
            if idx == 0:
                trainer.summary()
            trainer.train()
            loss_logger.add_loss(trainer.get_loss())
            Y_mask = mask[test_index, :]
            Y_gt = labels[test_index, :]
            Y_pred = trainer.predict(mode='test')
            Y_pred = np.asarray(Y_pred)
            Y_pred = (Y_pred - Y_pred.min()) / (Y_pred.max() - Y_pred.min()) # 使得K-fold每个model输出的分布相近, 避免average性能下降
            if idx == 0: # 探查prediction和gt差异大的点
                logger.info('Explore result')
                self.explore_cls_result(Y_pred, Y_gt, Y_mask, out_dir, f'idx={idx}')
            Y_mask = Y_mask[:]
            metric.add_prediction(Y_pred[:][Y_mask], Y_gt[:][Y_mask]) # 去掉mask外的数据
            self.dataset.mode('all') # 恢复原本状态
        loss_logger.plot(std_bar=False, log_loss=False, title='Loss for LSTM cls Model', 
            out_path=os.path.join(out_dir, 'loss.png'))
        metric.plot_roc(title='LSTM cls model ROC', save_path=os.path.join(out_dir, 'lstm_cls_ROC.png'))
        print(metric.generate_info())
        # self.create_final_result()


    def nearest_reg(self):
        '''查看自相关性'''
        params = self.loc_conf['baseline_nearest_reg']
        kf = KFold(n_splits=self.n_fold, shuffle=True)
        self.register_values.clear()
        model_name = 'baseline_nearest'
        start_points = params['start_points']
        metrics = {key:tools.DynamicPredictionMetric(target_name=self.target_name, out_dir=self.gbl_conf['paths']['out_dir']) for key in start_points}
        tools.reinit_dir(os.path.join(self.gbl_conf['paths']['out_dir'], model_name), build=True)
        params['target_idx'] = self.target_idx
        params['target_std'] = self.dataset.norm_dict[self.target_name]['std']
        
        pred_point = params['pred_point']
        log_paths = [os.path.join(self.gbl_conf['paths']['out_dir'], model_name, f'result_{sp}.log') for sp in start_points]
        for idx, (data_index, test_index) in enumerate(kf.split(X=self.dataset)):
            valid_num = round(len(data_index)*0.15)
            train_index, valid_index = data_index[valid_num:], data_index[:valid_num]
            self.dataset.register_split(train_index, valid_index, test_index)
            model = BaselineNearest(params, self.dataset)
            Y_gt = self.data[test_index, self.target_idx, :]
            Y_pred = model.predict(mode='test', start_points=start_points, pred_point=pred_point)
            # 生成对齐后的start_idx, 并不是原来的start_idx
            start_idx = np.zeros((len(test_index),), dtype=np.int32)
            duration = pred_point*np.ones((len(test_index),), dtype=np.int32)
            for idx, key in enumerate(start_points):
                yp = Y_pred[idx, :, :]
                gt = Y_gt[:, key+1:key+pred_point+1]
                # yp = self.dataset.restore_norm(self.target_name, Y_pred[idx, :, :]) # 恢复norm
                # gt = self.dataset.restore_norm(self.target_name, Y_gt[:, key+1:key+pred_point+1])
                metrics[key].add_prediction(prediction=yp , gt=gt, start_idx=start_idx, duration=duration)
            self.dataset.mode('all') # 恢复原本状态
        for idx, key in enumerate(start_points):
            metrics[key].write_result(model_name+'_sp_'+str(key), log_path=log_paths[idx])
            metrics[key].plot(model_name+'_sp_'+str(key))
    
    def nearest_cls(self):
        '''预测窗口内是否发生ARDS的分类器'''
        params = self.loc_conf['baseline_nearest_cls']
        kf = KFold(n_splits=self.n_fold, shuffle=True, random_state=100)
        self.register_values.clear()
        model_name = 'nearest_cls'
        out_dir = os.path.join(self.gbl_conf['paths']['out_dir'], model_name)
        tools.reinit_dir(out_dir, build=True)
        metric = tools.DichotomyMetric()
        # 制作ARDS标签
        self.dataset.mode('all') # 恢复原本状态
        logger.info('Generating ARDS label')
        generator = lstmcls.LabelGenerator(window=params['window'], threshold=self.ards_threshold)
        mask = lstmcls.make_mask((self.data.shape[0], self.data.shape[2]), self.dataset.seqs_len) # -> (batch, seq_lens)
        # 手动去掉最后一格
        mask[:, -1] = False
        labels = generator(self.data[:, self.target_idx, :], mask)
        positive_proportion = np.sum(labels[:]) / np.sum(mask)
        logger.info(f'Positive label={positive_proportion:.2f}')
        # 训练集划分
        for idx, (data_index, test_index) in enumerate(kf.split(X=self.dataset)):
            valid_num = round(len(data_index)*0.15)
            train_index, valid_index = data_index[valid_num:], data_index[:valid_num]
            self.dataset.register_split(train_index, valid_index, test_index)
            Y_mask = mask[test_index, :][:]
            Y_gt = labels[test_index, :][:][Y_mask]
            # prediction
            self.dataset.mode('test')
            Y_pred = np.zeros((len(test_index), self.data.shape[-1]), dtype=np.float32)
            for idx, data in tqdm(enumerate(self.dataset), desc='Testing', total=len(self.dataset)):
                np_data = data['data']
                seq_lens = data['length']
                for t_idx in range(seq_lens):
                    Y_pred[idx, t_idx] = np.mean(np_data[self.target_idx, :(t_idx+1)] < self.ards_threshold)
            Y_pred = np.asarray(Y_pred)[:][Y_mask]
            metric.add_prediction(Y_pred, Y_gt) # 去掉mask外的数据
            self.dataset.mode('all') # 恢复原本状态
        metric.plot_roc(title='Baseline cls model ROC', save_path=os.path.join(out_dir, 'baseline_nearest_cls_ROC.png'))
        print(metric.generate_info())

    # 收集各个文件夹里面的result.log, 合并为final result.log
    def create_final_result(self):
        logger.info('Creating final result')
        out_dir = self.gbl_conf['paths']['out_dir']
        with open(os.path.join(out_dir, 'final_result.log'), 'w') as final_f:
            for dir in os.listdir(out_dir):
                p = os.path.join(out_dir, dir)
                if os.path.isdir(p):
                    if 'result.log' in os.listdir(p):
                        rp = os.path.join(p, 'result.log')
                        logger.info(f'Find: {rp}')
                        with open(rp, 'r') as f:
                            final_f.write(f.read())
                            final_f.write('\n')
        logger.info(f'Final result saved at ' + os.path.join(out_dir, 'final_result.log'))