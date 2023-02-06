import tools
from dynamic_sepsis_dataset import DynamicSepsisDataset
import os
import numpy as np
import pandas as pd
from tools import logger as logger
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from dynamic_model import baseline as Baseline
import dynamic_model.net as NNet
from tqdm import tqdm
import pickle


class DynamicAnalyzer:
    def __init__(self, dataset:DynamicSepsisDataset):
        self.dataset = dataset
        self.conf = tools.GLOBAL_CONF_LOADER['dynamic_analyzer']
        self.data_pd = dataset.data_pd
        self.type_dict = dataset.get_type_dict()
        self.target_fea = dataset.target_fea # only one name
        self.dynamic_target_name = self.conf['dynamic_target_name'] # 这个是动态模型的k+1天特征
        self.fea_manager = dataset.fea_manager
        self.n_fold = 5

        # for feature importance
        self.register_values = {}
        self.shap_names = self.conf['shap_names'] # value是显示范围, 手动设置要注意左边-1项不需要, 右边太大的项要限

    '''
        生成数据集的总体特征, 这个方法不是必须运行的
    '''
    def feature_explore(self):
        logger.info('Analyzer: Feature explore')
        out_dir = self.conf['paths']['out_dir']
        # target feature avaliable days
        tools.plot_single_dist(self.dataset.target_time_arr[:,1] / 24, f"Available Days: {self.target_fea}", \
            save_path=os.path.join(out_dir, 'target_available_days.png'))
        # first ards days
        first_ards_days = self._cal_first_ards_days(self.data_pd, self.fea_manager.get_expanded_fea(self.target_fea))
        tools.plot_single_dist(first_ards_days, f"First ARDS Days", \
            save_path=os.path.join(out_dir, 'first_ards_days.png'))
        # random plot sample time series
        self._plot_time_series_samples(self.target_fea, n_sample=400, n_per_plots=40, \
            write_dir=os.path.join(out_dir, "samples_plot"))
        # plot fourier tranform result
        self._plot_fourier_transform(self.target_fea, self.dataset.target_time_arr[:,1], save_dir=out_dir)

    # 完成第一天的回归任务, 默认是catboost
    def static_regression(self, params):
        model_name = 'static_regression'
        kf = KFold(n_splits=self.n_fold, shuffle=True)
        self.register_values.clear()
        tools.reinit_dir(os.path.join(self.conf['paths']['out_dir'], model_name), build=True)
        log_path = os.path.join(self.conf['paths']['out_dir'], model_name, 'result.log')
        tools.clear_file(log_path)
        logger.info(f'Evaluating methods:{model_name}')
        
        metrics_list = params['slice_catboost_reg']['metrics_list']
        metrics = {
            key:tools.DynamicPredictionMetric(
                target_name=self.target_fea, expanded_fea=[self.fea_manager.get_expanded_fea(self.target_fea)[0]], \
                out_dir=self.conf['paths']['out_dir']) for key in metrics_list}
        metric = metrics['test']
        train_cols = self.fea_manager.get_names(sta=True)
        for name in self.dataset.configs['unused_fea_in_static_prediction']:
            if name in train_cols:
                train_cols.remove(name)
                logger.info(f'Remove feature in static prediction:{name}')
        target_name = self.fea_manager.get_expanded_fea(self.target_fea)[0][1]
        type_dict = {}
        for key,val in self.dataset.get_type_dict().items():
            if key == target_name or key in train_cols:
                type_dict[key] = val
        dataset = self.data_pd[train_cols + [target_name]].copy()
        dataset = tools.convert_type({'_default': -1}, dataset, type_dict)
        start, duration = self.dataset.get_time_target_idx()
        tools.assert_no_na(dataset)
        
        if params['slice_catboost_reg'].get('quantile') is not None:
            for key in metrics_list:
                metrics[key].set_quantile(params['slice_catboost_reg']['quantile'], round((len(params['slice_catboost_reg']['quantile']) - 1) / 2))
            logger.info(f'Enable quantile in model {model_name}')

        for idx, (train_index, test_index) in enumerate(kf.split(X=dataset)):
            model = Baseline.SliceCatboostRegression(type_dict=type_dict, params=params['slice_catboost_reg'])
            X_train = dataset.loc[train_index, train_cols]
            X_test = dataset.loc[test_index, train_cols]
            Y_train = dataset.loc[train_index, target_name]
            model.train(X_train, Y_train)
            Y_pred = model.predict(X_test=X_test)[...,None] # (quantile, sample, 1)
            Y_gt = dataset.loc[test_index, target_name][...,None]
            shap_array, shap, sorted_names = model.model_explanation()
            # 注册每个样本的每个特征对应的shap值
            if idx == 0:
                self.register_values['shap'] = {}
                self.register_values['shap_arr'] = {}
            for fea_idx, name in enumerate(train_cols):
                if name not in self.shap_names.keys():
                    continue
                pairs = np.concatenate((shap_array[:, [fea_idx]], model.X_valid[:, [fea_idx]].astype(np.float32)), axis=1)
                if self.register_values['shap_arr'].get(name) is None:
                    self.register_values['shap_arr'][name] = pairs
                else:
                    self.register_values['shap_arr'][name] = np.concatenate((self.register_values['shap_arr'][name], pairs), axis=0)
            # 将每次得到的shap值相加, 便于计算k_fold的平均重要性
            for i in range(len(shap)):
                if self.register_values['shap'].get(sorted_names[i]) is None:
                    self.register_values['shap'][sorted_names[i]] = shap[i]
                else:
                    self.register_values['shap'][sorted_names[i]] += shap[i]
            self._plot_fea_importance(model_name)
            metric.add_prediction(
                prediction=Y_pred, gt=Y_gt, start_idx=start[test_index], duration=duration[test_index])
            if 'train' in metrics_list:
                Y_pred = model.predict(X_test=X_train)[...,None] # (quantile, sample, 1)
                Y_gt = Y_train[...,None]
                metrics['train'].add_prediction(prediction=Y_pred, gt=Y_gt, start_idx=start[train_index], duration=duration[train_index])
        metric.write_result(model_name, log_path=log_path)
        metric.plot(model_name)
        if 'train' in metrics_list:
            corr_dir = os.path.join(self.conf['paths']['out_dir'], tools.remove_slash(model_name), f'correlation_train')
            tools.reinit_dir(write_dir_path=corr_dir)
            metrics['train'].plot_corr(corr_dir=corr_dir, comment='train')
        self.create_final_result()

    def lstm_model(self, params, load_path:str=None):
        kf = KFold(n_splits=self.n_fold, shuffle=True)
        self.register_values.clear()
        model_name = 'LSTM_model'
        tools.reinit_dir(os.path.join(self.conf['paths']['out_dir'], model_name), build=True)
        log_path = os.path.join(self.conf['paths']['out_dir'], model_name, 'result.log')
        metrics_list = params['metrics_list']
        # datasets and params
        in_channels = None
        sta_names = None
        dyn_names = None
        data_3d = None
        data_mask = None
        duration = None
        if load_path is None or not os.path.exists(load_path): # 制作数据集
            logger.info('Creating dataset for lstm_model, it may take few minutes')
            sta_names = self.fea_manager.get_names(sta=True)
            dyn_names = self.fea_manager.get_names(dyn=True)
            assert(self.target_fea in dyn_names)
            in_channels = len(sta_names) + len(dyn_names) # 动态计算用到特征的数量,
            t_channels = len(self.fea_manager.get_expanded_fea(self.target_fea))
            # 先填充-1, 后续再修正
            data_df = tools.convert_type({'_default': -1}, self.data_pd.copy(), self.dataset.get_type_dict())
            # 建立3D array
            dyn_name_dict = {}
            target_times = [val[0] for val in self.fea_manager.get_expanded_fea(self.target_fea)]
            for mode in dyn_names:
                dyn_name_dict[mode] = [self.fea_manager.get_nearest_fea(mode, time) for time in target_times]
            data_3d = np.empty((len(data_df), in_channels, t_channels), dtype=object) # size: (sample, [sta_fea, dyn_fea], time)
            for idx in tqdm(range(len(data_df)), desc='Build array:'):
                data_3d[idx, :len(sta_names), :] = data_df.loc[idx, sta_names].to_numpy(dtype=object, copy=True)[...,None] # copy sta names
                for d_idx, name in enumerate(dyn_names):
                    data_3d[idx, len(sta_names) + d_idx, :] = data_df.loc[idx, dyn_name_dict[name]]
            # 首位对齐和制作mask
            data_mask = np.zeros((data_3d.shape[0], data_3d.shape[2]), dtype=bool) # 可用时间是按照target计算的
            start, duration = self.dataset.get_time_target_idx()
            for idx in tqdm(range(len(data_df)), desc='Create mask:'):
                data_mask[idx, 0:duration[idx]] = True
                data_3d[idx, :, 0:duration[idx]] = data_3d[idx, :, start[idx]:start[idx]+duration[idx]]
            # 建立新的type_dict
            type_dict = {}
            old_type_dict = self.dataset.get_type_dict()
            for idx, name in enumerate(sta_names):
                type_dict[idx] = old_type_dict[name]
            for idx, name in enumerate(dyn_names):
                type_dict[idx+len(sta_names)] = old_type_dict[self.fea_manager.get_expanded_fea(name)[0][1]]
            # 进行TS和插值
            fea_names = sta_names + dyn_names
            train_cols = [] # 除了target以外的序号
            ctg_cols = [] # 类别特征序号
            target_col = None
            for idx, name in enumerate(fea_names):
                if name == self.target_fea:
                    target_col = idx
                else:
                    train_cols.append(idx)
                if type_dict[idx] == str:
                    ctg_cols.append(idx)
            # 这里的target statistic用同一时间的目标值做ts
            # 如果采样频率足够, 那么相邻时间短平稳性质不会改变
            # 如果采样频率不足, 为了避免最后一刻时间无法做TS导致的种种问题(类型不匹配), 还是选择直接TS
            for t_idx in range(data_3d.shape[2]):
                data_3d[:, :, t_idx], _ = tools.target_statistic(
                    X=data_3d[:, :, t_idx],Y=data_3d[:,target_col,t_idx], ctg_feas=ctg_cols, mode=params['ts_mode'], hist_val=None, mask=data_mask[:, t_idx])
            # 将mask以外的部分都置为-1
            data_3d = data_3d.astype(np.float32) * data_mask[:,np.newaxis,:] - np.ones(data_3d.shape, dtype=np.float32) * np.invert(data_mask[:,np.newaxis,:])
            # 插值消除na(线性插值)
            t_arr = np.linspace(0,1,num=data_3d.shape[2])
            na_mat = (data_3d <= -1 + 1e-3)
            valid_mat = np.asarray(1 - na_mat, dtype=bool)
            for r_idx in tqdm(range(data_3d.shape[0]), desc='Fill NA part1'):
                mask_idx = data_mask[r_idx, :]
                for c_idx in range(data_3d.shape[1]):
                    n_idx = (na_mat[r_idx, c_idx,:] * mask_idx).astype(bool) # 空缺的有效值
                    v_idx = (valid_mat[r_idx, c_idx,:] * mask_idx).astype(bool) # 有效部分
                    if not n_idx.any() or not v_idx.any():
                        continue
                    t_x = t_arr[n_idx]
                    data_3d[r_idx, c_idx, n_idx] = \
                        np.interp(x=t_x, xp=t_arr[v_idx], fp=data_3d[r_idx, c_idx, v_idx])
            # 有些全无效的列是保留-1的, 所需还需做均值填充
            for t_idx in tqdm(range(data_3d.shape[2]), desc='Fill NA part2'):
                data_3d[:,:,t_idx], _ = tools.fill_avg(data_3d[:,:,t_idx], num_feas=list(range(data_3d.shape[1])))
            data_3d = data_3d.astype(np.float32) * data_mask[:,np.newaxis,:] - np.ones(data_3d.shape, dtype=np.float32) * np.invert(data_mask[:,np.newaxis,:])
            # 保存array
            with open(load_path, 'wb') as f:
                pickle.dump([data_3d, data_mask, duration, sta_names, dyn_names], f)
                logger.info(f'Dataset dumped at {load_path}')
        else:
            with open(load_path, 'rb') as f:
                data_3d, data_mask, duration, sta_names, dyn_names = pickle.load(f)
                in_channels = len(sta_names) + len(dyn_names)
                logger.info(f'Dataset loaded from {load_path}')
        metrics = {key:tools.DynamicPredictionMetric(target_name=self.target_fea,
            expanded_fea=self.fea_manager.get_expanded_fea(self.target_fea), out_dir=self.conf['paths']['out_dir']) for key in metrics_list}
        if 'quantile' in params.keys():
            for mode in metrics_list:
                metrics[mode].set_quantile(params['quantile'], round((len(params['quantile']) - 1) / 2))
            logger.info(f'Enable quantile in model {model_name}')
        params['in_channels'] = in_channels
        params['target'] = self.dataset.target_fea
        params['fea_names'] = sta_names + dyn_names
        params['cache_path'] = self.conf['paths']['lstm_model_cache_dir']
        # loss logger
        loss_logger = tools.LossLogger()
        # 训练集划分
        for idx, (data_index, test_index) in enumerate(kf.split(X=self.data_pd)):
            valid_num = round(len(data_index)*0.15)
            train_index, valid_index = data_index[valid_num:], data_index[:valid_num]
            target_col = [False if name != params['target'] else True for name in params['fea_names']]
            dataset = NNet.Dataset(
                params=params, data=data_3d, mask=data_mask, train_index=train_index, valid_index=valid_index, test_index=test_index)
            trainer = NNet.Trainer(params, dataset)
            if idx == 0:
                trainer.summary()
            trainer.train()
            loss_logger.add_loss(trainer.get_loss())
            for mode in metrics_list:
                index = None
                if mode == 'test':
                    index = test_index
                elif mode == 'valid':
                    index = valid_index
                else:
                    index = train_index
                Y_gt = data_3d[index, target_col, :]
                Y_pred =  trainer.predict(mode=mode)
                Y_pred = np.asarray(Y_pred)
                if mode == 'test': # 生成输出采样图
                    if 'quantile' in params.keys():
                        quantile_idx = round(len(params['quantile'][:-1]) / 2)
                        tools.simple_plot(data=Y_pred[quantile_idx, np.random.randint(low=0, high=Y_pred.shape[1]-1, size=(10,)), :], 
                            title=f'Random Output Plot idx={idx}', out_path=os.path.join(self.conf['paths']['out_dir'], model_name, f'out_plot_idx={idx}.png'))
                # 生成对齐后的start_idx, 并不是原来的start_idx
                start_idx = np.zeros((Y_pred.shape[1 if 'quantile' in params.keys() else 0]), dtype=np.int32)
                metrics[mode].add_prediction(prediction=Y_pred, gt=Y_gt, start_idx=start_idx, duration=duration[index])
        loss_logger.plot(std_bar=False, log_loss=False, title='Loss for LSTM Model', 
            out_path=os.path.join(self.conf['paths']['out_dir'], tools.remove_slash(model_name), 'loss.png'))
        if metrics['test'].quantile_flag:
            self._plot_quantile_association(model_name, metrics['test'].get_record())
        metrics['test'].write_result(model_name, log_path=log_path)
        metrics['test'].plot(model_name)
        for key in metrics_list:
            if key != 'test':
                corr_dir = os.path.join(self.conf['paths']['out_dir'], tools.remove_slash(model_name), f'correlation_{key}')
                tools.reinit_dir(write_dir_path=corr_dir)
                metrics[key].plot_corr(corr_dir=corr_dir, comment=key)
        self.create_final_result()

    def baseline_methods(self, models:set, params=None):
        kf = KFold(n_splits=self.n_fold, shuffle=True)
        for model_name in models:
            self.register_values.clear()
            tools.reinit_dir(os.path.join(self.conf['paths']['out_dir'], model_name), build=True)
            log_path = os.path.join(self.conf['paths']['out_dir'], model_name, 'result.log')
            tools.clear_file(log_path)
            logger.info(f'Evaluating baseline methods:{model_name}')
            metric = None
            
            if 'simple' in model_name: # simple_nearest simple_average simple_holt
                metric = tools.DynamicPredictionMetric(target_name=self.target_fea,
                    expanded_fea=self.fea_manager.get_expanded_fea(self.target_fea), out_dir=self.conf['paths']['out_dir'])
                predictor = Baseline.SimpleTimeSeriesPredictor()
                slice_dict = self.dataset.target_time_dict
                dataset = slice_dict['data'].copy()
                dataset = tools.convert_type({'_default': -1}, dataset, {k:float for k in dataset.columns}).to_numpy()
                for idx, (train_index, test_index) in enumerate(kf.split(X=dataset)):
                    X_test = dataset[test_index, :]
                    start_idx_test = slice_dict['start_idx'][test_index]
                    duration_test = slice_dict['dur_len'][test_index]
                    result = predictor.predict(X_test, start_idx=start_idx_test, duration=duration_test, mode=model_name.split('simple_')[1], params=params)
                    metric.add_prediction(prediction=result, gt=X_test, start_idx=start_idx_test, duration=duration_test)

            elif model_name == 'slice_linear_reg':
                metric = tools.DynamicPredictionMetric(target_name=self.target_fea,
                    expanded_fea=self.fea_manager.get_expanded_fea(self.target_fea), out_dir=self.conf['paths']['out_dir'])
                slice_dict = self.dataset.slice_dict
                dataset = slice_dict['data'].copy()
                dataset = tools.convert_type({'_default': -1}, dataset, slice_dict['type_dict'])
                tools.assert_no_na(dataset)
                Y_pred = -np.ones(slice_dict['gt_table'].shape)
                train_cols = list(dataset.columns)
                train_cols.remove(self.dynamic_target_name)
                for idx, (train_index, test_index) in enumerate(kf.split(X=dataset)):
                    model = Baseline.SliceLinearRegression(slice_dict['type_dict'], params=params['slice_linear_reg'])
                    X_train = dataset.loc[train_index, train_cols]
                    X_test = dataset.loc[test_index, train_cols]
                    Y_train = dataset.loc[train_index, self.dynamic_target_name]
                    model.train(X_train, Y_train)
                    result = model.predict(X_test=X_test)
                    Y_pred = model.map_result(Y_pred=Y_pred, result=result, map_table=slice_dict['map_table'], index=test_index)
                metric.add_prediction(
                    prediction=Y_pred, gt=slice_dict['gt_table'].to_numpy(), start_idx=slice_dict['start_idx'], duration=slice_dict['dur_len'])

            elif model_name == 'slice_catboost_reg':
                metrics_list = params['slice_catboost_reg']['metrics_list']
                metrics = {
                    key:tools.DynamicPredictionMetric(
                        target_name=self.target_fea, \
                        expanded_fea=self.fea_manager.get_expanded_fea(self.target_fea), \
                        out_dir=self.conf['paths']['out_dir']) for key in metrics_list}
                metric = metrics['test']
                slice_dict = self.dataset.slice_dict
                dataset = slice_dict['data'].copy()
                dataset = tools.convert_type({'_default': -1}, dataset, slice_dict['type_dict'])
                tools.assert_no_na(dataset)
                train_cols = list(dataset.columns)
                train_cols.remove(self.dynamic_target_name)
                if params['slice_catboost_reg'].get('quantile') is not None:
                    for key in metrics_list:
                        metrics[key].set_quantile(params['slice_catboost_reg']['quantile'], round((len(params['slice_catboost_reg']['quantile']) - 1) / 2))
                    logger.info(f'Enable quantile in model {model_name}')
                    Y_pred = -np.ones((len(params['slice_catboost_reg']['quantile']), slice_dict['gt_table'].shape[0],slice_dict['gt_table'].shape[1]))
                else:
                    Y_pred = -np.ones(slice_dict['gt_table'].shape)
                
                for idx, (train_index, test_index) in enumerate(kf.split(X=dataset)):
                    model = Baseline.SliceCatboostRegression(slice_dict['type_dict'], params=params['slice_catboost_reg'])
                    X_train = dataset.loc[train_index, train_cols]
                    X_test = dataset.loc[test_index, train_cols]
                    Y_train = dataset.loc[train_index, self.dynamic_target_name]
                    model.train(X_train, Y_train)
                    result = model.predict(X_test=X_test)
                    Y_pred = model.map_result(Y_pred=Y_pred, result=result, map_table=slice_dict['map_table'], index=test_index)
                    shap_array, shap, sorted_names = model.model_explanation()
                    # 注册每个样本的每个特征对应的shap值
                    if idx == 0:
                        self.register_values['shap'] = {}
                        self.register_values['shap_arr'] = {}
                    for fea_idx, name in enumerate(train_cols):
                        if name not in self.shap_names.keys():
                            continue
                        pairs = np.concatenate((shap_array[:, [fea_idx]], model.X_valid[:, [fea_idx]].astype(np.float32)), axis=1)
                        if self.register_values['shap_arr'].get(name) is None:
                            self.register_values['shap_arr'][name] = pairs
                        else:
                            self.register_values['shap_arr'][name] = np.concatenate((self.register_values['shap_arr'][name], pairs), axis=0)
                    # 将每次得到的shap值相加, 便于计算k_fold的平均重要性
                    for i in range(len(shap)):
                        if self.register_values['shap'].get(sorted_names[i]) is None:
                            self.register_values['shap'][sorted_names[i]] = shap[i]
                        else:
                            self.register_values['shap'][sorted_names[i]] += shap[i]
                    self._plot_fea_importance(model_name)
                    # 记录train的值:
                    if 'train' in metrics_list:
                        Y_pred_train = -np.ones(
                            (len(params['slice_catboost_reg']['quantile']), slice_dict['gt_table'].shape[0],slice_dict['gt_table'].shape[1]))
                        result = model.predict(X_test=X_train)
                        Y_pred_train = model.map_result(Y_pred=Y_pred_train, result=result, map_table=slice_dict['map_table'], index=train_index)
                        metrics['train'].add_prediction(
                            prediction=Y_pred_train, gt=slice_dict['gt_table'].to_numpy(), start_idx=slice_dict['start_idx'], duration=slice_dict['dur_len'])
                metric.add_prediction(
                    prediction=Y_pred, gt=slice_dict['gt_table'].to_numpy(), start_idx=slice_dict['start_idx'], duration=slice_dict['dur_len'])
            else:
                logger.error('Invalid method name')
                assert(0)
            if metric.quantile_flag:
                self._plot_quantile_association(model_name, metric.get_record())
            metric.write_result(model_name, log_path=log_path)
            metric.plot(model_name)
            if 'train' in metrics_list:
                corr_dir = os.path.join(self.conf['paths']['out_dir'], tools.remove_slash(model_name), f'correlation_train')
                tools.reinit_dir(write_dir_path=corr_dir)
                metrics['train'].plot_corr(corr_dir=corr_dir, comment='train')
        self.create_final_result()

    # 收集各个文件夹里面的result.log, 合并为final result.log
    def create_final_result(self):
        logger.info('Creating final result')
        out_dir = self.conf['paths']['out_dir']
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

    def _plot_fea_importance(self, model_name):
        # 得到整体的特征重要性表
        items = list(self.register_values['shap'].items())
        items = np.asarray(sorted(items, key= lambda x:x[1]))
        shap_vals = np.asarray(items[:, 1], dtype=np.float32) / self.n_fold
        with open(os.path.join(self.conf['paths']['out_dir'], model_name,'shap.log'), 'w') as fp:
            fp.write(f'Slice CatBoost Regressor feature importance \n')
            for i in reversed(range(shap_vals.shape[0])):
                fp.write(f"{shap_vals[i]},{str(items[i,0])}\n")
        tools.plot_fea_importance(shap_vals[-10:], items[-10:, 0], save_path=os.path.join(self.conf['paths']['out_dir'], model_name,'shap.png'))
        # 给出单个特征的重要性图
        if 'shap_arr' in self.register_values.keys():
            for fea_name, vals in self.register_values['shap_arr'].items():
                tools.plot_shap_scatter(fea_name, vals[:, 0], vals[:, 1], self.shap_names[fea_name], 
                    os.path.join(self.conf['paths']['out_dir'], model_name))
    
    def _cal_first_ards_days(self, data:pd.DataFrame, expanded_target:list):
        assert(isinstance(expanded_target[0], tuple))
        # expanded_target = self.fea_manager.get_expanded_fea(dyn_fea)
        offset = expanded_target[1][0]
        names = [val[1] for val in expanded_target]
        valid_mat = (1 - data[names].isna()).astype(bool)
        result = -np.ones((len(data))) # start_time, duration
        for r_idx in range(len(data)):
            for time, name in expanded_target:
                if bool(valid_mat.at[r_idx, name]) is True and data.at[r_idx, name] < 300:
                    result[r_idx] = (time + offset)/24
                    break
        return result[result > -0.5]
    
    # 打印若干个sample的时间趋势
    def _plot_time_series_samples(self, dyn_name, n_sample:int=100, n_per_plots:int=10, write_dir=None):
        if write_dir is not None:
            tools.reinit_dir(write_dir)
        expanded = self.fea_manager.get_expanded_fea(dyn_name=dyn_name)
        names = [val[1] for val in expanded]
        times = np.asarray([val[0] for val in expanded])
        data_arr = self.data_pd[names].to_numpy(dtype=float)
        n_plot = int(np.ceil(n_sample / n_per_plots))
        idx = 0
        for p_idx in range(n_plot):
            stop_idx = min(idx + n_per_plots, n_sample)
            plt.plot(times, data_arr[idx:stop_idx, :].T, alpha=0.3)
            plt.title(f"Time series sample for {dyn_name}")
            plt.xlabel("time/hour")
            plt.ylabel(dyn_name)
            idx = stop_idx
            if write_dir is None:
                plt.show()
            else:
                plt.savefig(os.path.join(write_dir, f"plot_{p_idx}.png"))
            plt.close()

    def _plot_fourier_transform(self, fea_name, target_time_arr, save_dir=None):
        # 只拿天数=7的样本作傅里叶变换
        exp_fea = [val[1] for val in self.fea_manager.get_expanded_fea(self.target_fea)]

        data = self.data_pd.loc[target_time_arr > 24*6.5, exp_fea].to_numpy(dtype=float)
        result = np.log10(np.abs(np.fft.fft(data, axis=1))) # log amp
        result_mean = np.mean(result, axis=0)
        result_std = np.std(result, axis=0)
        freq = np.asarray([0.5*val/len(exp_fea) for val in range(len(exp_fea))])
        fig, ax = plt.subplots()
        ax.plot(freq, result_mean, 'b+-')
        ax.errorbar(freq, result_mean, result_std, capsize=4, ecolor='C0')
        plt.title(f"Frequent amplitude for target feature with std")
        plt.xlabel("frequency/day^-1")
        plt.ylabel('Log amplitude')
        if save_dir is None:
            plt.show()
        else:
            plt.savefig(os.path.join(save_dir, 'frequency_energy.png'))
        plt.close()

    # 作图寻找分位点预测和某些变量的关系
    def _plot_quantile_association(self, model_name:str, record:dict):
        pred = record['pred']
        gt = record['gt']
        mask = record['mask']
        quantile_list = [0.25, 0.5, 0.75]
        assert(pred.shape[0] == 3)
        end_idx = 0
        # data
        predictions = np.empty((pred.shape[0], gt.shape[0]*(gt.shape[1]-1))) # 预测值
        delta = np.empty((gt.shape[0]*(gt.shape[1]-1))) # gt-origin
        origin = np.empty((gt.shape[0]*(gt.shape[1]-1))) # 得到的target

        for idx in range(1, gt.shape[1], 1):
            valid = mask[:, idx]
            n_valid = np.sum(valid)
            predictions[:, end_idx:end_idx+n_valid] = pred[:, valid, idx]
            origin[end_idx:end_idx+n_valid] = gt[valid, idx-1]
            delta[end_idx:end_idx+n_valid] = gt[valid, idx] - gt[valid, idx-1]
            end_idx += n_valid

        predictions = predictions[:, :end_idx]
        delta = delta[:end_idx]
        origin = origin[:end_idx]
        logger.info('Plot quantile association')
        corr_dir = os.path.join(self.conf['paths']['out_dir'], model_name, 'explore')
        tools.reinit_dir(corr_dir, build=True)
        tools.plot_correlation_with_quantile(
            X_pred=predictions, x_name='Prediction', Y_gt=origin, target_name='origin',quantile=quantile_list,write_dir_path=corr_dir, equal_lim=False, plot_dash=False)
        tools.plot_correlation_with_quantile(
            X_pred=predictions, x_name='Prediction', Y_gt=delta, target_name='delta',quantile=quantile_list,write_dir_path=corr_dir, equal_lim=False, plot_dash=False)

        

if __name__ == "__main__":
    dataset = DynamicSepsisDataset(from_pkl=True)
    analyzer = DynamicAnalyzer(dataset=dataset)
    baseline_models = {
        'simple_nearest',
        'simple_average',
        'simple_holt',
        'slice_linear_reg',
        'slice_catboost_reg'
    }
    params = {
        'holt':{
            'alpha': 0.5,
            'beta':0,
            'step':1
        },
        'slice_linear_reg':{
            'ts_mode':'greedy'
        },
        'slice_catboost_reg':{
            'ts_mode':'greedy',
            # 'loss_function': 'RMSE',
            'quantile':[0.25, 0.5, 0.75],
            'iterations': 400,
            'depth': 5,
            'learning_rate': 0.04,
            'od_type' : "Iter",
            'od_wait' : 100,
            'verbose': 1,
            'metrics_list': ['train', 'valid', 'test'] # 控制生成的分布图来自哪些数据
        },
        'lstm_model':{
            'ts_mode':'greedy',
            'hidden_size':128,
            'batch_size':2048,
            'device':'cuda:0',
            'lr':1e-3,
            'epochs':150,
            'quantile':[0.25, 0.5, 0.75], # 可以退化为[0.5], 即0.5*MAE
            'punish':5.0, # 惩罚分位数违反规律的系数
            'metrics_list': ['train', 'valid', 'test'] # 控制生成的分布图来自哪些数据
        }
    }
    # module test
    def module_test():
        analyzer.feature_explore()
        analyzer.baseline_methods(models=baseline_models, params=params.copy())
    
    # module_test()
    #analyzer.lstm_model(params['lstm_model'].copy(), load_path=tools.GLOBAL_CONF_LOADER['dynamic_analyzer']['paths']['lstm_dataset_save_path'])
    # analyzer.static_regression(params=params)
    analyzer.baseline_methods(models={'slice_catboost_reg'}, params=params)
    # analyzer.baseline_methods(models={'slice_catboost_reg'}, params=params)

