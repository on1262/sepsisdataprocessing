import numpy as np
from sklearn.model_selection import KFold
import tools
import os
from tools import logger as logger
from .container import DataContainer
from .utils import generate_labels, map_func, cal_label_weight
from .feature_explore import plot_cover_rate
import models.mimic_model as mlib

class LSTMOriginalAnalyzer:
    '''
    动态模型, 四分类预测
    '''
    def __init__(self, params:dict, container:DataContainer) -> None:
        self.params = params
        self.paths = params['paths']
        self.container = container
        self.model_name = 'LSTM_original'
        self.loss_logger = tools.LossLogger()
        # copy attribute from container
        self.target_idx = container.dataset.target_idx
        self.dataset = container.dataset
        self.data = self.dataset.data
        self.robust = params['robust']
        # initialize
        self.out_dir = os.path.join(self.paths['out_dir'], self.model_name)
        tools.reinit_dir(self.out_dir, build=True)

    def label_explore(self, label, mask):
        logger.info('Label explore')
        # 2 class
        cls2_label = map_func(label)
        cls2_out = os.path.join(self.out_dir, '2cls')
        tools.reinit_dir(cls2_out, build=True)
        plot_cover_rate(['No_ARDS', 'ARDS'], cls2_label, mask, cls2_out)
        # 4 class
        cls4_out = os.path.join(self.out_dir, '4cls')
        tools.reinit_dir(cls4_out, build=True)
        plot_cover_rate(['Severe','Moderate', 'Mild', 'No_ARDS'], label, mask, cls4_out)

    def run(self):
        '''预测窗口内是否发生ARDS的分类器'''
        # step 1: append additional params
        limit_idx = {self.dataset.idx_dict[name] for name in self.params['feature_limit']}
        forbidden_idx = {self.dataset.idx_dict[name] for name in self.params['forbidden_feas']}
        self.params['forbidden_idx'] = forbidden_idx
        self.params['limit_idx'] = limit_idx
        # step 2: init variables
        out_dir = os.path.join(self.paths['out_dir'], self.model_name)
        tools.reinit_dir(out_dir, build=True)
        os.makedirs(os.path.join(out_dir, 'startstep'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'initial_steps'), exist_ok=True)
        # 登记metric
        if self.robust:
            metric_robust = tools.RobustClassificationMetric(class_names=self.params['class_names'], out_dir=out_dir)
            def dropout_func(missrate):
                    return np.asarray(trainer.predict(mode='test', addi_params={'dropout':missrate}))
        metric_startstep = tools.MultiClassMetric(class_names=self.params['class_names'], out_dir=os.path.join(out_dir, 'startstep')) 
        metric_4cls = tools.MultiClassMetric(class_names=self.params['class_names'], out_dir=out_dir) # 所有步平均性能
        metric_initial = tools.MultiClassMetric(class_names=self.params['class_names'], out_dir=os.path.join(out_dir, 'initial_steps')) # 起始一段时间的平均性能
        # metric_imp = tools.DeepFeatureImportance(device=self.params['device'], fea_names=[self.dataset.get_fea_label(key) for key in self.dataset.total_keys]) #  
        # step 3: generate labels & label explore
        generator = mlib.DynamicLabelGenerator(soft_label=False, window=self.params['window'], centers=self.params['centers'], smoothing_band=self.params['smoothing_band'], limit_idx=self.params['limit_idx'])
        available_idx = generator.available_idx(n_fea=self.data.shape[1])
        self.params['available_idx'] = available_idx
        self.params['in_channels'] = len(available_idx)
        logger.info(f'LSTM in channels:{len(available_idx)}')
        mask, label = generate_labels(self.dataset, self.data, generator, self.out_dir)
        self.label_explore(label, mask)
        # step 4: train and predict
        for idx, (train_index, valid_index, test_index) in enumerate(self.dataset.enumerate_kf()):
            self.params['kf_index'] = idx
            self.params['weight'] = cal_label_weight(len(self.params['centers']), mask[train_index,...], label[train_index,...])
            trainer = mlib.LSTMOriginalTrainer(self.params, self.dataset)
            if idx == 0:
                trainer.summary()
            if self.robust and 'train_miss_rate' in self.params.keys():
                trainer.train(addi_params={'dropout':self.params['train_miss_rate']}) # 训练时对训练集随机dropout
            else:
                trainer.train()
            self.loss_logger.add_loss(trainer.get_loss())
            Y_mask = mask[test_index, ...]
            Y_gt = label[test_index, ...]
            Y_pred = trainer.predict(mode='test', warm_step=self.params['warm_step'])
            Y_pred = np.asarray(Y_pred)
            metric_4cls.add_prediction(Y_pred, Y_gt, Y_mask) # 去掉mask外的数据
            metric_startstep.add_prediction(Y_pred[:, 0, :], Y_gt[:, 0, :], Y_mask[:,0])
            metric_initial.add_prediction(Y_pred[:, :16, :], Y_gt[:, :16, :], Y_mask[:,:16])
            if self.robust:
                for missrate in np.linspace(0, 1, 11):
                    R_pred = dropout_func(missrate)
                    metric_robust.add_prediction(missrate, R_pred[:, 0, :], Y_gt[:, 0, :], Y_mask[:,0])
            # create wrapper
            # metric_imp.add_record(trainer.create_wrapper(self.params['shap_time_thres']), self.data[valid_index,...], self.params['shap_time_thres'])
            self.dataset.mode('all') # 恢复原本状态
        # step 5: result explore
        if self.robust:
            metric_robust.plot_curve()
        # shap values
        # metric_imp.plot_beeswarm(os.path.join(out_dir, 'shap_overview.png'))
        # metric_imp.plot_hotspot(os.path.join(out_dir, 'shap_hotspot.png'))
        # loss logger
        self.loss_logger.plot(std_bar=False, log_loss=False, title='Loss for LSTM cls Model', 
            out_path=os.path.join(out_dir, 'loss.png'))
        # confusion matrix
        metric_4cls.confusion_matrix(comment=self.model_name)
        metric_startstep.confusion_matrix(comment='Start step ' + self.model_name)
        metric_initial.confusion_matrix(comment='Initial Steps' + self.model_name)
        # save result
        with open(os.path.join(out_dir, 'result.txt'), 'w') as fp:
            print('Overall performance:', file=fp)
            metric_4cls.write_result(fp)
            print('\n', file=fp)
            print('Startstep performance:', file=fp)
            metric_startstep.write_result(fp)
            print('\n', file=fp)
            print('Initial steps performance:', file=fp)
            metric_initial.write_result(fp)





