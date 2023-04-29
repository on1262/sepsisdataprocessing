import numpy as np
from sklearn.model_selection import KFold
import tools
import os
from tools import logger as logger
from .container import DataContainer
from .utils import generate_labels, map_func, cal_label_weight
import models.mimic_model as mlib

class LSTMCascadeAnalyzer:
    '''
    动态模型, 四分类预测
    '''
    def __init__(self, params:dict, container:DataContainer) -> None:
        self.params = params
        self.paths = params['paths']
        self.container = container
        self.model_name = 'LSTM_cascade'
        self.loss_logger = tools.LossLogger()
        # copy attribute from container
        self.target_idx = container.dataset.target_idx
        self.dataset = container.dataset
        self.data = self.dataset.data
        # initialize
        self.out_dir = os.path.join(self.paths['out_dir'], self.model_name)
        tools.reinit_dir(self.out_dir, build=True)

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
        metric_startstep = tools.MultiClassMetric(class_names=self.params['class_names'], out_dir=os.path.join(out_dir, 'startstep')) # 起始时刻性能
        metric_4cls = tools.MultiClassMetric(class_names=self.params['class_names'], out_dir=out_dir) # 所有步平均性能
        # metric_imp = tools.DeepFeatureImportance(device=self.params['device'], fea_names=[self.dataset.get_fea_label(key) for key in self.dataset.total_keys])
        # step 3: generate labels & label explore
        generator = mlib.DynamicLabelGenerator(soft_label=False, window=self.params['window'], centers=self.params['centers'], smoothing_band=self.params['smoothing_band'], limit_idx=self.params['limit_idx'])
        available_idx = generator.available_idx(n_fea=self.data.shape[1])
        self.params['available_idx'] = available_idx
        self.params['in_channels'] = len(available_idx)
        logger.info(f'LSTM in channels:{len(available_idx)}')
        mask, label = generate_labels(self.dataset, self.data, generator, self.out_dir)
        # step 4: train and predict
        for idx, (train_index, valid_index, test_index) in enumerate(self.dataset.enumerate_kf()):
            self.params['kf_index'] = idx
            self.params['weight'] = cal_label_weight(len(self.params['centers']), mask[train_index,...], label[train_index,...])
            trainer = mlib.LSTMCascadeTrainer(self.params, self.dataset)
            if idx == 0:
                trainer.summary()
            trainer.train()
            self.loss_logger.add_loss(trainer.get_loss())
            Y_mask = mask[test_index, ...]
            Y_gt = label[test_index, ...]
            Y_pred = trainer.predict(mode='test', warm_step=self.params['warm_step'])
            Y_pred = np.asarray(Y_pred)
            metric_4cls.add_prediction(Y_pred, Y_gt, Y_mask) # 去掉mask外的数据
            metric_startstep.add_prediction(Y_pred[:, 0, :], Y_gt[:, 0, :], Y_mask[:,0])
            # create wrapper
            # metric_imp.add_record(trainer.create_wrapper(self.params['shap_time_thres']), self.data[valid_index,...], self.params['shap_time_thres'])
            self.dataset.mode('all') # 恢复原本状态
        # step 5: result explore
        # shap values
        # metric_imp.plot_beeswarm(os.path.join(out_dir, 'shap_overview.png'))
        # metric_imp.plot_hotspot(os.path.join(out_dir, 'shap_hotspot.png'))
        # loss logger
        self.loss_logger.plot(std_bar=False, log_loss=False, title='Loss for LSTM cls Model', 
            out_path=os.path.join(out_dir, 'loss.png'))
        # confusion matrix
        metric_4cls.confusion_matrix(comment=self.model_name)
        metric_startstep.confusion_matrix(comment='Start step ' + self.model_name)
        # save result
        with open(os.path.join(out_dir, 'result.txt'), 'w') as fp:
            print('Overall performance:', file=fp)
            metric_4cls.write_result(fp)
            print('\n', file=fp)
            print('Startstep performance:', file=fp)
            metric_startstep.write_result(fp)




