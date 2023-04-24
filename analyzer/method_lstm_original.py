import numpy as np
from sklearn.model_selection import KFold
import tools
import os
from tools import logger as logger
from sklearn.linear_model import LogisticRegression
from .container import DataContainer
from .utils import generate_labels, map_func, cal_label_weight
from .feature_explore import plot_cover_rate

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
        if self.dataset.name() == 'mimic-iv':
            import models.mimic_model as mlib
        # step 1: append additional params
        limit_idx = {self.dataset.idx_dict[name] for name in self.params['feature_limit']}
        forbidden_idx = {self.dataset.idx_dict[name] for name in self.params['forbidden_feas']}
        self.params['forbidden_idx'] = forbidden_idx
        self.params['limit_idx'] = limit_idx
        # step 2: init variables
        kf = KFold(n_splits=self.container.n_fold, shuffle=True, random_state=self.container.seed)
        out_dir = os.path.join(self.paths['out_dir'], self.model_name)
        tools.reinit_dir(out_dir, build=True)
        os.makedirs(os.path.join(out_dir, 'startstep'), exist_ok=True)
        # metric_2cls = tools.DichotomyMetric()
        metric_startstep = tools.MultiClassMetric(class_names=self.params['class_names'], out_dir=os.path.join(out_dir, 'startstep')) # 起始时刻性能
        metric_4cls = tools.MultiClassMetric(class_names=self.params['class_names'], out_dir=out_dir) # 所有步平均性能
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
            trainer.train()
            self.loss_logger.add_loss(trainer.get_loss())
            Y_mask = mask[test_index, ...]
            Y_gt = label[test_index, ...]
            Y_pred = trainer.predict(mode='test', warm_step=self.params['warm_step'])
            Y_pred = np.asarray(Y_pred)
            metric_4cls.add_prediction(Y_pred, Y_gt, Y_mask) # 去掉mask外的数据
            metric_startstep.add_prediction(Y_pred[:, 0, :], Y_gt[:, 0, :], Y_mask[:,0])
            # metric_2cls.add_prediction(map_func(Y_pred)[..., 1][Y_mask][:], map_func(Y_gt)[..., 1][Y_mask][:])
            self.dataset.mode('all') # 恢复原本状态
        # step 5: result explore
        self.loss_logger.plot(std_bar=False, log_loss=False, title='Loss for LSTM cls Model', 
            out_path=os.path.join(out_dir, 'loss.png'))
        metric_4cls.confusion_matrix(comment=self.model_name)
        metric_startstep.confusion_matrix(comment='Start step ' + self.model_name)
        # metric_2cls.plot_roc(title=f'{self.model_name} model ROC (4->2 cls)', save_path=os.path.join(out_dir, f'{self.model_name}_ROC.png'))
        with open(os.path.join(out_dir, 'result.txt'), 'w') as fp:
            print('Overall performance:', file=fp)
            metric_4cls.write_result(fp)
            print('\n', file=fp)
            print('Startstep performance:', file=fp)
            metric_startstep.write_result(fp)




