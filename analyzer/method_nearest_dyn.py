import numpy as np
from sklearn.model_selection import KFold
import tools
import os
from tools import logger as logger
from .container import DataContainer
from .utils import generate_labels
import models.mimic_model as mlib

class NearestDynamicAnalyzer:
    def __init__(self, params:dict, container:DataContainer) -> None:
        self.params = params
        self.paths = params['paths']
        self.container = container
        self.model_name = self.params['analyzer_name']
        self.loss_logger = tools.LossLogger()
        # copy attribute from container
        self.target_idx = container.dataset.target_idx
        self.dataset = container.dataset
        self.data = self.dataset.data
        # initialize
        self.out_dir = os.path.join(self.paths['out_dir'], self.model_name)
        self.robust = params['robust']
        tools.reinit_dir(self.out_dir, build=True)


    def run(self):
        '''预测窗口内是否发生ARDS的分类器'''
        # step 1: append additional params
        self.params['in_channels'] = self.dataset.data.shape[1]
        forbidden_idx = {self.dataset.idx_dict[name] for name in self.params['forbidden_feas']}
        limit_idx = {self.dataset.idx_dict[name] for name in self.params['feature_limit']}
        self.params['forbidden_idx'] = forbidden_idx
        self.params['limit_idx'] = limit_idx
        # step 2: init variables
        out_dir = os.path.join(self.paths['out_dir'], self.model_name)
        tools.reinit_dir(out_dir, build=True)
        metric_4cls = tools.MultiClassMetric(class_names=self.params['class_names'], out_dir=out_dir)
        metric_initial = tools.MultiClassMetric(class_names=self.params['class_names'], out_dir=out_dir)
        if self.robust:
            metric_robust = tools.RobustClassificationMetric(class_names=self.params['class_names'], out_dir=out_dir)
            def dropout_func(missrate):
                    return np.asarray(trainer.predict(mode='test', addi_params={'dropout':missrate}))
        # step 3: generate labels
        generator = mlib.SliceLabelGenerator(
            slice_len=self.params['slice_len'],
            soft_label=False, window=self.params['window'], centers=self.params['centers'], smoothing_band=self.params['smoothing_band'], limit_idx=self.params['limit_idx']
        )
        mask, label = generate_labels(self.dataset, self.data, generator.dyn_generator, self.out_dir)
        label, mask = generator.adjust_result(label, mask)
        # fea_names = [self.dataset.get_fea_label(idx) for idx in generator.available_idx()]
        # imp_logger = tools.TreeFeatureImportance(fea_names=fea_names)
        # step 4: train and predict
        for idx, (train_index, valid_index, test_index) in enumerate(self.dataset.enumerate_kf()): 
            trainer = mlib.CatboostDynamicTrainer(self.params, self.dataset)
            if self.robust and 'train_miss_rate' in self.params.keys():
                trainer.train(addi_params={'dropout':self.params['train_miss_rate']}) # 训练时对训练集随机dropout
            else:
                trainer.train()
            self.loss_logger.add_loss(trainer.get_loss())
            Y_gt = label[test_index] # 这里test index需要在label制作之前
            Y_pred = trainer.predict(mode='test')
            Y_pred = np.asarray(Y_pred)
            _Y_pred = generator.restore_from_slice(Y_pred)
            metric_4cls.add_prediction(_Y_pred, Y_gt, mask[test_index])
            metric_initial.add_prediction(_Y_pred[:, :16, :], Y_gt[:, :16, :], mask[test_index, :16])
            # imp_logger.add_record(trainer.model, label['X'])
            if self.robust:
                for missrate in np.linspace(0, 1, 11):
                    R_pred = dropout_func(missrate)
                    R_pred = generator.restore_from_slice(R_pred)
                    metric_robust.add_prediction(missrate, R_pred[:, :16, :], Y_gt[:, :16, :], mask[test_index, :16])
            self.dataset.mode('all') # 恢复原本状态
        # step 5: result explore
        # imp_logger.plot_beeswarm(os.path.join(out_dir, 'shap_overview.png'))
        single_imp_out = os.path.join(out_dir, 'single_shap')
        tools.reinit_dir(single_imp_out, build=True)
        # imp_logger.plot_single_importance(out_dir=single_imp_out, select=10)
        if self.robust:
            metric_robust.save_df(self.model_name)
            metric_robust.plot_curve()
        self.loss_logger.plot(std_bar=False, log_loss=False, title='Loss for Catboost dynamic Model', 
            out_path=os.path.join(out_dir, 'loss.png'))
        metric_4cls.confusion_matrix(comment=self.model_name)
        with open(os.path.join(self.out_dir, 'result.txt'), 'a') as f:
            print('Overall performance:', file=f)
            metric_4cls.write_result(f)
            print('\n', file=f)
            print('Initial steps performance:', file=f)
            metric_initial.write_result(f)
