import numpy as np
from sklearn.model_selection import KFold
import tools
import os
from tools import logger as logger
from .container import DataContainer
from .utils import generate_labels

class RandomForestAnalyzer:
    def __init__(self, params:dict, container:DataContainer) -> None:
        self.params = params
        self.paths = params['paths']
        self.container = container
        self.model_name = self.params['analyzer_name']
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
        if self.dataset.name() == 'mimic-iv':
            import models.mimic_model as mlib
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
        if self.robust:
            metric_robust = tools.RobustClassificationMetric(class_names=self.params['class_names'], out_dir=out_dir)
            def dropout_func(missrate):
                    return np.asarray(trainer.predict(mode='test', addi_params={'dropout':missrate}))
        # step 3: generate labels
        generator = mlib.StaticLabelGenerator(
            window=self.params['window'], centers=self.params['centers'],
            target_idx=self.target_idx, forbidden_idx=self.params['forbidden_idx'], limit_idx=self.params['limit_idx'])
        mask, label = generate_labels(self.dataset, self.data, generator, self.out_dir)
        fea_names = [self.dataset.get_fea_label(idx) for idx in generator.available_idx()]
        imp_logger = tools.TreeFeatureImportance(fea_names=fea_names)
        # step 4: train and predict
        for idx, (train_index, valid_index, test_index) in enumerate(self.dataset.enumerate_kf()): 
            trainer = mlib.RandomForestTrainer(self.params, self.dataset)
            if 'train_miss_rate' in self.params.keys():
                trainer.train(addi_params={'dropout':self.params['train_miss_rate']}) # 训练时对训练集随机dropout
            else:
                trainer.train()
            Y_gt = label['Y'][test_index][mask[test_index]]
            Y_pred = trainer.predict(mode='test')
            Y_pred = np.asarray(Y_pred)
            metric_4cls.add_prediction(Y_pred, Y_gt, mask[test_index])
            imp_logger.add_record(trainer.model, label['X'])
            if self.robust:
                for missrate in np.linspace(0, 1, 11):
                    R_pred = dropout_func(missrate)
                    metric_robust.add_prediction(missrate, R_pred, Y_gt, mask[test_index])
            self.dataset.mode('all') # 恢复原本状态
        # step 5: result explore
        # imp_logger.plot_beeswarm(os.path.join(out_dir, 'shap_overview.png'))
        # single_imp_out = os.path.join(out_dir, 'single_shap')
        # tools.reinit_dir(single_imp_out, build=True)
        # imp_logger.plot_single_importance(out_dir=single_imp_out, select=10)
        if self.robust:
            metric_robust.plot_curve()
        metric_4cls.confusion_matrix(comment=self.model_name)
        with open(os.path.join(self.out_dir, 'result.txt'), 'a') as f:
            metric_4cls.write_result(f)
