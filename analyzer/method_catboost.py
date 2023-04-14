import numpy as np
from sklearn.model_selection import KFold
import tools
import os
from tools import logger as logger
from .container import DataContainer
from .utils import generate_labels

class CatboostAnalyzer:
    def __init__(self, params:dict, container:DataContainer) -> None:
        self.params = params
        self.paths = params['paths']
        self.container = container
        self.model_name = 'catboost_4cls'
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
        if self.dataset.name() == 'mimic-iv':
            import models.mimic_model as mlib
        # step 1: append additional params
        self.params['in_channels'] = self.dataset.data.shape[1]
        forbidden_idx = {self.dataset.idx_dict[name] for name in self.params['forbidden_feas']}
        limit_idx = {self.dataset.idx_dict[name] for name in self.params['feature_limit']}
        self.params['forbidden_idx'] = forbidden_idx
        self.params['limit_idx'] = limit_idx
        # step 2: init variables
        kf = KFold(n_splits=self.container.n_fold, shuffle=True, random_state=self.container.seed)
        out_dir = os.path.join(self.paths['out_dir'], self.model_name)
        tools.reinit_dir(out_dir, build=True)
        metric_4cls = tools.MultiClassMetric(class_names=self.params['class_names'], out_dir=out_dir)
        # step 3: generate labels
        generator = mlib.StaticLabelGenerator(
            window=self.params['window'], centers=self.params['centers'],
            target_idx=self.target_idx, forbidden_idx=self.params['forbidden_idx'], limit_idx=self.params['limit_idx'])
        mask, label = generate_labels(self.dataset, self.data, generator, self.out_dir)
        fea_names = [self.dataset.get_fea_label(idx) for idx in generator.available_idx()]
        imp_logger = tools.SHAPFeatureImportance(fea_names=fea_names, model_type='gbdt')
        # step 4: train and predict
        for _, (data_index, test_index) in enumerate(kf.split(X=self.dataset)): 
            valid_num = round(len(data_index)*0.15)
            train_index, valid_index = data_index[valid_num:], data_index[:valid_num]
            self.dataset.register_split(train_index, valid_index, test_index)
            trainer = mlib.CatboostTrainer(self.params, self.dataset)
            trainer.train()
            self.loss_logger.add_loss(trainer.get_loss())
            Y_gt = label['Y'][test_index][mask[test_index]]
            Y_pred = trainer.predict(mode='test')
            Y_pred = np.asarray(Y_pred)
            metric_4cls.add_prediction(Y_pred, Y_gt, mask[test_index])
            imp_logger.add_record(trainer.model, label['X'])
            self.dataset.mode('all') # 恢复原本状态
        # step 5: result explore
        # imp_logger.plot_beeswarm(os.path.join(out_dir, 'shap_overview.png'))
        single_imp_out = os.path.join(out_dir, 'single_shap')
        tools.reinit_dir(single_imp_out, build=True)
        # imp_logger.plot_single_importance(out_dir=single_imp_out, select=10)
        self.loss_logger.plot(std_bar=False, log_loss=False, title='Loss for Catboost cls Model', 
            out_path=os.path.join(out_dir, 'loss.png'))
        metric_4cls.confusion_matrix(comment=self.model_name)
        with open(os.path.join(self.out_dir, 'result.txt'), 'a') as f:
            metric_4cls.write_result(f)
