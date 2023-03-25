import numpy as np
from sklearn.model_selection import KFold
import tools
import os
from tools import logger as logger
from .container import DataContainer
from .utils import generate_labels



class Catboost2ClsAnalyzer:
    def __init__(self, params:dict, container:DataContainer) -> None:
        self.params = params
        self.paths = params['paths']
        self.container = container
        self.model_name = 'catboost_2cls'
        self.loss_logger = tools.LossLogger()
        # copy attribute from container
        self.target_idx = container.dataset.target_idx
        self.dataset = container.dataset
        self.data = self.dataset.data
        # initialize
        self.out_dir = os.path.join(self.paths['out_dir'], self.model_name)
        tools.reinit_dir(self.out_dir, build=True)

    def label_explore(self, label, mask):
        logger.info(f'2cls available labels: {np.sum(mask)}')
        postive_rate = np.mean(label['Y'][mask])
        logger.info(f'Positive Label: {postive_rate}')
    

    def run(self):
        '''预测窗口内是否发生ARDS的分类器'''
        if self.dataset.name() == 'mimic-iv':
            import models.mimic_model as mlib
        # step 1: append additional params
        self.params['in_channels'] = self.dataset.data.shape[1]
        # step 2: init variables
        kf = KFold(n_splits=self.container.n_fold, shuffle=True, random_state=self.container.seed)
        out_dir = os.path.join(self.paths['out_dir'], self.model_name)
        tools.reinit_dir(out_dir, build=True)
        metric_2cls = tools.DichotomyMetric()
        # step 3: generate labels
        forbidden_idx = {self.dataset.idx_dict[name] for name in self.params['forbidden_feas']}
        generator = mlib.Cls2LabelGenerator(
            window=self.params['window'], ards_threshold=self.params['ards_threshold'],
            target_idx=self.target_idx,  sepsis_time_idx=self.dataset.idx_dict['sepsis_time'],
            post_sepsis_time=self.params['max_post_sepsis_hour'], forbidden_idx=forbidden_idx)
        mask, label = generate_labels(self.dataset, self.data, generator, self.out_dir)
        self.label_explore(label, mask)
        # step 4: train and predict
        for idx, (data_index, test_index) in enumerate(kf.split(X=self.dataset)): 
            valid_num = round(len(data_index)*0.15)
            train_index, valid_index = data_index[valid_num:], data_index[:valid_num]
            self.dataset.register_split(train_index, valid_index, test_index)
            trainer = mlib.CatboostClsTrainer(self.params, self.dataset)
            trainer.train()
            self.loss_logger.add_loss(trainer.get_loss())
            Y_gt = label['Y'][test_index][mask[test_index]]
            Y_pred = trainer.predict(mode='test')
            Y_pred = np.asarray(Y_pred)
            metric_2cls.add_prediction(Y_pred, Y_gt)
            if idx == 0:
                total_names = [self.dataset.get_fea_label(key) for key in self.dataset.total_keys]
                tools.cal_feature_importance(trainer.model, label['X'], total_names, os.path.join(out_dir, 'shap.png'), model_type='gbdt')
            self.dataset.mode('all') # 恢复原本状态
        # step 5: result explore
        self.loss_logger.plot(std_bar=False, log_loss=False, title='Loss for Catboost cls Model', 
            out_path=os.path.join(out_dir, 'loss.png'))
        metric_2cls.plot_roc(title=f'{self.model_name} model ROC (4->2 cls)', save_path=os.path.join(out_dir, f'{self.model_name}_ROC.png'))
        print('Metric 2 classes:')
        print(metric_2cls.generate_info())
