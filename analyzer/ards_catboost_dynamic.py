import numpy as np
import tools
import os
from tqdm import tqdm
from tools import logger as logger
from .container import DataContainer
from tools.data import SliceDataGenerator, LabelGenerator_cls,  map_func, cal_label_weight, label_func_min
from catboost import Pool, CatBoostClassifier
from datasets.derived_ards_dataset import MIMICIV_ARDS_Dataset

class ARDSCatboostRegressionAnalyzer:
    def __init__(self, params:dict, container:DataContainer) -> None:
        self.params = params
        self.paths = params['paths']
        self.dataset = MIMICIV_ARDS_Dataset()
        self.dataset.load_version(params['dataset_version'])
        self.model_name = self.params['analyzer_name']
        self.target_idx = self.dataset.idx_dict['PF_ratio']

    def run(self):
        # step 1: init variables
        out_dir = os.path.join(self.paths['out_dir'], self.model_name)
        tools.reinit_dir(out_dir, build=True)
        # metric_2cls = tools.DichotomyMetric()
        metric_3cls = tools.MultiClassMetric(class_names=self.params['class_names'], out_dir=out_dir)
        generator = SliceDataGenerator(
            window_points=self.params['window'],
            n_fea=len(self.dataset.total_keys),
            label_generator=LabelGenerator_cls(
                centers=self.params['centers'], 
                soft_label=self.params['soft_label'], 
                smooth_band=self.params['smoothing_band']
            ),
            label_func=label_func_min,
            target_idx=self.target_idx,
            limit_idx=[],
            forbidden_idx=[]
        )
        # step 2: train and predict
        for idx, (train_index, valid_index, test_index) in enumerate(self.dataset.enumerate_kf()):
            train_result = generator(f'{idx}_train', self.dataset.data[train_index, :, :], self.dataset.seqs_len[train_index])
            X_train, Y_train = train_result['data'], train_result['label']

            valid_result = generator(f'{idx}_valid', self.dataset.data[valid_index, :, :], self.dataset.seqs_len[valid_index])
            X_valid, Y_valid = valid_result['data'], valid_result['label']

            test_result = generator(f'{idx}_test', self.dataset.data[test_index, :, :], self.dataset.seqs_len[test_index])
            X_test, Y_test = test_result['data'], test_result['label']

            model = CatBoostClassifier(
                iterations=self.params['iterations'],
                learning_rate=self.params['learning_rate'],
                loss_function=self.params['loss_function'],
                class_weights=cal_label_weight(4, Y_train)
            )
            pool_train = Pool(X_train, Y_train.argmax(axis=-1))
            pool_valid = Pool(X_valid, Y_valid.argmax(axis=-1))
            
            model.fit(pool_train, eval_set=pool_valid)

            Y_pred = model.predict_proba(X_test)
            metric_3cls.add_prediction(Y_pred, Y_test) # 去掉mask外的数据
            # metric_2cls.add_prediction(map_func(Y_pred)[..., 1].flatten(), map_func(Y_test)[..., 1].flatten())
        
        metric_3cls.confusion_matrix(comment=self.model_name)
        # metric_2cls.plot_roc(title=f'{self.model_name} model ROC (4->2 cls)', save_path=os.path.join(out_dir, f'{self.model_name}_ROC.png'))
        
        with open(os.path.join(out_dir, 'result.txt'), 'w') as fp:
            print('Overall performance:', file=fp)
            metric_3cls.write_result(fp)