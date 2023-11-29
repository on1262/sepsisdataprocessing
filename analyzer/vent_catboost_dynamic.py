import numpy as np
import tools
import os
from tqdm import tqdm
from tools import logger as logger
from .container import DataContainer
from tools.data import SliceDataGenerator, LabelGenerator_cls, cal_label_weight, vent_label_func
from catboost import Pool, CatBoostClassifier
from os.path import join as osjoin
from datasets.derived_vent_dataset import MIMICIV_Vent_Dataset

class VentCatboostDynamicAnalyzer:
    def __init__(self, params:dict, container:DataContainer) -> None:
        self.params = params
        self.paths = params['paths']
        self.dataset = MIMICIV_Vent_Dataset()
        self.dataset.load_version(params['dataset_version'])
        self.model_name = self.params['analyzer_name']
        self.target_idx = self.dataset.idx_dict['vent_status']

    def run(self):
        # step 1: init variables
        out_dir = os.path.join(self.paths['out_dir'], self.model_name)
        tools.reinit_dir(out_dir, build=True)
        # metric_2cls = tools.DichotomyMetric()
        metric_2cls = tools.DichotomyMetric()
        generator = SliceDataGenerator(
            window_points=self.params['window'],
            n_fea=len(self.dataset.total_keys),
            label_generator=LabelGenerator_cls(
                centers=self.params['centers']
            ),
            label_func=vent_label_func,
            target_idx=self.target_idx,
            limit_idx=[],
            forbidden_idx=[self.dataset.idx_dict[id] for id in ['hosp_expire', 'vent_status']]
        )
        print(f'Available features: {[self.dataset.total_keys[idx] for idx in generator.avail_idx]}')
        # step 2: train and predict
        for fold_idx, (train_index, valid_index, test_index) in enumerate(self.dataset.enumerate_kf()):
            train_result = generator(self.dataset.data[train_index, :, :], self.dataset.seqs_len[train_index])
            X_train, Y_train = train_result['data'], train_result['label']

            valid_result = generator(self.dataset.data[valid_index, :, :], self.dataset.seqs_len[valid_index])
            X_valid, Y_valid = valid_result['data'], valid_result['label']

            test_result = generator(self.dataset.data[test_index, :, :], self.dataset.seqs_len[test_index])
            X_test, Y_test = test_result['data'], test_result['label']

            label_weight = cal_label_weight(len(self.params['centers']), Y_train)
            logger.info(f'label weight: {label_weight}')
            
            model = CatBoostClassifier(
                iterations=self.params['iterations'],
                learning_rate=self.params['learning_rate'],
                loss_function=self.params['loss_function'],
                class_weights=label_weight
            )
            pool_train = Pool(X_train, Y_train.argmax(axis=-1))
            pool_valid = Pool(X_valid, Y_valid.argmax(axis=-1))
            
            model.fit(pool_train, eval_set=pool_valid)

            Y_test_pred = model.predict_proba(X_test)
            metric_2cls.add_prediction(Y_test_pred[:, 1], Y_test[:, 1])
        
        metric_2cls.plot_curve(curve_type='roc', title=f'ROC for ventilation', save_path=osjoin(out_dir, f'vent_roc.png'))
        metric_2cls.plot_curve(curve_type='prc', title=f'PRC for ventilation', save_path=osjoin(out_dir, f'vent_prc.png'))

        with open(os.path.join(out_dir, 'result.txt'), 'w') as fp:
            print('Overall performance:', file=fp)
            metric_2cls.write_result(fp)