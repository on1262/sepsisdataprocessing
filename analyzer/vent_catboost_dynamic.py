import numpy as np
import tools
import os
from tqdm import tqdm
from tools import logger as logger
from .container import DataContainer
from tools.data import SliceDataGenerator, DynamicDataGenerator, LabelGenerator_cls, cal_label_weight, vent_label_func
from tools.feature_importance import TreeFeatureImportance
from catboost import Pool, CatBoostClassifier
from os.path import join as osjoin
from datasets.derived_vent_dataset import MIMICIV_Vent_Dataset

import matplotlib.pyplot as plt

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
            limit_idx=[self.dataset.fea_idx(id) for id in self.params['limit_feas']],
            forbidden_idx=[self.dataset.fea_idx(id) for id in self.params['forbidden_feas']]
        )
        feature_names = [self.dataset.fea_label(idx) for idx in generator.avail_idx]
        print(f'Available features: {feature_names}')
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
                class_weights=label_weight,
                use_best_model=True
            )
            pool_train = Pool(X_train, Y_train.argmax(axis=-1))
            pool_valid = Pool(X_valid, Y_valid.argmax(axis=-1))
            
            model.fit(pool_train, eval_set=pool_valid)

            Y_test_pred = model.predict_proba(X_test)
            metric_2cls.add_prediction(Y_test_pred[:, 1], Y_test[:, 1])
            if fold_idx == 0:
                # plot sample
                self.plot_examples(test_index, model, 20, osjoin(out_dir, 'samples'))
                explorer = TreeFeatureImportance(map_func=lambda x:x[:, :, 1], fea_names=feature_names, missvalue=-1, n_approx=-1)
                explorer.add_record(model, valid_X=X_valid)
                explorer.plot_beeswarm(max_disp=10, plot_path=osjoin(out_dir, f'importance.png'))
        
        metric_2cls.plot_curve(curve_type='roc', title=f'ROC for ventilation', save_path=osjoin(out_dir, f'vent_roc.png'))
        metric_2cls.plot_curve(curve_type='prc', title=f'PRC for ventilation', save_path=osjoin(out_dir, f'vent_prc.png'))

        with open(os.path.join(out_dir, 'result.txt'), 'w') as fp:
            print('Overall performance:', file=fp)
            metric_2cls.write_result(fp)
    
    def plot_examples(self, test_index, model:CatBoostClassifier, n_sample:int, out_dir:str):
        tools.reinit_dir(out_dir)
        generator = DynamicDataGenerator(
            window_points=self.params['window'],
            n_fea=len(self.dataset.total_keys),
            label_generator=LabelGenerator_cls(
                centers=self.params['centers']
            ),
            label_func=vent_label_func,
            target_idx=self.target_idx,
            limit_idx=[],
            forbidden_idx=[self.dataset.idx_dict[id] for id in ['vent_status']]
        )
        test_result = generator(self.dataset.data[test_index, :, :], self.dataset.seqs_len[test_index])
        X_test, mask, Y_test = test_result['data'], test_result['mask'], test_result['label']
        # random sample 10 sequences
        seq_index = np.arange(len(Y_test))[np.logical_and(np.max(Y_test[:, :, 1], axis=-1) > 0.5, np.min(Y_test[:, :, 1], axis=-1) < 0.5)]
        seq_index = seq_index[np.random.choice(len(seq_index), n_sample)]
        origin_label = self.dataset.data[test_index[seq_index], self.target_idx, :] # origin label
        X_test = X_test[seq_index, :, :]
        for idx in range(n_sample):
            seq_mask = mask[seq_index[idx], :]
            Y_origin = np.clip(origin_label[idx, seq_mask], 0, 1) # 2->1
            Y_pred = model.predict_proba(X_test[idx, :, :].T)[seq_mask, 1]
            Y_target = Y_test[seq_index[idx], seq_mask, 1]
            plt.figure(figsize=(8, 8))
            ax = plt.gca()
            ax.plot(Y_origin, '-o', color='C0', label='origin ventilation status')
            ax.plot(Y_pred, '-o', color='C1', label='prediction probability')
            ax.plot(Y_target, '-o', color='C2', label='prediction target')
            ax.legend()
            plt.axhline(0.5, xmin=0, xmax=np.sum(seq_mask))
            plt.savefig(osjoin(out_dir, f'{idx}.png'))
            plt.close()
            



