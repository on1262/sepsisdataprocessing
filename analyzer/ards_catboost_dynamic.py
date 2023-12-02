import numpy as np
import tools
import os
from os.path import join as osjoin
from tqdm import tqdm
from tools import logger as logger
from .container import DataContainer
from tools.data import SliceDataGenerator, LabelGenerator_cls, cal_label_weight, label_func_min, map_func
from catboost import Pool, CatBoostClassifier
from tools.feature_importance import TreeFeatureImportance
from datasets.derived_ards_dataset import MIMICIV_ARDS_Dataset

class ARDSCatboostAnalyzer:
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
        metric_4cls = tools.MultiClassMetric(class_names=self.params['class_names'], out_dir=out_dir)
        metric_2cls = [tools.DichotomyMetric() for _ in range(4)]
        generator = SliceDataGenerator(
            window_points=self.params['window'],
            n_fea=len(self.dataset.total_keys),
            label_generator=LabelGenerator_cls(
                centers=self.params['centers']
            ),
            label_func=label_func_min,
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

            model = CatBoostClassifier(
                iterations=self.params['iterations'],
                learning_rate=self.params['learning_rate'],
                loss_function=self.params['loss_function'],
                class_weights=cal_label_weight(len(self.params['centers']), Y_train),
                use_best_model=True
            )
            pool_train = Pool(X_train, Y_train.argmax(axis=-1))
            pool_valid = Pool(X_valid, Y_valid.argmax(axis=-1))
            
            model.fit(pool_train, eval_set=pool_valid)

            Y_test_pred = model.predict_proba(X_test)
            metric_4cls.add_prediction(Y_test_pred, Y_test) # 去掉mask外的数据
            for idx, map_dict in zip([0,1,2,3], [{0:0,1:1,2:1,3:1}, {0:0,1:1,2:0,3:0}, {0:0,1:0,2:1,3:0}, {0:0,1:0,2:0,3:1}]): # TODO 这里写错了
                metric_2cls[idx].add_prediction(map_func(Y_test_pred, map_dict)[:, 1], map_func(Y_test, map_dict)[:, 1])
            if fold_idx == 0:
                explorer = TreeFeatureImportance(map_func=lambda x:x[:, :, 1], fea_names=feature_names, missvalue=-1, n_approx=2000)
                explorer.add_record(model, valid_X=X_valid)
                explorer.plot_beeswarm(max_disp=10, plot_path=osjoin(out_dir, f'importance.png'))
        
        metric_4cls.confusion_matrix(comment=self.model_name)
        for idx in range(4):
            metric_2cls[idx].plot_curve(curve_type='roc', title=f'ROC for {self.params["class_names"][idx]}', save_path=osjoin(out_dir, f'roc_cls_{idx}.png'))
        
        with open(os.path.join(out_dir, 'result.txt'), 'w') as fp:
            print('Overall performance:', file=fp)
            metric_4cls.write_result(fp)