import numpy as np
import tools
import os
from tqdm import tqdm
from tools import logger as logger
from .container import DataContainer
from tools.data import SliceDataGenerator, LabelGenerator_cls, cal_label_weight, vent_label_func, Normalization
from sklearn.linear_model import LogisticRegression
from os.path import join as osjoin
from datasets.derived_vent_dataset import MIMICIV_Vent_Dataset

class VentLogisticRegAnalyzer:
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
            norm=Normalization(self.dataset.norm_dict, self.dataset.total_keys),
            label_func=vent_label_func,
            target_idx=self.target_idx,
            limit_idx=[],
            forbidden_idx=[self.dataset.idx_dict[id] for id in ['vent_status']]
        )
        print(f'Available features: {[self.dataset.total_keys[idx] for idx in generator.avail_idx]}')
        # step 2: train and predict
        for fold_idx, (train_index, valid_index, test_index) in enumerate(self.dataset.enumerate_kf()):
            reg_train_index = np.concatenate([train_index, valid_index], axis=0) # lineat regression do not need validation
            train_result = generator(self.dataset.data[reg_train_index, :, :], self.dataset.seqs_len[reg_train_index])
            X_train, Y_train = train_result['data'], train_result['label']

            test_result = generator(self.dataset.data[test_index, :, :], self.dataset.seqs_len[test_index])
            X_test, Y_test = test_result['data'], test_result['label']

            class_weight = cal_label_weight(len(self.params['centers']), Y_train)
            class_weight = {idx:class_weight[idx] for idx in range(len(class_weight))}
            logger.info(f'class weight: {class_weight}')
            
            model = LogisticRegression(max_iter=self.params['max_iter'], multi_class='multinomial', class_weight=class_weight)
            model.fit(X_train, Y_train[:, 1])

            Y_test_pred = model.predict_proba(X_test)
            metric_2cls.add_prediction(Y_test_pred[:, 1], Y_test[:, 1])
        
        metric_2cls.plot_curve(curve_type='roc', title=f'ROC for ventilation', save_path=osjoin(out_dir, f'vent_roc.png'))
        metric_2cls.plot_curve(curve_type='prc', title=f'PRC for ventilation', save_path=osjoin(out_dir, f'vent_prc.png'))

        with open(os.path.join(out_dir, 'result.txt'), 'w') as fp:
            print('Overall performance:', file=fp)
            metric_2cls.write_result(fp)