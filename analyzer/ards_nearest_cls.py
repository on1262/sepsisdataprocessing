import numpy as np
import tools
import os
from tqdm import tqdm
from tools import logger as logger
from .container import DataContainer
from tools.data import DynamicDataGenerator, LabelGenerator_cls, map_func, label_func_min
from datasets.derived_ards_dataset import MIMICIV_ARDS_Dataset


class ArdsNearest4ClsAnalyzer:
    def __init__(self, params:dict, container:DataContainer) -> None:
        self.params = params
        self.paths = params['paths']
        self.dataset = MIMICIV_ARDS_Dataset()
        self.dataset.load_version(params['dataset_version'])
        self.model_name = self.params['analyzer_name']
        self.target_idx = self.dataset.idx_dict['PF_ratio']

    def predict(self, X_test:np.ndarray):
        '''
        input: batch, n_fea, seq_len
        output: (test_batch, seq_len, n_cls)
        '''
        prediction = np.zeros((X_test.shape[0], X_test.shape[2], 4))
        target = X_test[:, self.target_idx, :]
        prediction[:,:,0] = np.logical_and(target > 0, target <= 100)
        prediction[:,:,1] = np.logical_and(target > 100, target <= 200)
        prediction[:,:,2] = np.logical_and(target > 200, target <= 300)
        prediction[:,:,3] = target > 300
        return prediction

    def train(self):
        pass

    def run(self):
        # step 1: init variables
        out_dir = os.path.join(self.paths['out_dir'], self.model_name)
        tools.reinit_dir(out_dir, build=True)
        # metric_2cls = tools.DichotomyMetric()
        metric_4cls = tools.MultiClassMetric(class_names=self.params['class_names'], out_dir=out_dir)
        generator = DynamicDataGenerator(
            window_points=self.params['window'],
            n_fea=len(self.dataset.total_keys),
            label_generator=LabelGenerator_cls(
                centers=self.params['centers']
            ),
            label_func=label_func_min,
            target_idx=self.target_idx,
            limit_idx=[],
            forbidden_idx=[]
        )
        # step 2: train and predict
        for idx, (train_index, valid_index, test_index) in enumerate(self.dataset.enumerate_kf()):
            result = generator(self.dataset.data[test_index, :, :], self.dataset.seqs_len[test_index])
            X_test, Y_mask, Y_gt = result['data'], result['mask'], result['label']
            Y_pred = self.predict(X_test)
            Y_pred = np.asarray(Y_pred)
            metric_4cls.add_prediction(Y_pred, Y_gt, Y_mask) # 去掉mask外的数据
            # metric_2cls.add_prediction(map_func(Y_pred)[..., 1].flatten()[Y_mask], map_func(Y_gt)[..., 1].flatten()[Y_mask])
        
        metric_4cls.confusion_matrix(comment=self.model_name)
        # metric_2cls.plot_curve(curve_type='roc', title=f'{self.model_name} model ROC (4->2 cls)', save_path=os.path.join(out_dir, f'{self.model_name}_ROC.png'))
        
        with open(os.path.join(out_dir, 'result.txt'), 'w') as fp:
            print('Overall performance:', file=fp)
            metric_4cls.write_result(fp)