import numpy as np
from sklearn.model_selection import KFold
import tools
import os
from tqdm import tqdm
from tools import logger as logger
from .container import DataContainer
from .utils import generate_labels, map_func


class BaselineNearestClsAnalyzer:
    def __init__(self, params:dict, container:DataContainer) -> None:
        self.params = params
        self.paths = params['paths']
        self.dataset = container.dataset
        self.container= container
        self.target_idx = self.dataset.target_idx
        self.model_name = 'nearest_4cls'
        # copy params
        self.centers = params['centers']
        self.n_cls = len(self.centers)

    def predict(self, mode:str):
        '''
        input: mode: ['test']
        output: (test_batch, seq_len, n_cls)
        '''
        self.dataset.mode(mode)
        pred = np.zeros((len(self.dataset), self.dataset.data.shape[-1], len(self.centers)))
        for idx, data in tqdm(enumerate(self.dataset), desc='testing', total=len(self.dataset)):
            np_data = data['data']
            np_data = np_data + (-10)*(np_data < 150) - 10
            pred[idx, :, :] = tools.label_smoothing(self.centers, np_data[self.target_idx, :], band=50)
        return pred

    def run(self):
        if self.dataset.name() == 'mimic-iv':
            import models.mimic_model as mlib
        # step 1: append additional params
        self.params['in_channels'] = self.dataset.data.shape[1]
        # step 2: init variables
        kf = KFold(n_splits=self.container.n_fold, shuffle=True, random_state=self.container.seed)
        out_dir = os.path.join(self.paths['out_dir'], self.model_name)
        tools.reinit_dir(out_dir, build=True)
        #metric_2cls = tools.DichotomyMetric()
        metric_4cls = tools.MultiClassMetric(class_names=self.params['class_names'], out_dir=out_dir)
        # step 3: generate labels
        generator = mlib.DynamicLabelGenerator(window=self.params['window'], centers=self.params['centers'], smoothing_band=self.params['smoothing_band'])
        mask, label = generate_labels(self.dataset, self.dataset.data, generator, out_dir)
        # step 4: train and predict
        for _, (data_index, test_index) in enumerate(kf.split(X=self.dataset)): 
            valid_num = round(len(data_index)*0.15)
            train_index, valid_index = data_index[valid_num:], data_index[:valid_num]
            self.dataset.register_split(train_index, valid_index, test_index)
            Y_mask = mask[test_index, ...]
            Y_gt = label[test_index, ...]
            Y_pred = self.predict(mode='test')
            Y_pred = np.asarray(Y_pred)
            metric_4cls.add_prediction(Y_pred, Y_gt, Y_mask) # 去掉mask外的数据
            #metric_2cls.add_prediction(map_func(Y_pred)[..., 1][Y_mask][:], map_func(Y_gt)[..., 1][Y_mask][:])
            self.dataset.mode('all') # 恢复原本状态
        metric_4cls.confusion_matrix(comment=self.model_name)
        #metric_2cls.plot_roc(title=f'{self.model_name} model ROC (4->2 cls)', save_path=os.path.join(out_dir, f'{self.model_name}_ROC.png'))
        with open(os.path.join(out_dir, 'result.txt'), 'w') as fp:
            print('Overall performance:', file=fp)
            metric_4cls.write_result(fp)