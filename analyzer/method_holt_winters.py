import numpy as np
from sklearn.model_selection import KFold
import tools
import os
from tqdm import tqdm
from tools import logger as logger
from .container import DataContainer
from .utils import generate_labels, map_func


class HoltWintersDynamicAnalyzer:
    '''window=144时start step就是静态模型baseline, window=16时overall performance'''
    def __init__(self, params:dict, container:DataContainer) -> None:
        self.params = params
        self.paths = params['paths']
        self.dataset = container.dataset
        self.container= container
        self.target_idx = self.dataset.target_idx
        self.model_name = self.params['analyzer_name']
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
            np_data = data['data'] # (n_feature, seq_len)
            np_data = np_data[-1, :] # (seq_len,)
            t_pred = np.zeros((len(np_data), 16))
            alpha = 0.8 # smoothing factor
            beta = 0.2 # trend factor
            gamma = 0.2 # seasonal factor
            level = np.zeros(len(np_data))
            trend = np.zeros(len(np_data))
            season = np.zeros(16)
            # initialize level and trend
            level[0] = np_data[0]
            trend[0] = np_data[1] - np_data[0]

            # smoothing, trend and seasonality calculation
            for i in range(1, len(np_data)):
                if i < 16:
                    season[i] = np.mean(np_data[i::16])
                else:
                    season[i%16] = gamma*(np_data[i] - level[i-1] - trend[i-1]) + (1-gamma)*season[i%16]
                level[i] = alpha*(np_data[i] - season[i%16]) + (1-alpha)*(level[i-1] + trend[i-1])
                trend[i] = beta*(level[i] - level[i-1]) + (1-beta)*trend[i-1]

            # Holt-Winters forecasting
            for i in range(len(np_data)):
                for j in range(16):
                    if i == 0:
                        t_pred[i][j] = level[0]
                    else:
                        t_pred[i][j] = level[i-1] + trend[i-1]*j + season[(i+j)%16]

            min_pred = np.min(t_pred, axis=1) # get min prediction in window
            pred[idx, :, :] = tools.label_smoothing(self.centers, min_pred, band=50)
        return pred

    def run(self):
        if self.dataset.name() == 'mimic-iv':
            import models.mimic_model as mlib
        # step 1: append additional params
        self.params['in_channels'] = self.dataset.data.shape[1]
        # step 2: init variables
        out_dir = os.path.join(self.paths['out_dir'], self.model_name)
        tools.reinit_dir(out_dir, build=True)
        #metric_2cls = tools.DichotomyMetric()
        metric_4cls = tools.MultiClassMetric(class_names=self.params['class_names'], out_dir=out_dir)
        metric_startstep = tools.MultiClassMetric(class_names=self.params['class_names'], out_dir=out_dir)
        # step 3: generate labels
        generator = mlib.DynamicLabelGenerator(window=self.params['window'], centers=self.params['centers'], smoothing_band=self.params['smoothing_band'])
        mask, label = generate_labels(self.dataset, self.dataset.data, generator, out_dir)
        # step 4: train and predict
        for idx, (train_index, valid_index, test_index) in enumerate(self.dataset.enumerate_kf()): 
            Y_mask = mask[test_index, ...]
            Y_gt = label[test_index, ...]
            Y_pred = self.predict(mode='test')
            Y_pred = np.asarray(Y_pred)
            metric_4cls.add_prediction(Y_pred, Y_gt, Y_mask) # 去掉mask外的数据
            metric_startstep.add_prediction(Y_pred[:,0, ...], Y_gt[:,0, ...], Y_mask[:,0, ...])
            #metric_2cls.add_prediction(map_func(Y_pred)[..., 1][Y_mask][:], map_func(Y_gt)[..., 1][Y_mask][:])
            self.dataset.mode('all') # 恢复原本状态
        metric_4cls.confusion_matrix(comment=self.model_name)
        #metric_2cls.plot_roc(title=f'{self.model_name} model ROC (4->2 cls)', save_path=os.path.join(out_dir, f'{self.model_name}_ROC.png'))
        with open(os.path.join(out_dir, 'result.txt'), 'w') as fp:
            print('Overall performance:', file=fp)
            metric_4cls.write_result(fp)
            print('Start step performance:', file=fp)
            metric_startstep.write_result(fp)