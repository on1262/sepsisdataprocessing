import numpy as np
from sklearn.model_selection import KFold
import tools
import os
from tqdm import tqdm
from tools import logger as logger
from .container import DataContainer
from .utils import generate_labels, map_func
from .explore import plot_cover_rate

class LSTM4ClsAnalyzer:
    '''动态模型, 四分类预测'''
    def __init__(self, params:dict, container:DataContainer) -> None:
        self.params = params
        self.paths = params['paths']
        self.container = container
        self.model_name = 'LSTM_4cls'
        self.loss_logger = tools.LossLogger()
        # copy attribute from container
        self.target_idx = container.dataset.target_idx
        self.dataset = container.dataset
        self.data = self.dataset.data
        # initialize
        self.out_dir = os.path.join(self.paths['out_dir'], self.model_name)
        tools.reinit_dir(self.out_dir, build=True)

    def label_explore(self, label, mask):
        logger.info('Label explore')
        # 2 class
        cls2_label = map_func(label)
        cls2_out = os.path.join(self.out_dir, '2cls')
        tools.reinit_dir(cls2_out, build=True)
        plot_cover_rate(['No_ARDS', 'ARDS'], cls2_label, mask, cls2_out)
        # 4 class
        cls4_out = os.path.join(self.out_dir, '4cls')
        tools.reinit_dir(cls4_out, build=True)
        plot_cover_rate(['Severe','Moderate', 'Mild', 'No_ARDS'], label, mask, cls4_out)


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
        metric_4cls = tools.MultiClassMetric(class_names=self.params['class_names'], out_dir=out_dir)
        # step 3: generate labels & label explore
        generator = mlib.Cls4LabelGenerator(window=self.params['window'], centers=self.params['centers'], smoothing_band=self.params['smoothing_band'])
        mask, label = generate_labels(self.dataset, self.data, self.target_idx, generator, self.out_dir)
        self.label_explore(label, mask)
        # step 4: train and predict
        for idx, (data_index, test_index) in enumerate(kf.split(X=self.dataset)): 
            valid_num = round(len(data_index)*0.15)
            train_index, valid_index = data_index[valid_num:], data_index[:valid_num]
            self.dataset.register_split(train_index, valid_index, test_index)
            trainer = mlib.LSTMClsTrainer(self.params, self.dataset)
            if idx == 0:
                trainer.summary()
            trainer.train()
            self.loss_logger.add_loss(trainer.get_loss())
            Y_mask = mask[test_index, ...]
            Y_gt = label[test_index, ...]
            Y_pred = trainer.predict(mode='test')
            Y_pred = np.asarray(Y_pred)
            metric_4cls.add_prediction(Y_pred, Y_gt, Y_mask) # 去掉mask外的数据
            metric_2cls.add_prediction(map_func(Y_pred)[..., 1][Y_mask][:], map_func(Y_gt)[..., 1][Y_mask][:])
            self.dataset.mode('all') # 恢复原本状态
        # step 5: result explore
        self.loss_logger.plot(std_bar=False, log_loss=False, title='Loss for LSTM cls Model', 
            out_path=os.path.join(out_dir, 'loss.png'))
        metric_4cls.confusion_matrix(comment=self.model_name)
        metric_4cls.write_result()

        metric_2cls.plot_roc(title=f'{self.model_name} model ROC (4->2 cls)', save_path=os.path.join(out_dir, f'{self.model_name}_ROC.png'))
        print('Metric 2 classes:')
        print(metric_2cls.generate_info())

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

    def predict(self, mode:str):
        '''
        input: mode: ['test']
        output: (test_batch, seq_len, n_cls)
        '''
        self.dataset.mode(mode)
        pred = np.zeros((len(self.dataset), self.dataset.data.shape[-1], len(self.centers)))
        for idx, data in tqdm(enumerate(self.dataset), desc='testing', total=len(self.dataset)):
            np_data = data['data']
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
        metric_2cls = tools.DichotomyMetric()
        metric_4cls = tools.MultiClassMetric(class_names=self.params['class_names'], out_dir=out_dir)
        # step 3: generate labels
        generator = mlib.Cls4LabelGenerator(window=self.params['window'], centers=self.params['centers'], smoothing_band=self.params['smoothing_band'])
        mask, label = generate_labels(self.dataset, self.dataset.data, self.target_idx, generator, out_dir)
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
            metric_2cls.add_prediction(map_func(Y_pred)[..., 1][Y_mask][:], map_func(Y_gt)[..., 1][Y_mask][:])
            self.dataset.mode('all') # 恢复原本状态
        metric_4cls.confusion_matrix(comment=self.model_name)
        metric_4cls.write_result()
        metric_2cls.plot_roc(title=f'{self.model_name} model ROC (4->2 cls)', save_path=os.path.join(out_dir, f'{self.model_name}_ROC.png'))
        print('Metric 2 classes:')
        print(metric_2cls.generate_info())


def explore_result(ards_threshold, Y_pred, Y_gt, mask, out_dir, cmt):
    '''
    输出二分类误差和flips的统计关系, 观察误差大的样本是否存在特殊的分布
    Y_pred, Y_gt: (batch, seq_lens), 值域只能是[0,1]
    '''
    delta = np.abs(Y_pred - Y_gt)
    cover = (Y_gt > 0) * (Y_gt < ards_threshold) * mask
    diffs = np.diff(cover.astype(int), axis=1)
    # count the number of flips
    num_flips = np.count_nonzero(diffs, axis=1)
    num_flips = np.repeat(num_flips[:, None], Y_pred.shape[1], axis=1)
    mask = mask[:]
    num_flips = num_flips[:][mask]
    delta = delta[:][mask][:, None]
    tools.plot_reg_correlation(
        X=delta, fea_names=['Prediction Abs Error'], Y=num_flips, target_name='Num flips', adapt=True, write_dir_path=out_dir, plot_dash=False, comment=cmt)