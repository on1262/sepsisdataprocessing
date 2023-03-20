import torch
import torchinfo
import numpy as np
from datasets.mimic_dataset import MIMICDataset, Subject, Admission, Config # 这个未使用的import是pickle的bug
import models.mimic_model as mimic_model
from sklearn.model_selection import KFold

import tools
import os
from tqdm import tqdm
from tools import logger as logger


# 鉴于mimic数据集提取后的大小和数量都和之前的数据集在同一规模, 所以这里决定写一个接口(继承)
# 直接复用dynamic_analyzer的代码
class Analyzer:
    def __init__(self) -> None:
        pass

    def feature_explore(self):
        '''输出mimic-iv数据集的统计特征'''
        logger.info('Analyzer: Feature explore')
        out_dir = self.gbl_conf['paths']['out_dir']
        
        # random plot sample time series
        self._plot_time_series_samples(self.target_name, n_sample=400, n_per_plots=40, write_dir=os.path.join(out_dir, "target_plot"))
        self._plot_time_series_samples("220224", n_sample=400, n_per_plots=40, write_dir=os.path.join(out_dir, "pao2_plot"))
        self._plot_time_series_samples("223835", n_sample=400, n_per_plots=40, write_dir=os.path.join(out_dir, "fio2_plot"))
        self._plot_samples(num=50, id_list=["220224", "223835"], id_names=['PaO2', 'FiO2'], out_dir=os.path.join(out_dir, 'samples'))

        # self._miss_mat()
        # self._first_ards_time()
        #self._feature_count()