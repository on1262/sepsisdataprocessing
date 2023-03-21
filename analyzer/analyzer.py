import torch
import torchinfo
import numpy as np
import models.mimic_model as mimic_model

import tools
import os
from tqdm import tqdm
from tools import logger as logger
from .container import DataContainer
from .explore import FeatureExplorer
from .method_4cls import LSTM4ClsAnalyzer, BaselineNearestClsAnalyzer
from .method_reg import LSTMRegAnalyzer, BaselineNearestClsAnalyzer
from datasets import AbstractDataset


class Analyzer:
    def __init__(self, dataset:AbstractDataset) -> None:
        self.container = DataContainer(dataset)
        self.explorer = FeatureExplorer(self.container)
        self.analyzer_dict= {
            'lstm_4cls':LSTM4ClsAnalyzer,
            'nearest_4cls': BaselineNearestClsAnalyzer,
            'lstm_reg': LSTMRegAnalyzer,
            'nearest_reg': BaselineNearestClsAnalyzer
        }
        
    def run_sub_analyzer(self, analyzer_name):
        params = self.container.get_model_params(analyzer_name)
        sub_analyzer = self.analyzer_dict[analyzer_name](params, self.container.dataset)
        sub_analyzer.run()
        # utils.create_final_result()


    def feature_explore(self):
        '''输出mimic-iv数据集的统计特征, 独立于模型和研究方法'''
        logger.info('Analyzer: Feature explore')
        out_dir = self.container._conf['paths']['out_dir']
        # random plot sample time series
        self.explorer.plot_time_series_samples(self.target_name, n_sample=400, n_per_plots=40, write_dir=os.path.join(out_dir, "target_plot"))
        self.explorer.plot_time_series_samples("220224", n_sample=400, n_per_plots=40, write_dir=os.path.join(out_dir, "pao2_plot"))
        self.explorer.plot_time_series_samples("223835", n_sample=400, n_per_plots=40, write_dir=os.path.join(out_dir, "fio2_plot"))
        self.explorer.plot_samples(num=50, id_list=["220224", "223835"], id_names=['PaO2', 'FiO2'], out_dir=os.path.join(out_dir, 'samples'))
        # plot other information
        self.explorer.miss_mat()
        self.explorer.first_ards_time()
        self.explorer.feature_count()