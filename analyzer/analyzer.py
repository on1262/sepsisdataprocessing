import numpy as np
import tools
import os
from tools import logger as logger
from .container import DataContainer
from .explore import FeatureExplorer
from .method_4cls import LSTM4ClsAnalyzer, BaselineNearestClsAnalyzer, EnsembleClsAnalyzer
from .method_reg import LSTMRegAnalyzer, BaselineNearestRegAnalyzer
from .method_2cls import Catboost2ClsAnalyzer
from .method_quantile import LSTMQuantileAnalyzer
from datasets import AbstractDataset


class Analyzer:
    def __init__(self, params:list, dataset:AbstractDataset) -> None:
        '''
        params: 启动脚本, 否则需要手动run_sub_analyzer, 可以是None
        dataset: 数据集
        '''
        self.container = DataContainer(dataset)
        self.explorer = FeatureExplorer(self.container)
        self.analyzer_dict = {
            'LSTM_4cls':LSTM4ClsAnalyzer,
            'nearest_4cls': BaselineNearestClsAnalyzer,
            'LSTM_reg': LSTMRegAnalyzer,
            'nearest_reg': BaselineNearestRegAnalyzer,
            'catboost_2cls':Catboost2ClsAnalyzer,
            'LSTM_quantile': LSTMQuantileAnalyzer,
            'ensemble_4cls': EnsembleClsAnalyzer
        }
        if params is not None:
            for key in params:
                if key in self.analyzer_dict:
                    self.run_sub_analyzer(key)
                elif key == 'feature_explore':
                    self.feature_explore()

        
    def run_sub_analyzer(self, analyzer_name):
        logger.info(f'Run Analyzer: {analyzer_name}')
        params = self.container.get_model_params(analyzer_name)
        sub_analyzer = self.analyzer_dict[analyzer_name](params, self.container)
        sub_analyzer.run()
        # utils.create_final_result()


    def feature_explore(self):
        '''输出mimic-iv数据集的统计特征, 独立于模型和研究方法'''
        logger.info('Analyzer: Feature explore')
        out_dir = os.path.join(tools.GLOBAL_CONF_LOADER['analyzer'][self.container.dataset.name()]['paths']['out_dir'], 'feature_explore')
        tools.reinit_dir(out_dir, build=True)
        # random plot sample time series
        self.explorer.plot_time_series_samples(self.container.target_name, n_sample=400, n_per_plots=40, write_dir=os.path.join(out_dir, "target_plot"))
        # self.explorer.plot_time_series_samples("220224", n_sample=400, n_per_plots=40, write_dir=os.path.join(out_dir, "pao2_plot"))
        # self.explorer.plot_time_series_samples("223835", n_sample=400, n_per_plots=40, write_dir=os.path.join(out_dir, "fio2_plot"))
        self.explorer.plot_samples(num=50, id_list=["220224", "223835"], id_names=['PaO2', 'FiO2'], out_dir=os.path.join(out_dir, 'samples'))
        # plot other information
        self.explorer.miss_mat(out_dir)
        self.explorer.first_ards_time(out_dir)
        self.explorer.feature_count(out_dir)