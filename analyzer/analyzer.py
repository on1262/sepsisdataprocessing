import numpy as np
import tools
import os
from tools import logger as logger
from .container import DataContainer
from .explore import FeatureExplorer
from .method_lstm_original import LSTMOriginalAnalyzer
from .method_lstm_balanced import LSTMBalancedAnalyzer
from .method_nearest_cls import BaselineNearestClsAnalyzer
from .method_catboost_cls import CatboostAnalyzer
from .method_random_forest import RandomForestAnalyzer
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
            'LSTM_original':LSTMOriginalAnalyzer,
            "LSTM_balanced":LSTMBalancedAnalyzer,
            'nearest_4cls': BaselineNearestClsAnalyzer,
            'catboost_4cls':CatboostAnalyzer,
            'random_forest':RandomForestAnalyzer,
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
        # plot correlation matrix
        labels = [self.container.dataset.get_fea_label(id) for id in self.container.dataset.total_keys]
        corr_mat = tools.plot_correlation_matrix(self.container.data[:, :, 0], labels, save_path=os.path.join(out_dir, 'correlation matrix'))
        correlations = []
        for idx in range(corr_mat.shape[1]):
            correlations.append([corr_mat[-1, idx], labels[idx]])
        correlations = sorted(correlations, key=lambda x:np.abs(x[0]), reverse=True)
        for idx in range(corr_mat.shape[1]):
            logger.info(f'Correlation with target: {correlations[idx][0]} \t{correlations[idx][1]}')
        # random plot sample time series
        # self.explorer.plot_time_series_samples(self.container.target_name, n_sample=400, n_per_plots=40, write_dir=os.path.join(out_dir, "target_plot"))
        # self.explorer.plot_time_series_samples("220224", n_sample=400, n_per_plots=40, write_dir=os.path.join(out_dir, "pao2_plot"))
        # self.explorer.plot_time_series_samples("223835", n_sample=400, n_per_plots=40, write_dir=os.path.join(out_dir, "fio2_plot"))
        # self.explorer.plot_samples(num=50, id_list=["220224", "223835"], id_names=['PaO2', 'FiO2'], out_dir=os.path.join(out_dir, 'samples'))
        # plot other information
        self.explorer.miss_mat(out_dir)
        self.explorer.first_ards_time(out_dir)
        self.explorer.feature_count(out_dir)