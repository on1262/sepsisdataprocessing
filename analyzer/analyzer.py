import numpy as np
import tools
import os
from tools import logger as logger
from .container import DataContainer
from .feature_explore import FeatureExplorer
from .method_lstm_original import LSTMOriginalAnalyzer
from .method_lstm_balanced import LSTMBalancedAnalyzer
from .method_nearest_cls import BaselineNearestClsAnalyzer
from .method_catboost_cls import CatboostAnalyzer
from .method_random_forest import RandomForestAnalyzer
from .method_catboost_forest import CatboostForestAnalyzer
from .method_catboost_dynamic import CatboostDynamicAnalyzer
from .method_lstm_cascade import LSTMCascadeAnalyzer
from datasets import MIMICDataset


class Analyzer:
    def __init__(self, params:list, dataset:MIMICDataset) -> None:
        '''
        params: 启动脚本, 否则需要手动run_sub_analyzer, 可以是None
        dataset: 数据集
        '''
        self.container = DataContainer(dataset)
        self.analyzer_dict = {
            'LSTM_original_pf_dp': LSTMOriginalAnalyzer,
            'LSTM_original_pf':LSTMOriginalAnalyzer,
            'LSTM_original_dp':LSTMOriginalAnalyzer,
            'LSTM_original':LSTMOriginalAnalyzer,
            "LSTM_balanced":LSTMBalancedAnalyzer,
            "LSTM_cascade": LSTMCascadeAnalyzer,
            'nearest_4cls': BaselineNearestClsAnalyzer,
            'catboost_4cls':CatboostAnalyzer,
            'random_forest':RandomForestAnalyzer,
            'catboost_forest':CatboostForestAnalyzer,
            'catboost_dyn': CatboostDynamicAnalyzer,
            'feature_explore': FeatureExplorer,
        }
        if params is not None:
            for key in params:
                if key in self.analyzer_dict:
                    self.run_sub_analyzer(key)

        
    def run_sub_analyzer(self, analyzer_name):
        logger.info(f'Run Analyzer: {analyzer_name}')
        params = self.container.get_model_params(analyzer_name)
        if 'dataset_version' in params:
            self.container.dataset.load_version(params['dataset_version'])
        params['analyzer_name'] = analyzer_name
        sub_analyzer = self.analyzer_dict[analyzer_name](params, self.container)
        sub_analyzer.run()
        # utils.create_final_result()