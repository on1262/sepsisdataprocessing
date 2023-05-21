import numpy as np
import tools
import os
from tools import logger as logger
from .container import DataContainer
from .feature_explore import FeatureExplorer
from .method_lstm_original import LSTMOriginalAnalyzer
from .method_nearest_cls import BaselineNearestClsAnalyzer
from .method_catboost_cls import CatboostAnalyzer
from .method_random_forest import RandomForestAnalyzer
from .method_catboost_forest import CatboostForestAnalyzer
from .method_catboost_dynamic import CatboostDynamicAnalyzer
from .method_lstm_cascade import LSTMCascadeAnalyzer
from .method_lstm_cascade_extend import LSTMCascadeExtendAnalyzer
from .method_logistic_regression import LogisticRegAnalyzer
from datasets import MIMICDataset


class Analyzer:
    def __init__(self, params:list, dataset:MIMICDataset) -> None:
        '''
        params: 启动脚本, 否则需要手动run_sub_analyzer, 可以是None
        dataset: 数据集
        '''
        self.container = DataContainer(dataset)
        self.analyzer_dict = {
            'LSTM_original':LSTMOriginalAnalyzer,
            "LSTM_cascade": LSTMCascadeAnalyzer,
            'LSTM_extend_cascade': LSTMCascadeExtendAnalyzer,
            'nearest_4cls': BaselineNearestClsAnalyzer,
            'catboost_4cls':CatboostAnalyzer,
            'random_forest':RandomForestAnalyzer,
            'catboost_forest':CatboostForestAnalyzer,
            'catboost_dyn': CatboostDynamicAnalyzer,
            'feature_explore': FeatureExplorer,
            'logistic_reg': LogisticRegAnalyzer,
        }
        if params is not None:
            for name in params:
                for label in self.analyzer_dict.keys():
                    if label in name:
                        self.run_sub_analyzer(name, label)
                        break

        
    def run_sub_analyzer(self, analyzer_name, label):
        logger.info(f'Run Analyzer: {analyzer_name}')
        params = self.container.get_model_params(analyzer_name)
        if 'dataset_version' in params:
            self.container.dataset.load_version(params['dataset_version'])
        params['analyzer_name'] = analyzer_name
        sub_analyzer = self.analyzer_dict[label](params, self.container)
        sub_analyzer.run()
        # utils.create_final_result()