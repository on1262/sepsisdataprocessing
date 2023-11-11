import numpy as np
import tools
import os
from tools import logger as logger
from .container import DataContainer
from .method_catboost_dynamic import CatboostDynamicAnalyzer
from .feature_explore import FeatureExplorer
from .method_nearest_cls import BaselineNearest4ClsAnalyzer
from .cross_validation import CV_Analyzer
from datasets import MIMICIVDataset


class Analyzer:
    def __init__(self, params:list, dataset:MIMICIVDataset) -> None:
        '''
        params: 启动脚本, 否则需要手动run_sub_analyzer, 可以是None
        dataset: 数据集
        '''
        self.container = DataContainer(dataset)
        self.analyzer_dict = {
            'nearest_4cls': BaselineNearest4ClsAnalyzer,
            'catboost_dynamic': CatboostDynamicAnalyzer,
            'feature_explore': FeatureExplorer,
            'cross_validation': CV_Analyzer,
        }
        if params is not None:
            for name in params:
                for label in self.analyzer_dict.keys():
                    if label in name:
                        self.run_sub_analyzer(name, label)
                        break

    def run_sub_analyzer(self, analyzer_name, label):
        logger.info(f'Run Analyzer: {analyzer_name}')
        params = self.container.get_analyzer_params(analyzer_name)
        if 'dataset_version' in params:
            self.container.dataset.load_version(params['dataset_version'])
        params['analyzer_name'] = analyzer_name
        sub_analyzer = self.analyzer_dict[label](params, self.container)
        sub_analyzer.run()
        # utils.create_final_result()