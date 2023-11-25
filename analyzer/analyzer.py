import numpy as np
import tools
import os
from tools import logger as logger
from .container import DataContainer
from .ards_catboost_dynamic import ARDSCatboostRegressionAnalyzer
from .dataset_explore.ards_explore import ArdsFeatureExplorer
from .ards_nearest_cls import ArdsNearest4ClsAnalyzer
from .cross_validation import CV_Analyzer
from .dataset_explore.raw_explore import RawFeatureExplorer
from .vent_catboost_dynamic import VentCatboostDynamicAnalyzer


class Analyzer:
    def __init__(self, params:list) -> None:
        '''
        params: startup script, otherwise you need to run_sub_analyzer manually, can be None
        '''
        self.container = DataContainer()
        self.analyzer_dict = {
            'ards_nearest_4cls': ArdsNearest4ClsAnalyzer,
            'ards_catboost_dynamic': ARDSCatboostRegressionAnalyzer,
            'ards_feature_explore': ArdsFeatureExplorer,

            'vent_catboost_dynamic': VentCatboostDynamicAnalyzer,
            'cross_validation': CV_Analyzer,

            'raw_feature_explore': RawFeatureExplorer,
            
        }
        if params is not None:
            for name in params:
                for label in self.analyzer_dict.keys():
                    if label == name:
                        self.run_sub_analyzer(name, label)
                        break

    def run_sub_analyzer(self, analyzer_name, label):
        logger.info(f'Run Analyzer: {analyzer_name}')
        params = self.container.get_analyzer_params(analyzer_name)
        params['analyzer_name'] = analyzer_name
        sub_analyzer = self.analyzer_dict[label](params, self.container)
        sub_analyzer.run()
        # utils.create_final_result()