from datasets import MIMICIV_ARDS_Dataset
import tools
import os


class DataContainer():
    '''存放数据和一些与模型无关的内容'''
    def __init__(self):
        self._conf = tools.GLOBAL_CONF_LOADER['analyzer']['data_container'] # 这部分是global, 对外界不可见
        self.n_fold = self._conf['n_fold']
        self.seed = self._conf['seed']
        # for feature importance
        self.register_values = {}

    def get_analyzer_params(self, model_name) -> dict:
        '''根据数据集和模型名不同, 获取所需的模型参数'''
        paths = tools.GLOBAL_CONF_LOADER['paths']
        analyzer_params = tools.Config('configs/analyzers.yml')[model_name]
        analyzer_params['paths'] = paths # 添加global config的paths到params中
        return analyzer_params
    
    def clear_register(self):
        self.register_values.clear()