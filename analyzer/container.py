from datasets import AbstractDataset
import tools
import os


class DataContainer():
    '''存放数据和一些与模型无关的内容'''
    def __init__(self, dataset:AbstractDataset):
        self.dataset = dataset
        self.data = dataset.data
        self._conf = tools.GLOBAL_CONF_LOADER['analyzer']['data_container'] # 这部分是global, 对外界不可见
        self.target_name = dataset.target_name # only one name
        self.target_idx = dataset.target_idx
        self.n_fold = self._conf['n_fold']
        self.ards_threshold = self._conf['ards_threshold']
        self.seed = self._conf['seed']
        # for feature importance
        self.register_values = {}

    def get_model_params(self, model_name) -> dict:
        paths = tools.GLOBAL_CONF_LOADER['analyzer'][model_name]['paths']
        params = tools.Config(cache_path=params['paths']['conf_cache_path'], manual_path=params['paths']['conf_manual_path'])
        params['paths'] = paths # 添加global config的paths到params中
        return params
    
    def clear_register(self):
        self.register_values.clear()