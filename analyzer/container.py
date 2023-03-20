from datasets import AbstractDataset
import tools


class DataContainer():
    '''存放数据和一些与模型无关的内容'''
    def __init__(self, dataset:AbstractDataset):
        self.dataset = dataset
        self.data = dataset.data
        self.gbl_conf = tools.GLOBAL_CONF_LOADER['mimic_analyzer']
        self.loc_conf = tools.Config(cache_path=self.gbl_conf['paths']['conf_cache_path'], manual_path=self.gbl_conf['paths']['conf_manual_path'])
        self.target_name = dataset.target_name # only one name
        self.target_idx = dataset.target_idx
        self.n_fold = 4
        self.ards_threshold = self.gbl_conf['ards_threshold']
        # for feature importance
        self.register_values = {}