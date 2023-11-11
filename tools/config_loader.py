import yaml
import os


'''
ConfigLoader: load and process global configs from a single yaml file
    work directory will be appended if 'work_dir' in the keys of dict
'''
class ConfigLoader():
    def __init__(self, glob_conf_path:str) -> None:
        self.glob_conf_path = glob_conf_path
        with open(self.glob_conf_path, 'r',encoding='utf-8') as fp:
            self.glob_conf = yaml.load(fp, Loader=yaml.SafeLoader)
        # append work_dir to every values under 'paths'
        self.glob_conf['paths'] = self.process_path(self.glob_conf['work_dir'], self.glob_conf['paths'])
        
    def process_path(self, root, conf_dict):
        for key in conf_dict:
            if isinstance(conf_dict[key], str):
                conf_dict[key] = os.path.join(root, conf_dict[key])
            elif isinstance(conf_dict[key], dict):
                conf_dict[key] = self.process_path(root, conf_dict[key])
        return conf_dict

    def __getitem__(self, __key: str):
        return self.glob_conf[__key]
    


GLOBAL_CONF_LOADER = ConfigLoader('./configs/global_config.yaml')
        
        