import shap
import matplotlib.pyplot as plt
import numpy as np
import os


class TreeFeatureImportance():
    def __init__(self, fea_names) -> None:
        self.fea_names = fea_names
        # register
        self.records = []

    def add_record(self, model, valid_X):
        explainer = shap.Explainer(model)
        shap_values = explainer(valid_X)
        self.records.append((shap_values.base_values, shap_values.data, shap_values.values)) # (sample, n_fea)

    def update_record(self):
        if isinstance(self.records, list):
            base_values = np.concatenate([record[0] for record in self.records], axis=0)
            data = np.concatenate([record[1] for record in self.records], axis=0)
            shap_values = np.concatenate([record[2] for record in self.records], axis=0)
            self.records = shap.Explanation(base_values=base_values, data=data, values=shap_values, feature_names=self.fea_names)

    def plot_beeswarm(self, plot_path):
        self.update_record()
        plt.subplots_adjust(left=0.3)
        shap.plots.beeswarm(self.records, order=self.records.abs.mean(0), show=False, plot_size=(14,10))
        plt.savefig(plot_path)
        plt.close()

    def plot_single_importance(self, out_dir, select=None):
        '''
        输出每个特征的取值和重要性关系
        select: 可以是list/int/None
            int: 选择前k个特征输出
            None: 输出所有特征
        '''
        self.update_record()
        imp = self.records.abs.mean(0).values
        order = sorted(list(range(len(self.fea_names))), key=lambda x:imp[x], reverse=True)
        if isinstance(select, int):
            order = order[:min(select, len(order))]
        names = [self.fea_names[idx] for idx in order]
        for idx, name in zip(order, names):
            plt.subplots_adjust(left=0.3) 
            shap.plots.scatter(self.records[:,name], )
            plt.savefig(os.path.join(out_dir, f'{name}.png'))
            plt.close()


class IntegratedFeatureImportance():
    '''基于intergrated-gradients对深度学习网络计算重要性'''
    def __init__(self, fea_names) -> None:
        self.fea_names = fea_names
        # register
        self.records = []

    def add_record(self, model, valid_X):
        explainer = shap.Explainer(model)
        shap_values = explainer(valid_X)
        self.records.append((shap_values.base_values, shap_values.data, shap_values.values)) # (sample, n_fea)

    def update_record(self):
        if isinstance(self.records, list):
            base_values = np.concatenate([record[0] for record in self.records], axis=0)
            data = np.concatenate([record[1] for record in self.records], axis=0)
            shap_values = np.concatenate([record[2] for record in self.records], axis=0)
            self.records = shap.Explanation(base_values=base_values, data=data, values=shap_values, feature_names=self.fea_names)

    def plot_beeswarm(self, plot_path):
        self.update_record()
        plt.subplots_adjust(left=0.3)
        shap.plots.beeswarm(self.records, order=self.records.abs.mean(0), show=False, plot_size=(14,10))
        plt.savefig(plot_path)
        plt.close()

    def plot_single_importance(self, out_dir, select=None):
        '''
        输出每个特征的取值和重要性关系
        select: 可以是list/int/None
            int: 选择前k个特征输出
            None: 输出所有特征
        '''
        self.update_record()
        imp = self.records.abs.mean(0).values
        order = sorted(list(range(len(self.fea_names))), key=lambda x:imp[x], reverse=True)
        if isinstance(select, int):
            order = order[:min(select, len(order))]
        names = [self.fea_names[idx] for idx in order]
        for idx, name in zip(order, names):
            plt.subplots_adjust(left=0.3)
            shap.plots.scatter(self.records[:,name], )
            plt.savefig(os.path.join(out_dir, f'{name}.png'))
            plt.close()

    
        
        
        
        
        
        
        
        
        
        