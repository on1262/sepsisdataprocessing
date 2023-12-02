import shap
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import numpy as np
import random
from captum.attr import DeepLift
import seaborn as sns
import sys


class TreeFeatureImportance():
    def __init__(self, map_func, fea_names, missvalue=None, n_approx=-1) -> None:
        self.map_func = map_func # map shape value of each class into a total shap value
        self.fea_names = fea_names
        self.n_approx = n_approx
        self.missvalue = missvalue
        # register
        self.records = []

    def add_record(self, model, valid_X:np.ndarray):
        explainer = shap.Explainer(model)
        n_in = min(valid_X.shape[0], self.n_approx) if self.n_approx > 0 else valid_X.shape[0]
        permutation = np.random.permutation(valid_X.shape[0])[:n_in]
        shap_values = explainer(valid_X[permutation])
        if self.missvalue is not None:
            data = shap_values.data # (N, n_fea)
            values = shap_values.values # (N, n_fea, n_cls)
            values[data == self.missvalue, :] = 0
            self.records.append((shap_values.base_values, data, values)) # (sample, n_fea)
        else:
            self.records.append((shap_values.base_values, shap_values.data, shap_values.values)) # (sample, n_fea)

    def _update_record(self):
        if isinstance(self.records, list):
            base_values = np.concatenate([record[0] for record in self.records], axis=0)
            if len(base_values.shape) == 2: # prediction bias for each category
                base_values = np.mean(base_values, axis=-1)
            data = np.concatenate([record[1] for record in self.records], axis=0)
            shap_values = np.concatenate([record[2] for record in self.records], axis=0)
            shap_values = self.map_func(shap_values)
            self.records = shap.Explanation(base_values=base_values, data=data, values=shap_values, feature_names=self.fea_names)

    def plot_beeswarm(self, max_disp=20, plot_path=None):
        self._update_record()
        plt.subplots_adjust(left=0.3)
        shap.plots.beeswarm(self.records, order=self.records.abs.mean(0), max_display=max_disp, show=False, plot_size=(14,14))
        plt.savefig(plot_path)
        plt.close()

    def plot_single_importance(self, out_dir, select=None):
        '''
        输出每个特征的取值和重要性关系
        select: 可以是list/int/None
            int: 选择前k个特征输出
            None: 输出所有特征
        '''
        self._update_record()
        imp = self.records.abs.mean(0).values
        order = sorted(list(range(len(self.fea_names))), key=lambda x:imp[x], reverse=True)
        if isinstance(select, int):
            order = order[:min(select, len(order))]
        names = [self.fea_names[idx] for idx in order]
        for idx, name in zip(order, names):
            data = self.records[:,name]
            if len(data) >= 20:
                data_sorted = np.sort(data.data)[5:-5]
                xmin, xmax = data_sorted[0], data_sorted[-1]
            plt.subplots_adjust(left=0.3) 
            shap.plots.scatter(self.records[:,name])
            plt.xlabel(name)
            plt.xlim((xmin, xmax))
            plt.savefig(os.path.join(out_dir, f'{name}.png'))
            plt.close()

    def get_importance_array(self, feature_ids, fp=sys.stdout):
        '''返回特征重要性的排序'''
        self._update_record()
        imp = self.records.abs.mean(0).values
        order = sorted(list(range(len(self.fea_names))), key=lambda x:imp[x], reverse=True)
        ids = [feature_ids[idx] for idx in order]
        names = [self.fea_names[idx] for idx in order]
        result = ['\nFeature Importance\n']
        for id, name in zip(ids, names):
            result.append(f'\"{id}\", {name}\n')
        fp.writelines(result)


class DeepFeatureImportance():
    '''基于intergrated-gradients对深度学习网络计算重要性'''
    def __init__(self, device, fea_names) -> None:
        self.fea_names = fea_names
        self.device = torch.device(device)
        # register
        self.records = []

    def add_record(self, model:torch.nn.Module, valid_X:np.ndarray, threshold:int):
        '''
        要求forward_func输入为(batch, seq_len, n_fea)
        threshold: 时序上截取的最终时刻
        '''
        max_k = min(500, valid_X.shape[0]//2)
        valid_X = torch.as_tensor(valid_X, dtype=torch.float32).to(self.device)
        model = model.eval().to(self.device)
        background = torch.mean(valid_X[:max_k,...], dim=0)[None, ...]
        valid = valid_X[max_k:,...]
        explainer = DeepLift(model=model)
        shap_values = explainer.attribute(valid, background)
        shap_values = shap_values # (batch, n_fea, seq_len)
        valid = valid.detach().clone().cpu().numpy()[:, :, :threshold]
        shap_values = shap_values.detach().clone().cpu().numpy()[:, :, :threshold]
        self.records.append((valid, shap_values)) # (batch, n_fea, threshold)
        

    def update_record(self):
        if isinstance(self.records, list):
            data = np.concatenate([record[0] for record in self.records], axis=0)
            shap_values = np.concatenate([record[1] for record in self.records], axis=0)
            self.records = {}
            self.records['exp'] = shap.Explanation(base_values=None, data=np.mean(data, axis=-1), values=np.mean(shap_values, axis=-1), feature_names=self.fea_names)
            self.records['data'] = data
            self.records['shap_values'] = shap_values


    def plot_beeswarm(self, plot_path):
        self.update_record()
        plt.subplots_adjust(left=0.3)
        shap.plots.beeswarm(self.records['exp'], order=self.records['exp'].abs.mean(0), max_display=20, show=False, plot_size=(14,14))
        plt.savefig(plot_path)
        plt.close()

    def plot_hotspot(self, plot_path):
        self.update_record()
        shap_values = np.mean(np.abs(self.records['shap_values']), axis=0) # (n_fea, threshold)
        shap_values = np.log10(shap_values-np.min(shap_values)+1e-6)
        # reference_index = list(self.fea_names).index('age')
        # reference_seq = shap_values[reference_index, :].copy()
        # for idx in range(shap_values.shape[0]):
        #     shap_values[idx, :] -= reference_seq
        imps = np.mean(shap_values, axis=-1)
        sorted_idx = sorted(list(range(len(imps))), key=lambda x:imps[x])
        sorted_names = [self.fea_names[idx] for idx in sorted_idx]
        sorted_values = np.asarray([shap_values[idx, :] for idx in sorted_idx])
        time_ticks = [n for n in range(shap_values.shape[-1])]
        f, ax = plt.subplots(figsize=(15, 15))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(sorted_values, cmap=cmap, annot=False, yticklabels=sorted_names, xticklabels=time_ticks,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.title('Feature importance DeepSHAP', fontsize = 13)
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

    
        
        
        
        
        
        
        
        
        
        