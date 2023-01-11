import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy
import pandas as pd
from collections.abc import Iterable
import os, sys
import subprocess
import missingno as msno
from .generic import reinit_dir
from .colorful_logging import logger


def plot_loss(data, title='Title'):
    plt.plot(data)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    plt.close()

def plot_single_dist(data:np.ndarray, data_name:str):
    (mu, sigma) = scipy.stats.norm.fit(data)

    ax = sns.histplot(data=data, stat='proportion',kde=True)
    x0, x1 = ax.get_xlim()  # extract the endpoints for the x-axis
    x_pdf = np.linspace(x0, x1, 100)
    y_pdf = scipy.stats.norm.pdf(x_pdf, loc = mu, scale=sigma) # add norm information
    ax.plot(x_pdf, y_pdf, 'gray', lw=2, label='pdf')                                                     
    plt.title('distribution for ' + data_name, fontsize = 13)
    plt.legend(['data', 'Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.show()

def plot_hotspot(data:np.ndarray, fea_names:list):
    mat = np.corrcoef(x=data, rowvar=False)
    f, ax = plt.subplots(figsize=(60, 60))
    mask = np.triu(np.ones_like(mat, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(mat, mask=mask, cmap=cmap, vmax=1, center=0, annot = False, xticklabels=fea_names, yticklabels=fea_names,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title('Pearson Correlation Matrix', fontsize = 13)
    plt.show()

'''
生成X的每一列关于Y的线性回归, 用来探究单变量对目标的影响
write_dir_path: 将每个变量保存为一张图, 放在给定文件夹中
'''
def plot_reg_correlation(X:np.ndarray, fea_names:Iterable, Y:np.ndarray, target_name: str, write_dir_path=None):
    if write_dir_path is not None:
        os.makedirs(write_dir_path, exist_ok=True)
    ymin = Y.min()
    ymax = Y.max()
    corr_list = []
    for idx, _ in enumerate(fea_names):
        corr = np.corrcoef(x=X[:,idx], y=Y, rowvar=False)[0,1]
        corr_list.append(corr)
    idx_list = list(range(len(fea_names)))
    idx_list = sorted(idx_list, key= lambda idx:corr_list[idx], reverse=True)
    for idx in idx_list:
        name = fea_names[idx]
        plt.figure(figsize = (12,6))
        sns.regplot(x=X[:, idx], y=Y, scatter_kws={'alpha':0.2})
        plt.title(f'{name} vs {target_name}', fontsize = 12)
        plt.ylim(bottom=ymin, top=ymax)
        plt.legend(['$Pearson=$ {:.2f}'.format(corr_list[idx])], loc = 'best')
        if write_dir_path is None:
            plt.show()
        else:
            plt.savefig(
                os.path.join(write_dir_path, rf'{fea_names[idx]}_vs_{target_name}.png')
            )
        plt.close()

'''
生成某个特征所有样本的shap value和特征值的对应关系
'''
def plot_shap_scatter(fea_name:str, shap:np.ndarray, values:np.ndarray, x_lim=(0, -1), write_dir_path=None):
    plt.figure(figsize = (6,6))
    sns.scatterplot(x=values, y=shap)
    plt.title(f'SHAP Value scatter plot for {fea_name}', fontsize = 12)
    plt.xlabel(fea_name)
    plt.ylabel('SHAP Value')
    if x_lim[1] > x_lim[0]:
        plt.xlim(left=x_lim[0], right=min(x_lim[1], values.max()))
    if write_dir_path is None:
        plt.show()
    else:
        plt.savefig(
            os.path.join(write_dir_path, f'shap_scatter_{fea_name}.png')
        )

'''
生成X的每一列关于Y的条件分布, 用来探究单变量对目标的影响, 要求Y能转换为Bool型
write_dir_path: 将每个变量保存为一张图, 放在给定文件夹中
'''
def plot_dis_correlation(X:np.ndarray, fea_names, Y:np.ndarray, target_name, write_dir_path=None):
    Y = np.nan_to_num(Y, copy=False, nan=0) 
    reinit_dir(write_dir_path)
    Y = Y.astype(bool)
    convert_list = []
    for idx, name in enumerate(fea_names):
        try:
            x = X[:,idx].astype(float)
            x = np.nan_to_num(x, copy=False, nan=-1)
            valid = (x > -0.5)
            x_valid = x[valid]
            Y_valid = Y[valid]
            corr = np.corrcoef(x=x_valid, y=Y_valid, rowvar=False)[0,1]
            convert_list.append(corr)
        except Exception as e:
            logger.info(f'plot_dis_correlation: No correlation for {name}.')
            convert_list.append(-2)
    idx_list = list(range(len(fea_names)))
    idx_list = sorted(idx_list, key= lambda idx:abs(convert_list[idx]), reverse=True)
    for rank, idx in enumerate(idx_list):
        logger.debug(f'Plot correlation: {fea_names[idx]}')
        name = fea_names[idx]
        if convert_list[idx] > -1:
            df = pd.DataFrame(data=np.stack([X[:, idx].astype(float),Y],axis=1), columns=[name,'y'])
            df = df[df[name] > -0.5] # remove missing value
        else:
            df = pd.DataFrame(data=np.stack([X[:, idx].astype(str),Y],axis=1), columns=[name,'y'])
        sns.displot(
            data=df, x=name, hue='y', kind='hist', stat='proportion', common_norm=False, bins=20
        )
        if convert_list[idx] > -1:
            plt.annotate(f'corr={convert_list[idx]:.3f}', xy=(0.05, 0.95), xycoords='axes fraction')
        if write_dir_path is None:
            plt.show()
        else:
            plt.savefig(
                os.path.join(write_dir_path, rf'{rank}.png')
            )
        plt.close()
    if write_dir_path is not None:
        logger.info(f'dis correlation is saved in {write_dir_path}')


def plot_na(data:pd.DataFrame, mode='matrix', disp=False, save_path=None):
    if mode == 'matrix':
        msno.matrix(data)
        plt.title('feature valid matrix (miss=white)')
    elif mode == 'bar':
        p=0.5
        df = msno.nullity_filter(data, filter='bottom', p=p)
        if not df.empty:
            fea_count = len(df.columns)
            msno.bar(df=df,fontsize=10,
                figsize=(15, (25 + max(fea_count,50) - 50) * 0.5),
                sort='descending')
            plt.title(f'feature valid rate, thres={p}')
    elif mode == 'sample': # calculate row missing rate
        na_mat = data.isna().to_numpy(dtype=np.int32)
        valid_mat = 1 - np.mean(na_mat, axis=1)
        sns.histplot(x=valid_mat, bins=20, stat='proportion')
        plt.title('sample valid rate')
    else:
        assert(False)
    if save_path:
        plt.savefig(save_path)
    if disp:
        plt.show()
    
    
    plt.close()

def plot_category_dist(data:pd.DataFrame, type_dict:dict, output_dir=None):
    reinit_dir(output_dir)
    for name in type_dict.keys():
        if isinstance(type_dict[name], dict):
            sns.histplot(data=data[name], stat='proportion')
            plt.title(f'Category items distribution of {name}')
            plt.savefig(os.path.join(output_dir, f'{name}_dist.png'))
            plt.close()

def test_fea_importance(model, X_test, fea_name):
    import shap
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap_importance = shap_values.abs.mean(0).values
    sorted_idx = shap_importance.argsort()
    shap_vals = shap_importance[sorted_idx]
    sorted_names = np.asarray(fea_name)[sorted_idx]
    # plot_fea_importance(shap_vals, sorted_names, save_path)
    return shap_values.values, shap_vals, sorted_names

def plot_fea_importance(shap_vals, sorted_names, save_path=None):

    plt.figure(figsize=(12, 12))
    plt.barh(range(len(shap_vals)), shap_vals, align='center')
    plt.yticks(range(len(shap_vals)), np.array(sorted_names))
    plt.title('SHAP Importance')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()