import matplotlib.pyplot as plt
from matplotlib.colors import Normalize as ColorNorm
import numpy as np
import seaborn as sns
import scipy
import pandas as pd
from collections.abc import Iterable
import os, sys
import random
from tqdm import tqdm
import subprocess
import missingno as msno
from .generic import reinit_dir, remove_slash
from matplotlib.colors import to_rgb
from .logging import logger

'''
    用于分位数回归的作图, 通过线性插值得到待测点所对应的分位数, 使得数据的分布不会改变出图的色彩多样性
'''
class HueColorNormlize(ColorNorm):
    def __init__(self, data:np.ndarray) -> None:
        super().__init__(vmin=data.min(), vmax=data.max(), clip=False)
        n_points = 21
        points = [data.min()]
        data_sorted = np.sort(data, axis=None)
        for idx in range(n_points-1):
            points.append(data_sorted[round(((idx+1)/(n_points-1))*(data.shape[0]-1))])
        self.x = np.asarray(points)
        self.y = np.asarray(list(range(len(points)))) / (len(points)-1)

    def get_ticks(self):
        return self.x

    def __call__(self, value, clip: bool = None):
        return np.interp(value, self.x, self.y, left=0, right=1)

    def inverse(self, value):
        return np.interp(value, self.y, self.x, left=self.vmin, right=self.vmax)

def simple_plot(data, title='Title', out_path=None):
    plt.figure(figsize = (6,12))
    for idx in range(data.shape[0]):
        plt.plot(data[idx, :])
    plt.title(title)
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)
    plt.close()


def plot_bar_with_label(data:np.ndarray, labels:list, title:str, sort=True, out_path=None):
    # Validate the input
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array.")
    if not isinstance(labels, list) or not all(isinstance(l, str) for l in labels):
        raise ValueError("Input labels must be a list of strings.")

    # sort
    if sort:
        ind = np.argsort(data)
        data = data[ind]
        labels = [labels[i] for i in ind]

    # Set up the histogram
    fig, ax = plt.subplots(figsize=(12,12)) # Set figure size
    plt.subplots_adjust(bottom=0.4, left=0.2, right=0.8)
    ind = np.arange(len(data))
    width = 0.8
    if len(labels) < 20:
        fontsize = 20
    else:
        fontsize = 10
    ax.bar(ind, data, width, color='SkyBlue')

    # Set up the X axis
    ax.set_xticks(range(0, len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=fontsize)

    ax.set_title(title)
    
    if out_path is None:
        # Show the histogram
        plt.show()
    else:
        plt.savefig(out_path)
    plt.close()

def plot_stack_proportion(data:dict[str, tuple], out_path=None):
    plt.figure(figsize=(25, 10))
    height = 0.5
    names = list(data.keys())
    style = [to_rgb(f'C{idx}') for idx in range(10)]
    plt.barh(names, [0 for _ in names], height=height)
    idx = 0
    for k_idx, (key, (x, label)) in enumerate(data.items()):
        x_sum = 0
        for idx in range(len(x)):
            color = np.asarray(style[k_idx % 10])
            color = np.clip(color + 0.2 * (idx % 2), 0, 1.0)
            plt.barh([key], x[idx], left=x_sum, color=tuple(color), height=height)
            label_wid = len(label[idx])*0.006
            if x[idx] > label_wid:
                plt.annotate(label[idx], (x_sum + x[idx]*0.5 - label_wid*0.5, k_idx), fontsize=10)
            x_sum += x[idx]
    
    plt.xlim(left=0, right=1)
    plt.subplots_adjust(left=0.1, right=0.9)
    plt.savefig(out_path)


def plot_density_matrix(data:np.ndarray, title:str, xlabel:str, ylabel:str, aspect='equal', save_path=None):
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(data, cmap='jet', aspect=aspect) # auto for square picture, equal for original aspect ratio
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.close()


def plot_single_dist(data:np.ndarray, data_name:str, save_path=None, discrete=True, adapt=False, label=False, **kwargs):
    '''
    从源数据直接打印直方图
    data: shape任意, 每个元素代表一个样本
    data_name: str 特征名
    discrete: 取值是否离散, True=离散取值
    adapt: 自动调整输出取值范围, 可能会忽略某些极端值
    '''
    data = data[:]
    if data.shape[0] > 2000:
        data = np.asarray(random.choices(data, k=2000))
    assert(not np.any(np.isnan(data)))
    mu, sigma = scipy.stats.norm.fit(data)
    if adapt and sigma > 0.01:
        data = data[np.logical_and(data >= mu-3*sigma, data <= mu+3*sigma)]
    
    plt.figure(figsize=(8,8))
    ax = sns.histplot(data=data, stat='proportion', discrete=discrete, **kwargs)
    if adapt:
        if discrete:
            ax.set_xlim(left=max(mu-3*sigma, np.min(data))-0.5, right=min(mu+3*sigma, np.max(data))+0.5)
        else:
            ax.set_xlim(left=max(mu-3*sigma, np.min(data)), right=min(mu+3*sigma, np.max(data)))
    if label:
        ax.bar_label(ax.containers[1], fontsize=10, fmt=lambda x:f'{x:.3f}')
    plt.title('Distribution of ' + data_name, fontsize = 13)
    plt.legend(['data', 'Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()


def plot_correlation_matrix(data:np.ndarray, fea_names:list, invalid_flag=-1, corr_thres=-1, save_path=None):
    '''
    生成相关矩阵
    data: (sample, n_fea)
    fea_names: (n_fea)
    percent_corr: 只打印相关性大于阈值的特征，我们对多重共线性特征更感兴趣
    '''
    assert(len(fea_names) == data.shape[1])
    fig_size = int(8+len(fea_names)*0.2)
    mat = np.zeros((data.shape[1], data.shape[1]))
    for i in tqdm(range(data.shape[1]), 'plot correlation'):
        for j in range(data.shape[1]):
            if i < j:
                idx_i = np.logical_and(data[:, i] != invalid_flag, np.logical_not(np.isnan(data[:, i])))
                idx_j = np.logical_and(data[:, j] != invalid_flag, np.logical_not(np.isnan(data[:, j])))
                avail_idx = np.logical_and(idx_i, idx_j) # availble subject index
                if np.sum(avail_idx) < 100:
                    continue
                vi = data[avail_idx, i]
                vj = data[avail_idx, j]
                coef = np.corrcoef(vi, vj) # (2,2)
                if not np.isnan(coef[0,1]):
                    mat[j, i] = coef[0,1]
    
    if corr_thres > 0:
        valid_idx = np.logical_or((np.abs(mat).max(axis=0) >= corr_thres), (np.abs(mat).max(axis=1) >= corr_thres))
        mat = mat[valid_idx, :][:, valid_idx]
        fea_names = [name for idx, name in enumerate(fea_names) if valid_idx[idx]]
    f, ax = plt.subplots(figsize=(fig_size, fig_size))
    mask = np.triu(np.ones_like(mat, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(mat, mask=mask, cmap=cmap, vmax=1, center=0, annot=False, xticklabels=fea_names, yticklabels=fea_names,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title('Pearson Correlation Matrix', fontsize = 13)
    plt.subplots_adjust(left=0.3)
    plt.savefig(save_path)
    plt.close()
    return mat

def plot_confusion_matrix(cm:np.ndarray, labels:list, title='Confusion matrix', comment='', save_path='./out.png'):
    '''
    生成混淆矩阵
    cm: cm[x][y]代表pred=x, gt=y
    labels: list(str) 各个class的名字
    save_path: 完整路径名
    '''
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 10))
    plt.gca().grid(False)
    plt.imshow(cm.T, interpolation='nearest', cmap=plt.cm.OrRd)
    if comment != '':
        title = title + f'[{comment}]'
    plt.title(title, size=18)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, size=15)
    plt.yticks(tick_marks, labels, size=15)
    plt.ylabel('True label', size=18)
    plt.xlabel('Predicted label', size=18)
    width, height = cm.shape
    out_type = 'int' if np.max(cm) > 1+1e-3 else 'float'
    for x in range(width):
        for y in range(height):
            num_color = 'black' if cm[y][x] < 1.5*cm.mean() else 'white'
            cm_str = str(cm[y][x]) if out_type == 'int' else f'{cm[y][x]:.2f}'
            plt.annotate(cm_str, xy=(y, x), fontsize=24, color=num_color,
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.savefig(save_path)
    plt.close()

def plot_reg_correlation(X:np.ndarray, fea_names:Iterable, Y:np.ndarray, 
    target_name: str, adapt=False, write_dir_path=None, plot_dash=True, comment:str=''):
    '''
    生成X的每一列关于Y的线性回归, 用来探究单变量对目标的影响
    write_dir_path: 将每个变量保存为一张图, 放在给定文件夹中
    X: (sample, n_fea)
    fea_names: list(str)
    Y: (sample,)
    target_name:str
    plot_dash: 是否画出Y=X的虚线
    
    '''
    if write_dir_path is not None:
        os.makedirs(write_dir_path, exist_ok=True)
    Y = Y.reshape(Y.shape[0], 1)
    ymin, ymax = np.inf, -np.inf
    X,Y =  X.astype(np.float32), Y.astype(np.float32)
    x_valid = ((1 - np.isnan(X)) * (1 - np.isnan(Y))).astype(bool) # 两者都是true才行
    corr_list = []
    for idx, _ in enumerate(fea_names):
        x,y = X[x_valid[:, idx],idx], Y[x_valid[:, idx]]
        ymin, ymax = min(ymin, y.min()), max(ymax, y.max())
        corr = np.corrcoef(x=x, y=y, rowvar=False)[1][0] # 相关矩阵, 2*2
        corr_list.append(corr)
    idx_list = list(range(len(fea_names)))
    idx_list = sorted(idx_list, key = lambda idx:abs(corr_list[idx]), reverse=True) # 按相关系数绝对值排序
    for rank, idx in enumerate(idx_list):
        name = fea_names[idx]
        logger.debug(f'Plot correlation: {name} cmt=[{comment}]')
        plt.figure(figsize = (12,12))
        sns.regplot(x=X[x_valid[:, idx], idx], y=Y[x_valid[:, idx]], scatter_kws={'alpha':0.2})
        # plot line y=x
        d_min, d_max = ymin, ymax
        if plot_dash:
            plt.plot(np.asarray([d_min, d_max]),np.asarray([d_min, d_max]), 
                linestyle='dashed', color='C7', label='Y=X')
        plt.title(f'{name} vs {target_name} cmt=[{comment}]', fontsize = 12)
        if adapt and Y.shape[0] > 20:
            # 去除20个极值, 使得显示效果更好
            Y_sorted = np.sort(Y[x_valid[:, idx], 0], axis=0)
            X_sorted = np.sort(X[x_valid[:, idx], idx], axis=0)
            Y_span = Y_sorted[-10] - Y_sorted[10]
            X_span = X_sorted[-10] - X_sorted[10]
            plt.ylim(bottom=Y_sorted[10]-Y_span*0.05, top=Y_sorted[-10]+Y_span*0.05)
            plt.xlim(left=X_sorted[10]-X_span*0.05, right=X_sorted[-10]+X_span*0.05)
        else:
            plt.ylim(bottom=ymin, top=ymax)
        plt.xlabel(name)
        plt.ylabel(target_name)
        plt.legend(['$Pearson=$ {:.2f}'.format(corr_list[idx])], loc = 'best')
        if write_dir_path is None:
            plt.show()
        else:
            plt.savefig(
                os.path.join(write_dir_path, remove_slash(rf'{rank}@{fea_names[idx]}_vs_{target_name}{comment}.png'))
            )
        plt.close()


def plot_shap_scatter(fea_name:str, shap:np.ndarray, values:np.ndarray, x_lim=(0, -1), write_dir_path=None):
    '''
    生成某个特征所有样本的shap value和特征值的对应关系
    '''
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
            os.path.join(write_dir_path, f'shap_scatter_{remove_slash(fea_name)}.png')
        )
    plt.close()


def plot_dis_correlation(X:np.ndarray, fea_names, Y:np.ndarray, target_name, write_dir_path=None):
    '''
    生成X的每一列关于Y的条件分布, 用来探究单变量对目标的影响, 要求Y能转换为Bool型
    write_dir_path: 将每个变量保存为一张图, 放在给定文件夹中
    '''
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

def plot_model_comparison(csv_path, title, out_path):
    '''输出模型对比的散点图'''
    df = pd.read_csv(csv_path, encoding='utf-8')
    plt.figure(figsize=(10,10))
    # columns=[model_name, hyper_params, metricA, metricB]
    sns.scatterplot(data=df, x="4cls_accuracy", y="robust_AUC",
                hue="model",
                palette="ch:r=-.2,d=.3_r", linewidth=0)
    plt.title(title)
    plt.savefig(out_path)