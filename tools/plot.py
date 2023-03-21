import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.colors import Normalize as ColorNorm
from matplotlib.patches import PathPatch
import numpy as np
import seaborn as sns
import scipy
import pandas as pd
from collections.abc import Iterable
import os, sys
import subprocess
import missingno as msno
from .generic import reinit_dir, remove_slash
from .colorful_logging import logger

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

class LossLogger:
    def __init__(self) -> None:
        self.data = None

    def add_loss(self, data:dict):
        '''增加一条独立的model训练记录'''
        for key in data.keys():
            assert(isinstance(data[key], np.ndarray))
            if len(data[key].shape) == 1:
                data[key] = data[key][None, ...]
            assert(len(data[key].shape) <= 2)
        if self.data is None:
            self.data = data
        else:
            for key in data.keys():
                if key in self.data.keys():
                    self.data[key] = np.concatenate([self.data[key], data[key]], axis=0)

    def clear(self):
        self.data = None
    
    
    '''
        提供单个model的train和valid的loss下降图
        data: dict
            'train': [n, epochs] or [epochs]
            'valid': [n, epochs] or [epochs]
            'epochs': [epochs]
        std_bar: bool 是否作标准差(对于n>1)误差区间
        title: str
        out_path: str
    '''
    def plot(self, data:dict=None, std_bar=False, log_loss=False, title='Title', out_path:str=None):
        if data is None:
            data = self.data
        assert('train' in data.keys() or 'valid' in data.keys())
        for key in data.keys():
            assert(isinstance(data[key], np.ndarray))
        if 'train' in data.keys() and len(data['train'].shape) == 1:
            data['train'] = data['train'][None, :]
        
        if 'valid' in data.keys() and len(data['valid'].shape) == 1:
            data['valid'] = data['valid'][None, :]
        n = data['train'].shape[0] if 'train' in data.keys() else data['valid'].shape[0]
        std_flag = (n > 1) and std_bar
        epoch_len = data['train'].shape[1] if 'train' in data.keys() else data['valid'].shape[1]
        if 'epochs' in data.keys():
            epochs = data['epoch'][:]
        else:
            epochs = np.linspace(start=0, stop=epoch_len-1, num=epoch_len)

        # create figure
        plt.figure(figsize = (round(min(12+epoch_len/50, 15)),6))
        if 'train' in data.keys():
            train_data = data['train'] if not log_loss else np.log10(data['train'])
            train_mean = np.mean(train_data, axis=0)
            plt.plot(epochs, train_mean, color='C0', label='train loss')
            if std_flag:
                train_std = np.std(train_data, axis=0)
                draw_band(plt.gca(), x=epochs, y=train_mean, err=train_std, facecolor=f"C0", edgecolor="none", alpha=.2)
        if 'valid' in data.keys():
            valid_data = data['valid'] if not log_loss else np.log10(data['valid'])
            valid_mean = np.mean(valid_data, axis=0)
            plt.plot(epochs, valid_mean, color="C1", label='valid loss')
            if std_flag:
                valid_std = np.std(valid_data, axis=0)
                draw_band(plt.gca(), x=epochs, y=valid_mean, err=valid_std, facecolor=f"C1", edgecolor="none",  alpha=.2)
        plt.title(title)
        plt.xlabel('Epoch')
        if log_loss:
            plt.ylabel('Log Loss')
        else:
            plt.ylabel('Loss')
        plt.legend()
        # plt.legend(['train_loss', 'valid_loss'] if 'valid' in data.keys() else ['train_loss'])
        if out_path is None:
            plt.show()
        else:
            plt.savefig(out_path)
        plt.close()

def plot_bar_with_label(data:np.ndarray, labels:list, title:str, out_path=None):
    '''打印柱状图, 按标签顺序'''
    # Validate the input
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array.")
    if not isinstance(labels, list) or not all(isinstance(l, str) for l in labels):
        raise ValueError("Input labels must be a list of strings.")

    # sort
    # ind = np.argsort(data)
    # data = data[ind]
    # labels = [labels[i] for i in ind]

    # Set up the histogram
    fig, ax = plt.subplots(figsize=(12,12)) # Set figure size
    plt.subplots_adjust(bottom=0.3)
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


def plot_single_dist(data:np.ndarray, data_name:str, save_path=None, discrete=True, adapt=False):
    '''
    从源数据直接打印直方图
    data: shape任意, 每个元素代表一个样本
    data_name: str 特征名
    discrete: 取值是否离散, True=离散取值
    adapt: 自动调整输出取值范围, 可能会忽略某些极端值
    '''
    data = data[:]
    mu, sigma = scipy.stats.norm.fit(data)
    ax = sns.histplot(data=data, stat='proportion', discrete=discrete)
    if adapt:
        ax.set_xlim(left=max(mu-3*sigma, np.min(data)), right=min(mu+3*sigma, np.max(data)))

    plt.title('Distribution of ' + data_name, fontsize = 13)
    plt.legend(['data', 'Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()


def plot_hotspot(data:np.ndarray, fea_names:list):
    '''生成相关矩阵'''
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
    plt.close()

def plot_confusion_matrix(cm:np.ndarray, labels:list, title='Confusion matrix', comment='', save_path='./out.png'):
    '''
    生成混淆矩阵
    cm: 沿axis=0是predicted label轴, 沿axis=1是true label轴, cm[x][y]代表pred=x, gt=y
    labels: list(str) 各个class的名字
    save_path: 完整路径名
    '''
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 10))
    plt.gca().grid(False)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.OrRd)
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
    for x in range(width):
        for y in range(height):
            num_color = 'black' if cm[x][y] < 1.5*cm.mean() else 'white'
            plt.annotate(str(cm[x][y]), xy=(y, x), fontsize=24, color=num_color,
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

def plot_correlation_with_quantile(
    X_pred:np.ndarray, x_name:str, Y_gt:np.ndarray, target_name: str, quantile:list, equal_lim=False, plot_dash=True, write_dir_path=None, comment:str=''
):
    '''
    生成预测值和真实值的散点图, 并用颜色表明分位数的范围, 从而表现出模型认为该预测是否可靠
    默认不生成回归线, 但是会生成一条Y=X线
    不支持生成多个变量, X只能是(quantile, data_len)的形状
    '''
    if write_dir_path is not None:
        os.makedirs(write_dir_path, exist_ok=True)
    Y_gt = Y_gt.reshape(Y_gt.shape[0], 1)
    assert(len(X_pred.shape) == 2)
    X_pred, Y_gt =  X_pred.astype(np.float32), Y_gt.astype(np.float32)
    valid = ((1 - np.isnan(X_pred[0,...])) * (1 - np.isnan(Y_gt[:,0]))).astype(bool) # 两者都是true才行
    X_pred, Y_gt = X_pred[:, valid], Y_gt[valid,:]
    
    median_idx = round((X_pred.shape[0] - 1) / 2)
    ymin, ymax = Y_gt.min(), Y_gt.max()
    xmin, xmax = X_pred[median_idx,:].min(), X_pred[median_idx,:].max()
    d_min = min(ymin, xmin)
    d_max = max(xmax, ymax)
    # generate dataframe
    data_arr = np.ndarray((2+median_idx, X_pred.shape[1]), dtype=float)
    data_arr[0, :] = Y_gt[:,0]
    data_arr[1,:] = X_pred[median_idx, :]
    columns = ['gt','pred']
    for idx in range(1, median_idx+1): # median=2, 1,2->(2-1, 2+1), (2-2,2+2)
        data_arr[idx+1,:] = X_pred[median_idx+idx,:] - X_pred[median_idx-idx,:]
        columns.append(f'alpha={quantile[median_idx-idx]:.2f}-{quantile[median_idx+idx]:.2f}')
    df = pd.DataFrame(data=data_arr.T, columns=columns, index=None)
    for idx in range(2, len(columns)):
        logger.debug(f'Plot correlation with quantile: {x_name}, alpha=[{columns[idx]}]]')
        plt.figure(figsize = (14,12)) # W,H
        norm = HueColorNormlize(df[columns[idx]].to_numpy())
        c_map = sns.color_palette("coolwarm", as_cmap=True)
        sns.scatterplot(data=df, x="pred", y="gt", hue_norm=norm,
            hue=columns[idx], palette=c_map,
            alpha=0.5, linewidth=0, size=1)
        sm = plt.cm.ScalarMappable(cmap=c_map, norm=norm)
        sm.set_array([]) # magic
        # sm.set_clim(vmin=norm.vmin, vmax=norm.vmax)
        # Remove the legend and add a colorbar
        plt.gca().get_legend().remove()
        plt.gca().figure.colorbar(sm, ticks=norm.get_ticks())
        # plot line y=x
        if plot_dash:
            plt.plot(np.asarray([d_min, d_max]),np.asarray([d_min, d_max]), 
                linestyle='dashed', color='C7', label='Y=X')
        plt.title(f'{x_name} vs {target_name} quantile={columns[idx]} cmt=[{comment}]', fontsize = 12)
        # if restrict_area and Y.shape[0] > 20:
        #     # 去除20个极值, 使得显示效果更好
        #     Y_sorted = np.sort(Y[x_valid[:, idx], 0], axis=0)
        #     X_sorted = np.sort(X[x_valid[:, idx], idx], axis=0)
        #     Y_span = Y_sorted[-10] - Y_sorted[10]
        #     X_span = X_sorted[-10] - X_sorted[10]
        #     plt.ylim(bottom=Y_sorted[10]-Y_span*0.05, top=Y_sorted[-10]+Y_span*0.05)
        #     plt.xlim(left=X_sorted[10]-X_span*0.05, right=X_sorted[-10]+X_span*0.05)
        # else:
        if equal_lim: # 强制x和y的范围一致
            plt.ylim(bottom=d_min, top=d_max)
            plt.xlim(left=d_min, right=d_max)
        else:
            plt.ylim(bottom=ymin, top=ymax)
            plt.xlim(left=xmin, right=xmax)
        plt.xlabel(x_name)
        plt.ylabel(target_name)
        if write_dir_path is None:
            plt.show()
        else:
            plt.savefig(
                os.path.join(write_dir_path, remove_slash(rf'{x_name}vs{target_name}@{columns[idx]}{comment}.png'))
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

def draw_band(ax, x, y, err, **kwargs):
    '''绘制标准差条块'''
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)
    # Calculate normals via centered finite differences (except the first point
    # which uses a forward difference and the last point which uses a backward
    # difference).
    dx = np.concatenate([[x[1] - x[0]], x[2:] - x[:-2], [x[-1] - x[-2]]])
    dy = np.concatenate([[y[1] - y[0]], y[2:] - y[:-2], [y[-1] - y[-2]]])
    l = np.hypot(dx, dy)
    nx = dy / l
    ny = -dx / l
    # end points of errors
    xp = x + nx * err
    yp = y + ny * err
    xn = x - nx * err
    yn = y - ny * err
    vertices = np.block([[xp, xn[::-1]],
                         [yp, yn[::-1]]]).T
    codes = np.full(len(vertices), Path.LINETO)
    codes[0] = Path.MOVETO
    path = Path(vertices, codes)
    ax.add_patch(PathPatch(path, **kwargs))
