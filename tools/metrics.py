import matplotlib.pyplot as plt
import numpy as np
import os, sys
from scipy.interpolate import interp1d
from sklearn.metrics import auc as sk_auc
from .logging import logger
from .plot import plot_confusion_matrix
from tools import GLOBAL_CONF_LOADER
import pandas as pd

class DichotomyMetric:
    '''二分类评价指标'''
    def __init__(self) -> None:
        self.data = []
        self.n_thres = 30
        self.annotate_thres= 10 # ROC曲线上显示多少个阈值
        self.is_calculated = False
        self.thres_eps = 0.01 # 当阈值间隔小于eps时, n_thres将被自动调整
        self.eps = 0.001
        # calculation resources
        self.thres = []
        self.points = []
        self.combined_points = {}
        self.combined_data = None
        self.tpr_std = []
        
    def clear(self):
        self.__init__()
    
    def add_prediction(self, pred:np.ndarray, gt:np.ndarray):
        '''
            pred: (batch,)  gt: (batch,)
            gt取值是0/1, pred取值是一个概率
        '''
        assert(len(pred.shape)==1 and len(gt.shape)==1 and pred.shape[0]==gt.shape[0])
        self.data.append(np.stack((pred.copy(), gt.copy()), axis=1)) # -> (batch, 2)
        self.is_calculated = False # 更新状态

    def _cal_single_point(self, Y_gt, Y_pred, thres):
        tp = np.sum(Y_pred[Y_gt > 0.5] > thres)
        fp = np.sum(Y_pred[Y_gt < 0.5] > thres)
        fn = np.sum(Y_pred[Y_gt > 0.5] <= thres)
        tn = np.sum(Y_pred[Y_gt < 0.5] <= thres)
        return {
            'tp': tp, 'fp':fp, 
            'tn': tn, 'fn':fn,
            'tpr':tp/(tp+fn), 'fpr':fp/(fp+tn), 
            'acc':(tp+tn)/(tp+fp+tn+fn),
            'recall':tp/(tp+fn), 
            'spec':tn/(tn+fp), 
            'prec': tp/(tp+fp) if tp + fp > 0 else 1,
            'f1': 2*(tp/(tp+fp))*(tp/(tp+fn)) / (tp/(tp+fp) + tp/(tp+fn)) if tp + fp > 0 else 2*tp/(2*tp+fn),
            'thres':thres
        }
    
    def _cal_thres(self):
        # 合并+排序
        assert(self.n_thres >= 2)
        data = np.concatenate(self.data, axis=0)
        self.combined_data = data
        sorted_pred = np.sort(data[:,0], axis=0) # ascend
        p_len = sorted_pred.shape[0]
        thres = [data.min()]
        for n in range(1, self.n_thres, 1): # 1,2,3,...,n_thres-1
            idx = round((p_len-1)*n/(self.n_thres-1))
            next_thres = sorted_pred[idx] # determined by percentage of prediction
            if np.abs(next_thres-thres[-1]) > self.thres_eps:
                thres.append(next_thres)
        # add threshold for calculate recall-precision
        thres = np.asarray(sorted(np.unique(thres + [0.01, 0.02, 0.04, 0.08, 0.16, 0.99, 0.98, 0.96, 0.92, 0.84])))
        # thres = thres[np.logical_and(thres >= self.eps, thres <= 1.0 - self.eps)] # prec = nan when thres=1
        self.n_thres = len(thres)
        self.thres = thres
    
    def process_data(self):
        self.is_calculated = True
        self._cal_thres()
        # 分别计算每条曲线
        for idx in range(len(self.data)):
            first = True
            for thres in self.thres:
                point = self._cal_single_point(self.data[idx][:,1], self.data[idx][:,0], thres=thres)
                if first == True:
                    self.points.append({key:[val] for key, val in point.items()})
                    first = False
                else:
                    for key in point.keys():
                        self.points[-1][key].append(point[key])
        
        if len(self.data) > 1:
            # 计算平均曲线
            first = True
            for thres in self.thres:
                point = self._cal_single_point(self.combined_data[:,1], self.combined_data[:,0], thres=thres)
                if first == True:
                    self.combined_points = {key:[val] for key, val in point.items()}
                    first = False
                else:
                    for key in point.keys():
                        self.combined_points[key].append(point[key])

            self.tpr_std = []
            # 计算fpr对应的tpr标准差
            for idx in range(len(self.combined_points['fpr'])):
                fpr = self.combined_points['fpr'][idx]
                tprs = []
                for p_idx in range(len(self.points)):
                    tprs.append(self.search_point('fpr', fpr, p_idx, {'tpr'})['tpr'])
                self.tpr_std.append(np.asarray(tprs).std())
        else:
            self.combined_points = self.points[0]
    
    
    '''
    线性插值得到曲线上的某个点， 例如，查询tpr=0.8时的fpr
    params:
        key: 查询特征名称, value: 查询值
        idx: 曲线的索引, -1表示对平均曲线查询
        search_keys: 需要知道的特征的名字, 例如'tpr', 'sens'
    return:
        一个dict, 每一项对应search_keys查找的值
    '''
    def search_point(self, key:str, value:float, idx:int, search_keys:set):
        if self.is_calculated == False:
            logger.error("search point should be called after process_data.")
            assert(0)
        assert(idx < len(self.points))
        points = self.combined_points if idx < 0 else self.points[idx]
        # 线性插值
        result = {}
        for k in search_keys:
            if k not in points.keys():
                logger.warning(f"{k} not in points.keys")
                continue
            if k == key:
                result[key] = value
            else:
                result[k] = self._interpolate1d(x=points[key],y=points[k], value=value)
        return result

    def _interpolate1d(self, x:list,y:list, value):
        # 检查x是否重合, 如果重合则挪动一下
        x,y = x.copy(), y.copy()
        valid_set = set()
        for idx in range(len(x)):
            if x[idx] not in valid_set:
                valid_set.add(x[idx])
            else:
                x[idx] = x[idx] - self.eps
        f1 = interp1d(x,y,kind='linear', fill_value='extrapolate')
        return f1([value])[0]

    def plot_curve(self, curve_type=['roc', 'prc'], title='curve', save_path=None):
        assert(curve_type in ['roc', 'prc'])
        name_x = 'fpr' if curve_type == 'roc' else 'recall'
        name_y = 'tpr' if curve_type == 'roc' else 'prec'
        len_data = len(self.data)
        if self.is_calculated == False:
            self.process_data()
        aucs = []
        plt.figure(figsize=(8,8))
        if curve_type == 'roc':
            random_guess = [x for x in np.linspace(start=0, stop=1, num=10)]
            plt.plot(random_guess, random_guess, dashes=[6, 2])
        # draw each curve
        for idx in range(len_data):
            aucs.append(sk_auc(self.points[idx][name_x], self.points[idx][name_y]))
            plt.plot(self.points[idx][name_x], self.points[idx][name_y], '0.7')
        # draw mean curve
        plt.plot(self.combined_points[name_x], self.combined_points[name_y], 'b+-')
        # add thres
        selected_thres = np.unique(np.asarray([np.argmin(np.abs(self.points[idx][name_x] - num)) for num in np.linspace(0, 1, 11)])).astype(int)
        for k, idx in enumerate(selected_thres):
            thres_str = f"{self.combined_points['thres'][idx]:.2f}"
            if k == 0:
                thres_str = "Thres:" + thres_str
            xytext = (-10, 10) if curve_type == 'roc' else (10, 10)
            ha, va = ('right', 'bottom') if curve_type == 'roc' else ('left', 'bottom')
            plt.annotate(thres_str,
                xy=[self.combined_points[name_x][idx],
                self.combined_points[name_y][idx]],
                fontsize=8, color="tab:blue",
                xytext=xytext, textcoords='offset points',
                horizontalalignment=ha, verticalalignment=va,
                arrowprops=dict(arrowstyle="->", color='tab:blue',alpha=0.7))

        auc_str = f'AUC={aucs[0]:.3f}' if len(aucs) == 1 else f'AUC={np.asarray(aucs).mean():.3f} (std {np.asarray(aucs).std():.3f})'
        if curve_type == 'roc':
            plt.annotate(auc_str, xy=[0.7, 0.05], fontsize=12)
        else:
            plt.annotate(auc_str, xy=[0.05, 0.05], fontsize=12)
        plt.title(title)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate" if curve_type == 'roc' else 'Recall')
        plt.ylabel("True Positive Rate" if curve_type == 'roc' else 'Precision')
        if save_path is not None:
            plt.savefig(save_path)
        plt.close()
    
    # 生成需要关注的信息
    def write_result(self, fp=sys.stdout):
        assert(self.is_calculated == True)
        result = {}
        for metric in ['acc', 'f1']:
            result.update({f'best_{metric}':0, f'best_{metric}_thres':0})
            for idx in range(len(self.combined_points[metric])):
                if self.combined_points[metric][idx] > result[f'best_{metric}']:
                    result[f'best_{metric}'] = self.combined_points[metric][idx]
                    result[f'best_{metric}_thres'] = self.combined_points['thres'][idx]
        for key, val in result.items():
            print(f'{key}: {val}', file=fp)

class MultiClassMetric:
    def __init__(self, class_names:list, out_dir:str) -> None:
        self.records = {'pred':[], 'gt':[]} # 记录格式: (sample, n_cls)
        self.class_names = class_names
        self.n_cls = len(class_names)
        self.out_dir = out_dir

        # calculate statistics
        self.calculated = False # lazy update
        self.cm = None # [pred, gt]
        self.recalls = None
        self.recall_std = None
    
    def update_metrics(self):
        if self.calculated:
            return
        self.calculated = True
        self.cm = np.zeros((self.n_cls, self.n_cls), dtype=np.int32)
        pred = np.concatenate(self.records['pred'], axis=0)
        gt = np.concatenate(self.records['gt'], axis=0)
        gt_label = np.argmax(gt, axis=1)
        pred_label = np.argmax(pred, axis=1)
        for idx in range(pred.shape[0]):
            self.cm[pred_label[idx]][gt_label[idx]] += 1
        self.recalls = [self.cm[idx, idx] / np.sum(self.cm[:, idx]) for idx in range(self.n_cls)]
        # 计算acc的std
        k = 5
        self.recall_std = []
        for k_idx in range(k):
            idx_start = round(k_idx * pred.shape[0] / k)
            idx_end = min(round((k_idx + 1) * pred.shape[0] / k), pred.shape[0])
            k_pred = pred[idx_start:idx_end]
            k_gt = gt[idx_start:idx_end]
            k_cm = np.zeros((self.n_cls, self.n_cls), dtype=np.int32)
            k_gt_label = np.argmax(k_gt, axis=1)
            k_pred_label = np.argmax(k_pred, axis=1)
            for idx in range(k_pred.shape[0]):
                k_cm[k_pred_label[idx]][k_gt_label[idx]] += 1
            k_recall = [k_cm[idx, idx] / np.sum(k_cm[:, idx]) for idx in range(self.n_cls)] 
            self.recall_std.append(np.mean(k_recall))
        self.recall_std = np.std(self.recall_std)

    def add_prediction(self, _prediction:np.ndarray, _gt:np.ndarray, _mask:np.ndarray=None):
        '''
        添加若干条记录, mask=True代表有效值
        prediction: (..., n_cls)
        gt: (..., n_cls) 可以是one-hot也可以是smooth label
        mask: (...) 必须和前两者保持一致
        '''
        assert(len(_prediction.shape) >= 2)
        assert(_prediction.shape == _gt.shape)
        expand_len = 1
        for x in (_prediction.shape[:-1]):
            expand_len *= x
        prediction = np.reshape(_prediction, (expand_len, self.n_cls))
        gt = np.reshape(_gt, (expand_len, self.n_cls))
        if _mask is not None:
            mask = np.reshape(_mask, (expand_len, ))
            _prediction.shape[:-1] == _mask.shape
            self.records['pred'].append(prediction[mask, :])
            self.records['gt'].append(gt[mask, :])
        else:
            self.records['pred'].append(prediction)
            self.records['gt'].append(gt)
        self.calculated = False

    def confusion_matrix(self, comment:str=''):
        '''
        输出混淆矩阵
        cm[x][y] 代表pred=x, gt=y
        '''
        self.update_metrics()
        plot_confusion_matrix(self.cm, labels=self.class_names, 
            title='Confusion matrix', save_path=os.path.join(self.out_dir, 'confusion_matrix.png'))
        cm_norm = self.cm / np.sum(self.cm, axis=0)[None, ...]
        plot_confusion_matrix(cm_norm, labels=self.class_names, 
            title='Confusion matrix(norm)', save_path=os.path.join(self.out_dir, 'confusion_matrix(norm).png'))
    
    def mean_recall(self):
        self.update_metrics()
        # 平均正确率, 每个类的权重是相等的
        mean_recall = np.mean(self.recalls)
        return mean_recall
    
    def calculate_other_metrics(self):
        result = {}
        self.update_metrics()
        # accuracy for each class
        for n in range(self.n_cls):
            result[f'cls_{n}_recall'] = self.cm[n, n] / np.sum(self.cm[:, n])
            result[f'cls_{n}_prec'] = self.cm[n, n] / np.sum(self.cm[n, :])
            other_cls = [i for i in range(self.n_cls) if i != n]
            result[f'cls_{n}_acc'] = (self.cm[n, n] + np.sum(self.cm[other_cls, :][:, other_cls])) / np.sum(self.cm)
            result[f'cls_{n}_f1'] = 2*result[f'cls_{n}_prec']*result[f'cls_{n}_recall'] / (result[f'cls_{n}_prec']+result[f'cls_{n}_recall'])
        
        result['micro-acc'] = np.diag(self.cm).sum() / (np.sum(self.cm))
        result['micro-f1'] = result['micro-acc']
        result['macro-f1'] = np.mean([result[f'cls_{n}_f1'] for n in range(self.n_cls)])
        return result
        

    def write_result(self, fp=sys.stdout):
        '''输出准确率等信息'''
        print('='*10 + 'Metric 4 classes:'+ '='*10, file=fp)
        self.update_metrics()
        # 平均正确率, 每个类的权重是相等的
        mean_acc = self.mean_recall()
        other_result = self.calculate_other_metrics()
        for idx, name in enumerate(self.class_names):
            print(f'{name} recall={self.recalls[idx]}', file=fp)
        print('='*20, file=fp)
        print(f'Mean recall={mean_acc}, std={self.recall_std}', file=fp)
        for key in other_result:
            print(f'{key} = {other_result[key]}', file=fp)



class RobustClassificationMetric:
    '''
    记录一个模型在样本不同缺失率下的性能曲线, 支持K-fold
    性能曲线计算方法为MultiClassMetric
    '''
    def __init__(self, class_names, out_dir:str) -> None:
        self.out_dir = out_dir
        self.class_names = class_names
        self.records = {}

    def add_prediction(self, missrate,  _prediction:np.ndarray, _gt:np.ndarray, _mask:np.ndarray):
        '''
        添加若干条记录, mask=True代表有效值
        prediction: (..., n_cls)
        gt: (..., n_cls) 可以是one-hot也可以是smooth label
        mask: (...) 必须和前两者保持一致
        '''
        record_key = round(missrate * 1000) # 精度支持
        if self.records.get(record_key) is None:
            self.records[record_key] = (MultiClassMetric(class_names=self.class_names, out_dir=None), [])
        self.records[record_key][1].append(MultiClassMetric(class_names=self.class_names, out_dir=None))
        self.records[record_key][1][-1].add_prediction(_prediction, _gt, _mask)
        self.records[record_key][0].add_prediction(_prediction, _gt, _mask)

    def plot_curve(self):
        '''绘制缺失率和性能关系曲线'''
        sorted_keys = sorted(list(self.records.keys()))
        missrates = np.asarray([key/1000 for key in sorted_keys])
        mean_metrics = np.asarray([self.records[key][0].mean_acc() for key in sorted_keys])
        metrics = np.asarray([[m.mean_acc() for m in self.records[key][1]] for key in sorted_keys]).T
        mean_auc = sk_auc(missrates, mean_metrics)
        aucs = [sk_auc(missrates, metrics[idx, :]) for idx in range(metrics.shape[0])]
        std = np.std(aucs)
        # draw mean curve
        plt.plot(missrates, mean_metrics, 'b+-')
        auc_str = f'AUC={mean_auc:.3f} ({std:.4f})'
        plt.annotate(auc_str, xy=[0.7, 0.05], fontsize=12)
        plt.title('Performance with missrate')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("Missing rate")
        plt.ylabel("Metric")
        save_path = os.path.join(self.out_dir, 'missrate_performance.png')
        plt.savefig(save_path)
        plt.close()

    def save_df(self, model_name):
        result = {}
        sorted_keys = sorted(list(self.records.keys()))
        missrates = np.asarray([key/1000 for key in sorted_keys])
        mean_metrics = np.asarray([self.records[key][0].mean_acc() for key in sorted_keys])
        mean_auc = sk_auc(missrates, mean_metrics)
        metrics = np.asarray([[m.mean_acc() for m in self.records[key][1]] for key in sorted_keys]).T
        aucs = [sk_auc(missrates, metrics[idx, :]) for idx in range(metrics.shape[0])]
        std = np.std(aucs)
        result['name'] = model_name
        result['std'] = std
        result['auc'] = mean_auc
        idx_cols = list(range(11))
        for idx, m in enumerate(mean_metrics):
            result[idx] = m
        result = {key:[val] for key, val in result.items()}
        df = pd.DataFrame(data=result, columns=idx_cols + ['auc','std', 'name'])
        df.to_csv(os.path.join(self.out_dir, 'missrate_performance.csv'), encoding='utf-8')
            
            
