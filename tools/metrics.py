import matplotlib.pyplot as plt
import numpy as np
import os, sys
from scipy.interpolate import interp1d
from sklearn.metrics import auc as sk_auc
from .colorful_logging import logger
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
        self.interpolate_eps = 0.0001
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
        self.data.append(np.stack((pred.copy(), gt.copy()), axis=1))
        self.is_calculated = False # 更新状态

    def _cal_single_point(self, Y_gt, Y_pred, thres):
        tp = np.sum(Y_pred[Y_gt > 0.5] > thres)
        fp = np.sum(Y_pred[Y_gt < 0.5] > thres)
        fn = np.sum(Y_pred[Y_gt > 0.5] < thres)
        tn = np.sum(Y_pred[Y_gt < 0.5] < thres)
        return {
            'tp': tp, 'fp':fp, 
            'tn': tn, 'fn':fn,
            'tpr':tp/(tp+fn), 'fpr':fp/(fp+tn), 
            'acc':(tp+tn)/(tp+fp+tn+fn),'sens':tp/(tp+fn), 'spec':tn/(tn+fp), 'thres':thres
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
            next_thres = sorted_pred[idx]
            if np.abs(next_thres-thres[-1]) > self.thres_eps:
                thres.append(next_thres)
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
    线性插值得到曲线上的某个点
    params:
        key: 点的特征, value: 点的值
        idx: 曲线的索引, -1表示对平均曲线查询
        search_keys: 需要知道的特征的名字, 例如'tpr', 'sens'
    return:
        一个dict, 每一项对应search_keys查找的值
    '''
    def search_point(self, key:str, value:float, idx, search_keys:set):
        if self.is_calculated == False:
            logger.error("search point should be called after process_data.")
            assert(0)
        assert(idx < len(self.points))
        if idx < 0:
            points = self.combined_points
        else:
            points = self.points[idx]
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
                x[idx] = x[idx] - self.interpolate_eps
        f1 = interp1d(x,y,kind='linear', fill_value='extrapolate')
        return f1([value])[0]

    def plot_roc(self, title='roc', disp=False, save_path=None):
        len_data = len(self.data)
        if self.is_calculated == False:
            self.process_data()
        aucs = []
        random_guess = [x for x in np.linspace(start=0, stop=1, num=10)]
        plt.figure(figsize=(8,8))
        plt.plot(random_guess, random_guess, dashes=[6, 2])
        # draw each curve
        for idx in range(len_data):
            aucs.append(sk_auc(self.points[idx]['fpr'], self.points[idx]['tpr']))
            plt.plot(self.points[idx]['fpr'], self.points[idx]['tpr'], '0.7')
        # draw mean curve
        plt.plot(self.combined_points['fpr'], self.combined_points['tpr'], 'b+-')
        # add thres
        for k in range(self.annotate_thres):
            idx = round((len(self.thres)-1) * k / (self.annotate_thres - 1))
            thres_str = f"{self.combined_points['thres'][idx]:.2f}"
            if k == 0:
                thres_str = "Thres:" + thres_str
            plt.annotate(thres_str,
                xy=[self.combined_points['fpr'][idx],
                self.combined_points['tpr'][idx]],
                fontsize=8, color="tab:blue",
                xytext=(-10, 10), textcoords='offset points',
                horizontalalignment='right', verticalalignment='bottom',
                arrowprops=dict(arrowstyle="->", color='tab:blue',alpha=0.7))

        # draw std band
        # draw_band(ax=plt.gca(), x=self.combined_points['fpr'], y=self.combined_points['tpr'], \
        #     err=self.tpr_std, facecolor=f"C0", edgecolor="none", alpha=.2)
        auc_str = f'AUC={aucs[0]:.3f}' if len(aucs) == 1 else f'AUC={np.asarray(aucs).mean():.3f} (std {np.asarray(aucs).std():.3f})'

        plt.annotate(auc_str, xy=[0.7, 0.05], fontsize=12)
        plt.title(title)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        if disp:
            plt.show()
        elif save_path is not None:
            plt.savefig(save_path)
        plt.close()
    
    # 生成需要关注的信息
    def generate_info(self):
        assert(self.is_calculated == True)
        # best acc
        result = {'best_acc':0, 'best_acc_thres':0}
        for idx in range(len(self.combined_points['acc'])):
            if self.combined_points['acc'][idx] > result['best_acc']:
                result['best_acc'] = self.combined_points['acc'][idx]
                result['best_acc_thres'] = self.combined_points['thres'][idx]
        return result

class MultiClassMetric:
    '''
    多分类指标, 主要是输出混淆矩阵
    '''
    def __init__(self, class_names:list, out_dir:str) -> None:
        self.records = {'pred':[], 'gt':[]} # 记录格式: (sample, n_cls)
        self.class_names = class_names
        self.n_cls = len(class_names)
        self.out_dir = out_dir

        # calculate statistics
        self.calculated = False # lazy update
        self.cm = None
        self.accs = None
        self.acc_std = None
    
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
        self.accs = [self.cm[idx, idx] / np.sum(self.cm[:, idx]) for idx in range(self.n_cls)]
        # 计算acc的std
        k = 5
        self.acc_std = []
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
            k_accs = [k_cm[idx, idx] / np.sum(k_cm[:, idx]) for idx in range(self.n_cls)]
            self.acc_std.append(np.mean(k_accs))
        self.acc_std = np.std(self.acc_std)

    def add_prediction(self, _prediction:np.ndarray, _gt:np.ndarray, _mask:np.ndarray):
        '''
        添加若干条记录, mask=True代表有效值
        prediction: (..., n_cls)
        gt: (..., n_cls) 可以是one-hot也可以是smooth label
        mask: (...) 必须和前两者保持一致
        '''
        assert(len(_prediction.shape) >= 2)
        assert(_prediction.shape == _gt.shape and _prediction.shape[:-1] == _mask.shape)
        expand_len = 1
        for x in (_prediction.shape[:-1]):
            expand_len *= x
        prediction = np.reshape(_prediction, (expand_len, self.n_cls))
        gt = np.reshape(_gt, (expand_len, self.n_cls))
        mask = np.reshape(_mask, (expand_len, ))
        self.records['pred'].append(prediction[mask, :])
        self.records['gt'].append(gt[mask, :])
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
    
    def mean_acc(self):
        self.update_metrics()        
        # 平均正确率, 每个类的权重是相等的
        mean_acc = np.mean(self.accs)
        return mean_acc
    
    def write_result(self, fp=sys.stdout):
        '''输出准确率等信息'''
        print('='*10 + 'Metric 4 classes:'+ '='*10, file=fp)
        self.update_metrics()
        # 平均正确率, 每个类的权重是相等的
        mean_acc = self.mean_acc()
        for idx, name in enumerate(self.class_names):
            print(f'{name} accuracy={self.accs[idx]}', file=fp)
        print('='*20, file=fp)
        print(f'Mean accuracy={mean_acc}, std={self.acc_std}', file=fp)



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
            
            
