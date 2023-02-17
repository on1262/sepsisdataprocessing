import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
from scipy.interpolate import interp1d
import pandas as pd
import json
from tqdm import tqdm
from sklearn.metrics import auc as sk_auc
from .colorful_logging import logger
from .generic import reinit_dir, remove_slash, clear_file
from .plot import draw_band, plot_single_dist, plot_reg_correlation, plot_correlation_with_quantile

'''
合并两个文件中cmp_cols都相同的样本, 同时列标签加上new和old
'''
def combine_and_select_samples(data_a: pd.DataFrame, data_b: pd.DataFrame, rename_prefix:list):
    cmp_cols = [u'唯一号', u'住院号', u'姓名', u'年龄']
    for col in cmp_cols:
        assert(col in data_a.columns and col in data_b.columns)
    # make hash dict
    a_dict = {}
    b_dict = {}
    for name in ['a', 'b']:
        data_dict = a_dict if name == 'a' else b_dict
        data_pd = data_a if name == 'a' else data_b
        for r_idx, row in data_pd.iterrows():
            key = '+'.join([str(row[col]) for col in cmp_cols])
            if key in data_dict:
                logger.warning(f'Conflict: {key}')
            else:
                data_dict[key] = r_idx
    a_rows, b_rows = [], []
    for key, val in a_dict.items():
        if key in b_dict.keys():
            a_rows.append(val)
            b_rows.append(b_dict[key])
    logger.info(f'Detected {len(a_rows)} rows able to be combined')
    data_a, data_b = data_a.loc[a_rows,:], data_b.loc[b_rows, :]
    data_a = data_a.rename(columns={col:rename_prefix[0] + col for col in data_a.columns})
    data_b = data_b.rename(columns={col:rename_prefix[1] + col for col in data_b.columns})
    data_b.index = data_a.index
    return pd.concat([data_a, data_b], axis=1, join='inner')

# 通过给定的json进行特征离散化
def feature_discretization(config_path:str, df:pd.DataFrame):
    logger.info('feature_discretization')
    with open(config_path, 'r', encoding='utf-8') as fp:
        config = json.load(fp)
    used_fea = config[u"可用特征"]
    thres_dict = config[u"离散化阈值"]
    df = df.loc[:, used_fea]
    df = df.astype({col: 'str' for col in df.columns})
    for col in df.columns:
        if col not in thres_dict.keys():
            logger.warning('skipped feature_discretization on:', col)
            continue
        for ridx in range(len(df)):
            cond_flag = False
            for cond in thres_dict[col]: # dict, example: {"大于等于":200, "小于":300,"名称":"轻度ARDS"}
                val = cond[u"名称"]
                if not pd.isna(df.at[ridx,col]):
                    flag = True
                    df_val = float(df.at[ridx,col])
                    for cond_key in cond.keys():
                        if u"大于等于" == cond_key:
                            flag = False if df_val < cond[cond_key] else flag
                        elif u"大于" in cond_key:
                            flag = False if df_val <= cond[cond_key] else flag
                        elif u"小于等于" == cond_key:
                            flag = False if df_val > cond[cond_key] else flag
                        elif u"小于" in cond_key:
                            flag = False if df_val >= cond[cond_key] else flag
                        elif u"等于" == cond_key:
                            flag = False if df_val != cond[cond_key] else flag
                    if flag:
                        df.at[ridx,col] = val
                        cond_flag = True
                        break
            if cond_flag == False:
                df.at[ridx, col] = u"NAN" # 包括正常指标和缺失值, 正常值在apriori中不予考虑
    # 预处理
    df = df.reset_index(drop=True)
    for col in df.columns:
        for ridx in range(len(df)):
            df.at[ridx, col] = col + "=" + str(df.at[ridx, col])
    return df

"""
第一次数据存在一些问题, 这段代码将第二次数据的PaO2/FiO2拷贝到第一次数据的氧合指数上
拼接依赖于唯一码, 这段代码应当只用一次
"""
def fix_feature_error_in_old_sys(old_csv: str, new_csv:str, output:str, rebuild_ards=False):
    def detect_ards_label(in_str:str)->bool:
        for fea in [u'ARDS', u'急性呼吸窘迫综合征']:
            if fea in in_str:
                return True
        return False

    old_data = pd.read_csv(old_csv, encoding='utf-8')
    new_data = pd.read_csv(new_csv, encoding='utf-8')
    try:
        now_fea = None
        now_file = 'old'
        for fea in [
            u'ARDS', u'唯一号', u'姓名', u'SOFA_氧合指数', \
            u'CIS诊断', u'CIS诊断1', u'出院诊断/死亡诊断','PaO2' \
            ]:
            now_fea = fea
            assert(fea in old_data.columns)
        now_file = 'new'
        for fea in [u'唯一号', u'姓名', u'D1_PaO2/FiO2', u'D1_PaO2']:
            now_fea = fea
            assert(fea in new_data.columns)
    except Exception as e:
        logger.error(f'fix_feature_error_in_old_sys: 特征缺失: {now_fea} in {now_file}')
        return
    # 统计信息
    statistics = {'hash_target':0, 'ARDS_target':0}
    old_data.reset_index(drop=True, inplace=True)
    new_data.reset_index(drop=True, inplace=True)
    # ARDS 标签不一定重建, 按需要调整, 
    logger.info(f'Rebuild ARDS Switch={rebuild_ards}')
    if not rebuild_ards:
        for r_idx in range(len(old_data)):
            if old_data.at[r_idx, u'ARDS'] == 1:
                statistics['ARDS_target'] += 1
    else:
        for r_idx in range(len(old_data)):
            in_str = str(old_data.at[r_idx, u'出院诊断/死亡诊断']) + \
                str(old_data.at[r_idx, u'CIS诊断']) + str(old_data.at[r_idx, u'CIS诊断1'])
            in_str = str(old_data.at[r_idx, u'CIS诊断']) + str(old_data.at[r_idx, u'CIS诊断1'])
            if detect_ards_label(in_str):
                old_data.at[r_idx, u'ARDS'] = 1
                statistics['ARDS_target'] += 1
            else:
                old_data.at[r_idx, u'ARDS'] = 0
    # 构建氧合指数哈希表
    hash_dict = {}
    for r_idx in range(len(new_data)):
        hash_dict['+'.join([new_data.at[r_idx, u'唯一号'], new_data.at[r_idx, u'姓名']])] = \
            (new_data.at[r_idx, u'D1_PaO2/FiO2'], new_data.at[r_idx, u'D1_PaO2'])
    for r_idx in range(len(old_data)):
        result = hash_dict.get(
            '+'.join([old_data.at[r_idx, u'唯一号'], old_data.at[r_idx, u'姓名']])
        )
        if result is not None:
            statistics['hash_target'] += 1
            old_data.at[r_idx, u'SOFA_氧合指数'] = result[0]
            old_data.at[r_idx, u'PaO2'] = result[1]
        else:
            old_data.at[r_idx, u'SOFA_氧合指数'] = None
            old_data.at[r_idx, u'PaO2'] = None

    old_data.to_csv(output, encoding='utf-8', index=False)
    logger.info(f'new_data样本量={len(new_data)}, \
        old_data样本量={len(old_data)}, hash_table命中=%d' % statistics['hash_target'])
    logger.info('ARDS标签占比=%.3f' % (statistics['ARDS_target'] / len(old_data)))
    logger.info(f'Output to {output}')


'''
更加完善的二分类评价指标
'''
class DichotomyMetric:
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
        self.data.append(np.stack((pred.copy(), gt.copy()), axis=1))
        self.is_calculated = False # 更新状态

    def _cal_single_point(self, Y_gt, Y_pred, thres):
        tp = np.sum(Y_pred[Y_gt > 0.5] > thres)
        fp = np.sum(Y_pred[Y_gt < 0.5] > thres)
        fn = np.sum(Y_pred[Y_gt > 0.5] < thres)
        tn = np.sum(Y_pred[Y_gt < 0.5] < thres)
        return {
            'tp': tp, 'fp':fp, 
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
        thres = [0]
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
        draw_band(ax=plt.gca(), x=self.combined_points['fpr'], y=self.combined_points['tpr'], \
            err=self.tpr_std, facecolor=f"C0", edgecolor="none", alpha=.2)
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


class DynamicPredictionMetric:
    def __init__(self, target_name:str, expanded_fea:list, out_dir:str) -> None:
        self.target_name = target_name
        self.fea_list = expanded_fea # [(time, expanded_name)]
        self.records = None
        self.out_dir = out_dir
        self.quantile_flag = False
        self.quantile_idx = None
        self.quantile_list = None
    
    def set_quantile(self, q_list, q_idx):
        self.quantile_flag = True
        self.quantile_list = q_list
        self.quantile_idx = q_idx

    def get_record(self):
        return {key:val.copy() for key, val in self.records.items()}

    # 要求对每个可行点都预测, prediction中-1代表无效数据
    def add_prediction(self, prediction:np.ndarray, gt:np.ndarray, start_idx:np.ndarray, duration:np.ndarray):
        if self.records is None:
            self.records = {
                'pred': prediction.copy(),
                'gt': gt.copy(),
                'start_idx': start_idx.copy(),
                'duration':duration.copy()
            }
            self.records['mask'] = self._make_mask(prediction.shape, start_idx, duration)
        else:
            # 所有记录的行都累积起来
            self.records['pred'] = np.concatenate((self.records['pred'], prediction),axis=(1 if self.quantile_flag else 0))
            self.records['gt'] = np.concatenate((self.records['gt'], gt), axis=0)
            self.records['start_idx'] = np.concatenate((self.records['start_idx'], start_idx), axis=0)
            self.records['duration'] = np.concatenate((self.records['duration'], duration), axis=0)
            self.records['mask'] = np.concatenate((self.records['mask'], self._make_mask(prediction.shape, start_idx, duration)), axis=0)

    def _make_mask(self, m_shape, start_idx, duration):
        if self.quantile_flag: # quantile
            mask = np.zeros((m_shape[1], m_shape[2]), dtype=bool)
        else:
            mask = np.zeros((m_shape), dtype=bool)
        for idx in range(start_idx.shape[0]):
            mask[idx, start_idx[idx]:start_idx[idx] + duration[idx]] = True
        return mask

    def plot(self, method_name:str):
        corr_dir = os.path.join(self.out_dir, remove_slash(method_name), 'correlation')
        res_dir = os.path.join(self.out_dir, remove_slash(method_name), 'residual')
        reinit_dir(write_dir_path=corr_dir)
        reinit_dir(write_dir_path=res_dir)
        self.plot_residual(res_dir=res_dir)
        self.plot_corr(corr_dir=corr_dir)

    def write_result(self, method_name:str, log_path:str):
        if self.quantile_flag:
            valid_mat = (self.records['pred'][self.quantile_idx, ...] > 0) * self.records['mask']
            pred = self.records['pred'][self.quantile_idx, ...][valid_mat]
            gt = self.records['gt'][valid_mat]
        else:
            valid_mat = (self.records['pred'] > 0) * self.records['mask']
            pred = self.records['pred'][valid_mat]
            gt = self.records['gt'][valid_mat]
        rmse = np.sqrt(np.mean((pred-gt)**2))
        mae = np.abs(pred - gt).mean()
        bias = (pred - gt).mean()
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('='*10 + f'Method: {method_name}' + '='*10 + '\n')
            f.write(f'Root Mean Squared Error(RMSE)={rmse}'+ '\t') # 误差的均方根值
            f.write(f'Mean Absolute Error(MAE)={mae}'+ '\t') # 误差的平均值
            f.write(f'Prediction bias={bias}'+ '\n')
        

    # 绘制残差分布
    def plot_residual(self, res_dir:str):
        logger.info('DynamicPredictionMetric: Plotting residual')
        os.makedirs(res_dir,exist_ok=True)
        if self.records is None:
            logger.error('DynamicPredictionMetric: no record')
            return
        for idx, (time, old_name) in enumerate(self.fea_list):
            if self.quantile_flag:
                r_pred = self.records['pred'][self.quantile_idx, ...]
            else:
                r_pred = self.records['pred']
            valid_mat = (r_pred[:, idx] > 0) * self.records['mask'][:, idx]
            if np.any(valid_mat):
                pred = r_pred[:,idx][valid_mat]
                gt = self.records['gt'][:,idx][valid_mat]
                res = pred - gt
                plot_single_dist(data=res, data_name='Residual distribution: ' + old_name, save_path=os.path.join(res_dir, f'{time}_name={remove_slash(old_name)}.png'), discrete=False)
            else:
                logger.warning(f'Plot residual: no valid row in name={old_name}')

    # 绘制预测值和真实值的关联度
    def plot_corr(self, corr_dir:str, comment:str=''):
        logger.info('DynamicPredictionMetric: Plotting Correlation')
        
        os.makedirs(corr_dir, exist_ok=True)
        if self.records is None:
            logger.error('DynamicPredictionMetric: no record')
            return
        if not self.quantile_flag:
            for idx, (time, old_name) in enumerate(self.fea_list):
                valid_mat = (self.records['pred'][:, idx] > 0) * self.records['mask'][:, idx]
                if np.any(valid_mat):
                    pred = self.records['pred'][:,idx][valid_mat]
                    gt = self.records['gt'][:,idx][valid_mat]
                    plot_reg_correlation(X=gt[:,None], fea_names=[f'GT_T={time}' + '_' + remove_slash(old_name)], Y=pred, target_name='Prediction', \
                        restrict_area=True, write_dir_path=corr_dir, comment=comment)
                else:
                    logger.warning(f'Plot residual: no valid row in name={old_name}')
            # plot all correlation
            valid_mat = (self.records['pred'] > 0) * self.records['mask']
            pred = self.records['pred'][valid_mat]
            gt = self.records['gt'][valid_mat]
            plot_reg_correlation(
                X=gt[:,None], fea_names=['ALL_gt'], Y=pred, target_name='ALL_Prediction', restrict_area=True, write_dir_path=corr_dir, comment=comment)
        else:
            for idx, (time, old_name) in enumerate(self.fea_list):
                valid_mat = (self.records['pred'][self.quantile_idx, :, idx] > 0) * self.records['mask'][:, idx]
                if np.any(valid_mat):
                    pred = self.records['pred'][:, :, idx][:, valid_mat]
                    gt = self.records['gt'][:, idx][valid_mat]
                    plot_correlation_with_quantile(X_pred=pred, 
                        x_name=[f'GT_T={time}' + '_' + remove_slash(old_name)], Y_gt=gt, target_name='Prediction', quantile=self.quantile_list, equal_lim=True,
                        plot_dash=True, write_dir_path=corr_dir, comment=comment)
                else:
                    logger.warning(f'Plot residual: no valid row in name={old_name}')
            valid_mat = (self.records['pred'][self.quantile_idx, ...] > 0) * self.records['mask']
            pred = self.records['pred'][:, valid_mat]
            gt = self.records['gt'][valid_mat]
            plot_correlation_with_quantile(X_pred=pred, x_name=['ALL_Prediction'], Y_gt=gt, equal_lim=True, target_name='ALL_gt',quantile=self.quantile_list, plot_dash=True, write_dir_path=corr_dir, comment=comment)


# 分割icu_events
def split_csv(path, max_len=100000, out_folder:str=None):
    dir_path, name = os.path.split(path)
    out_prefix = name.split('.')[0]
    file_count = 1
    nfp = None
    # TODO tqdm
    with open(path, 'r',encoding='utf-8') as fp:
        title = fp.readline()
        tmp = fp.readline() + '\n'
        count = 0
        while(tmp):
            if count == max_len:
                nfp.close()
                count = 0
                nfp = None
            if count == 0:
                nfp = open(os.path.join(out_folder, out_prefix + str(file_count) + '.csv'), 'w', encoding='utf-8')
                nfp.write(title)
            nfp.write(tmp)
            tmp = fp.readline() + '\n'
        if nfp is not None:
            nfp.close()
    




