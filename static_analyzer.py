import sklearn as sk
import pandas as pd
import numpy as np
import tools
from os.path import join as osjoin

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostClassifier, Pool
from static_sepsis_dataset import StaticSepsisDataset
from sklearn.model_selection import train_test_split
from static_model.neural_network import NNTrainer
import apriori


class StaticAnalyzer:
    def __init__(self, dataset:pd.DataFrame, target_col:str, type_dict:dict=None):
        self.conf = tools.GLOBAL_CONF_LOADER['static_analyzer']
        if type_dict is not None: # classification mode
            self.dataset = tools.convert_type({'_default': -1}, dataset, type_dict)
            self.target = target_col
            self.plots_path = self.conf['paths']['plot_dir']
            self.feas = list(dataset.columns)
            self.feas.remove(self.target)
            self.Y = dataset[target_col].to_numpy(float)
            # for ROC log
            self.roc_log_path = self.conf['paths']['ROC_path']
            tools.set_sk_random_seed(100)
            tools.clear_file(self.roc_log_path)

            # for catboost only
            self.register_values = {}
            self.shap_names = self.conf['shap_names'] # value是显示范围, 手动设置要注意左边-1项不需要, 右边太大的项要限制

            # configs:
            self.k_fold = 5
            # change available methods in t his line
            self.methods = {
                # 'logistic_regression': self.logistic_reg,
                # 'KNN_reg': self.KNN_reg,
                'catboost_reg': self.catboost_reg,
                # 'neural_network': self.neural_network
                # 'logistic_regression_4_cls' : self.logistic_regression_4_cls
            }
        else: # apriori mode
            self.dataset = dataset
            df = tools.feature_discretization(config_path=self.conf['fea_discrt_conf'],
                df=self.dataset)
            df.to_csv(self.conf['feature_discretization_out'], index=False)
            self.dataset = df

        # self.X = dataset.loc[self.feas]

    def sepsis_baseline(self, numeric_feas):
        numeric_feas, _ = numeric_feas
        kf = KFold(n_splits=self.k_fold, shuffle=True)
        metric = tools.DichotomyMetric()
        for idx, sofa_name in enumerate(self.conf['baseline_names']):
            metric.clear()
            data_num = self.dataset[sofa_name].to_numpy(np.float32)
            tprs = []
            fprs = []
            for k_idx, (_, test_index) in enumerate(kf.split(data_num)):
                Y_sofa = data_num[test_index]
                Y_test = self.Y[test_index]
                # -1会影响氧合指数取1/x, 因为值会特别小
                Y_gt = Y_test[Y_sofa > 0]
                Y_sofa = Y_sofa[Y_sofa > 0]
                if idx == 0:
                    Y_sofa = 100.0 / Y_sofa
                else:
                    Y_sofa = (Y_sofa - Y_sofa.min()) / (Y_sofa.max() - Y_sofa.min()) # to 0->1
                metric.add_prediction(pred=Y_sofa, gt=Y_gt)
            title = f'{sofa_name} ROC'
            fprs, tprs = np.asarray(fprs), np.asarray(tprs)
            metric.plot_roc(title=title, disp=False, save_path=osjoin(self.plots_path, title))

    def apriori(self):
        consequents = [{item} for item in self.conf['apriori_consequents']]
        result = apriori.runApriori(df=self.dataset, 
            consequents=consequents, 
            max_iter=self.conf['apriori_iter'], 
            minSupport=self.conf['apriori_min_support'], 
            minConfidence=self.conf['apriori_min_confidence'])
        result[0].to_csv(self.conf['paths']['apriori_items'], index=False)
        result[1].to_csv(self.conf['paths']['apriori_rules'], index=False)

    def classification(self, numeric_feas, category_feas, death_label:str):
        numeric_feas, n_idx = numeric_feas
        category_feas, c_idx = category_feas
        assert(len(list(self.dataset.columns)) == len(numeric_feas) + len(category_feas) + 2)
        data_num = self.dataset[numeric_feas].to_numpy(np.float32)
        data_ctg = self.dataset[category_feas].to_numpy(str)
        data_all = np.concatenate((data_num, data_ctg), axis=1)
        numeric_index = list(range(len(numeric_feas)))
        category_index = [len(numeric_feas) + idx for idx in range(len(category_feas))]
        
        kf = KFold(n_splits=self.k_fold, shuffle=True)
        for key in self.methods.keys():
            metric = tools.DichotomyMetric()
            if key == 'logistic_regression_4_cls':
                y_ards = self.Y.copy().astype(bool)
                y_death = self.dataset[death_label].to_numpy(bool)
                Y_4cls = tools.create_4_cls_label(y_ards, y_death)

                data_others,_ = tools.target_statistic(data_all, Y_4cls, ctg_feas=category_index, mode='greedy')
                data_others, _ = tools.fill_avg(data_others, num_feas=numeric_index)
                data_others = tools.normalize(data_others, axis=1)
                # kf.get_n_splits(data_others)
                for idx, (train_index, test_index) in enumerate(kf.split(data_others)):
                    X_train, X_test = data_others[train_index,:], data_others[test_index,:]
                    Y_train, Y_test = Y_4cls[train_index], Y_4cls[test_index]
                    Y_pred = self.methods[key](X_train, Y_train, X_test)
                    metric.add_prediction(pred=Y_pred, gt=Y_test)
            elif key == 'catboost_reg':
                # kf.get_n_splits(data_all)
                params = None
                fea_name = [self.dataset.columns[idx] for idx in n_idx + c_idx]
                for idx, (train_index, test_index) in enumerate(kf.split(data_num)):
                    X_data = data_all[train_index,:]
                    X_test = data_all[test_index,:]
                    Y_data = self.Y[train_index]
                    Y_test = self.Y[test_index]
                    Y_pred, params = self.catboost_reg(idx, X_data, X_test, Y_data, category_index, params, fea_name)
                    metric.add_prediction(pred=Y_pred, gt=Y_test)
                # plot fea importance
                items = list(self.register_values['shap'].items())
                items = np.asarray(sorted(items, key= lambda x:x[1]))
                shap_vals = np.asarray(items[:, 1], dtype=np.float32) / self.k_fold
                with open(self.conf['paths']['shap_values'], 'w') as fp:
                    for i in reversed(range(shap_vals.shape[0])):
                        fp.write(f"{shap_vals[i]},{str(items[i,0])}\n")
                tools.plot_fea_importance(
                    shap_vals, items[:, 0], save_path=self.conf['paths']['fea_importance'])
                # plot single feature importance
                if 'shap_arr' in self.register_values.keys():
                    for fea_name, vals in self.register_values['shap_arr'].items():
                        tools.plot_shap_scatter(fea_name, vals[:, 0], vals[:, 1], self.shap_names[fea_name], 
                        self.conf['paths']['plot_dir'])
            elif key == 'neural_network':
                data_nn,_ = tools.target_statistic(data_all, self.Y,
                    ctg_feas=category_index, mode='greedy')
                data_nn, _ = tools.fill_avg(data_nn, num_feas=numeric_index)
                data_nn = tools.normalize(data_nn, axis=1)
                # kf.get_n_splits(data_nn)
                valid_losses = []
                for idx, (train_index, test_index) in enumerate(kf.split(data_nn)):
                    X_train, X_test = data_nn[train_index,:], data_nn[test_index,:]
                    Y_train, Y_test = self.Y[train_index], self.Y[test_index]
                    Y_pred, valid_loss = self.methods[key](X_train, Y_train, X_test)
                    valid_losses.append(valid_loss)
                    metric.add_prediction(pred=Y_pred, gt=Y_test)
                valid_losses = np.asarray(valid_losses).T
                tools.plot_loss(data=valid_losses, title='Neural Network Validation Loss')
            else:
                data_others,_ = tools.target_statistic(data_all, self.Y,
                    ctg_feas=category_index, mode='greedy')
                data_others, _ = tools.fill_avg(data_others, num_feas=numeric_index)
                data_others = tools.normalize(data_others, axis=1)
                # kf.get_n_splits(data_others)
                for idx, (train_index, test_index) in enumerate(kf.split(data_others)):
                    X_train, X_test = data_others[train_index,:], data_others[test_index,:]
                    Y_train, Y_test = self.Y[train_index], self.Y[test_index]
                    Y_pred = self.methods[key](X_train, Y_train, X_test)
                    metric.add_prediction(pred=Y_pred, gt=Y_test)
            title = f'{key} ROC for K-fold {self.k_fold}'
            metric.plot_roc(title=title, disp=False, save_path=osjoin(self.plots_path, title))

    def logistic_reg(self, X_train:np.ndarray, Y_train:np.ndarray, X_test:np.ndarray):
        model = LogisticRegression(solver='lbfgs', max_iter=5000)
        model.fit(X_train, Y_train)
        Y_pred = model.predict_proba(X_test)[:,1] # prob of class 1
        return Y_pred

    def logistic_regression_4_cls(self, X_train:np.ndarray, Y_train:np.ndarray, X_test:np.ndarray):
        model = LogisticRegression(solver='lbfgs', max_iter=5000)
        model.fit(X_train, Y_train)
        Y_pred = model.predict_proba(X_test)[:,1] + model.predict_proba(X_test)[:,2] # prob of class 1
        return Y_pred

    def catboost_reg(self, idx, X_data, X_test, Y_data, cat_features:list, params=None, fea_name=None):
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_data, Y_data, test_size=0.15)
        pool_train = Pool(X_train, Y_train, cat_features=cat_features)
        pool_valid = Pool(X_valid, Y_valid, cat_features=cat_features)
        pool_test = Pool(data=X_test, cat_features=cat_features)
        if params is None:
            params = {
                'loss_function': 'CrossEntropy',
                'iterations': 200,
                'depth': 5,
                'learning_rate': 0.04,
            }
        
        model = CatBoostClassifier(
            iterations=params['iterations'],
            depth=params['depth'],
            loss_function=params['loss_function'],
            learning_rate=params['learning_rate'],
            od_type = "Iter",
            od_wait = 50
        )
        model.fit(pool_train, eval_set=pool_valid, use_best_model=True)
        shap_array, shap, sorted_names = tools.test_fea_importance(model, pool_valid, fea_name)
        
        # plot_shap_idx = []
        # for name in :
        # 
        # tools.plot_shap_scatter(model, pool_valid, , write_dir_path='plots')
        if idx == 0:
            self.register_values['shap'] = {}
            self.register_values['shap_arr'] = {}
        for fea_idx, name in enumerate(fea_name):
            if name not in self.shap_names.keys(): 
                continue
            pairs = np.concatenate((shap_array[:, [fea_idx]], X_valid[:, [fea_idx]].astype(np.float32)), axis=1)
            if self.register_values['shap_arr'].get(name) is None:
                self.register_values['shap_arr'][name] = pairs
            else:
                self.register_values['shap_arr'][name] = np.concatenate(
                    (self.register_values['shap_arr'][name], pairs), axis=0
                )
        
        for i in range(len(shap)):
            if self.register_values['shap'].get(sorted_names[i]) is None:
                self.register_values['shap'][sorted_names[i]] = shap[i]
            else:
                self.register_values['shap'][sorted_names[i]] += shap[i]
        return model.predict(pool_test, prediction_type='Probability')[:,1], params

    def neural_network(self, X_data, Y_data, X_test):
        trainer = NNTrainer(device='cuda:0', in_fea=X_data.shape[1], hidden=100)
        h_param = {
            'epoch': 400,
            'lr': 1e-4
        }
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_data, Y_data, test_size=0.15, shuffle=True)
        Y_pred, valid_loss = trainer.train(h_param, X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid, X_test=X_test)
        return Y_pred, valid_loss

    def KNN_reg(self, X_train:np.ndarray, Y_train:np.ndarray, X_test:np.ndarray):
        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        return Y_pred
        

if __name__ == '__main__':
    tools.set_chinese_font()
    mode = 'classification'
    dataset = StaticSepsisDataset(from_pkl=True)
    if mode == 'classification':
        analyzer = StaticAnalyzer(dataset=dataset.data_pd, target_col=dataset.target_fea, type_dict=dataset.get_type_dict())
        analyzer.sepsis_baseline(dataset.get_numeric_feas())
        analyzer.classification(dataset.get_numeric_feas(), dataset.get_category_feas(), dataset.get_death_label())
    else:
        analyzer = StaticAnalyzer(dataset=dataset.data_pd, target_col=dataset.target_fea, type_dict=None)
        analyzer.apriori()
