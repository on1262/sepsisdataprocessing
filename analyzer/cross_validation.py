import numpy as np
from sklearn.model_selection import KFold
import tools
import os
from tools import logger as logger
from .container import DataContainer
from tools.data import SliceDataGenerator, LabelGenerator_cls, cal_label_weight
from catboost import CatBoostClassifier, Pool
from os.path import join as osjoin
from datasets.cv_dataset import CrossValidationDataset
from datasets.derived_ards_dataset import MIMICIV_ARDS_Dataset
from datasets.derived_raw_dataset import MIMICIV_Raw_Dataset



class CV_Analyzer:
    def __init__(self, params:dict, container:DataContainer) -> None:
        self.params = params
        self.paths = params['paths']
        self.container = container
        self.model_name = self.params['analyzer_name']
        # copy attribute from container
        self.mimic_raw_dataset = MIMICIV_Raw_Dataset()
        self.mimic_raw_dataset.load_version('raw_version')
        self.mimic_dataset = MIMICIV_ARDS_Dataset()
        self.mimic_dataset.load_version('no_fill_version')
        # prepare mimic-iv data
        self.mimic_data = self.mimic_dataset.data
        # initialize
        self.out_dir = os.path.join(self.paths['out_dir'], self.model_name)
        tools.reinit_dir(self.out_dir, build=True)
        # prepare cross validation dataset
        self.cv_dataset = CrossValidationDataset()
        
    def prepare_cross_validation_data(self):
        # alignment
        # init validation out dir
        val_out_dir = osjoin(self.out_dir, 'seperate_validation')
        tools.reinit_dir(val_out_dir)
        # generate label and mask
        self.val_generator = SliceDataGenerator(
            window_points=self.params['window'],
            n_fea=len(self.cv_dataset.total_keys),
            label_generator=LabelGenerator_cls(
                centers=self.params['centers'],
                soft_label=False,
                smooth_band=self.params['smoothing_band']
            ),
            target_idx=self.cv_dataset.total_keys.index('dX_PaO2（mmHg） / FiO2（%）')
        )
        
        # make validation data into slice
        mimic_limit_idx = sorted([self.mimic_dataset.idx_dict[n] for n in self.params['feature_limit']]) # generator会对idx排序
        self.fea_names = [self.mimic_dataset.fea_label(idx) for idx in mimic_limit_idx]
        val_fea_names = [
            self.params['alignment_dict'][self.mimic_dataset.fea_label(idx)] for idx in mimic_limit_idx]
        
        val_idx = [self.cv_dataset.total_keys.index(name) for name in val_fea_names]
        cv_result = self.val_generator('val_cv', self.cv_dataset.data, seq_lens=self.cv_dataset.seqs_len)
        X_cv, Y_cv = cv_result['data'][:, val_idx], cv_result['label']
        return X_cv, Y_cv

    def train(self, dataset, label):
        out_dir = os.path.join(self.paths['out_dir'], self.model_name, label)
        tools.reinit_dir(out_dir, build=True)
        metric_4cls = tools.MultiClassMetric(class_names=self.params['class_names'], out_dir=os.path.join(out_dir, '4cls_mimic'))
        metric_4cls_sep_val = tools.MultiClassMetric(class_names=self.params['class_names'], out_dir=os.path.join(out_dir, '4cls_sep_val'))
        tools.reinit_dir(os.path.join(out_dir, '4cls_mimic'), build=True)
        tools.reinit_dir(os.path.join(out_dir, '4cls_sep_val'), build=True)
        # step 3: generate mimic-iv labels
        mimic_limit_idx = [dataset.idx_dict[n] for n in self.params['feature_limit']]
        
        # fea_names = [self.dataset.get_fea_label(idx) for idx in generator.available_idx()]
        imp_logger = tools.TreeFeatureImportance(fea_names=self.fea_names)
        generator = SliceDataGenerator(
            window_points=self.params['window'], 
            n_fea=len(dataset.total_keys),
            label_generator=LabelGenerator_cls(
                centers=self.params['centers'],
                soft_label=False,
                smooth_band=self.params['smoothing_band']
            ),
            target_idx=dataset.idx_dict['PF_ratio'],
            limit_idx=mimic_limit_idx
        )
        # step 4: train and predict
        for idx, (train_index, valid_index, test_index) in enumerate(dataset.enumerate_kf()): 
            # train model on mimic-iv
            train_result = generator(f'{idx}_train', dataset.data[train_index, :, :], dataset.seqs_len[train_index])
            X_train, Y_train = train_result['data'], train_result['label']
            valid_result = generator(f'{idx}_train', dataset.data[valid_index, :, :], dataset.seqs_len[valid_index])
            X_valid, Y_valid = valid_result['data'], valid_result['label']
            test_result = generator(f'{idx}_test', dataset.data[test_index, :, :], dataset.seqs_len[test_index])
            X_test, Y_test = test_result['data'], test_result['label']
            model = CatBoostClassifier(
                iterations=self.params['iterations'],
                learning_rate=self.params['learning_rate'],
                loss_function=self.params['loss_function'],
                class_weights=cal_label_weight(4, Y_train)
            )
            pool_train = Pool(X_train, Y_train.argmax(axis=-1))
            pool_valid = Pool(X_valid, Y_valid.argmax(axis=-1))
            model.fit(pool_train, eval_set=pool_valid)
            Y_pred = model.predict_proba(X_test)
            metric_4cls.add_prediction(Y_pred, Y_test) # 去掉mask外的数据
            # test model on cross validation dataset
            cv_pred = model.predict_proba(self.X_cv)
            metric_4cls_sep_val.add_prediction(cv_pred, self.Y_cv)

            imp_logger.add_record(model, self.X_cv)
            dataset.mode('all') # 恢复原本状态
        
        # step 5: result explore
        imp_logger.plot_beeswarm(os.path.join(out_dir, f'cv_shap_{label}.png'))
        single_imp_out = os.path.join(out_dir, 'single_shap')
        tools.reinit_dir(single_imp_out, build=True)
        imp_logger.plot_single_importance(out_dir=single_imp_out, select=10)
        metric_4cls.confusion_matrix(comment=self.model_name + '_all')
        metric_4cls_sep_val.confusion_matrix(comment='cross validation k-fold=5')

        return {
            'metric_4cls': metric_4cls,
            'metric_4cls_seq_val': metric_4cls_sep_val
        }

    def run(self):
        # init cross validation data
        self.X_cv, self.Y_cv = self.prepare_cross_validation_data()
        # step 2: init variables
        
        pipeline_result = self.train(dataset=self.mimic_dataset, label='with_pipeline')
        raw_result = self.train(dataset=self.mimic_raw_dataset, label='raw')
        
        with open(os.path.join(self.out_dir, 'result.txt'), 'a') as f:
            print('mimic-iv with pipeline:', file=f)
            pipeline_result['metric_4cls'].write_result(f)
            print('mimic-iv without pipeline:', file=f)
            raw_result['metric_4cls'].write_result(f)

            print('cross validation with pipeline:', file=f)
            pipeline_result['metric_4cls_seq_val'].write_result(f)
            print('cross validation without pipeline:', file=f)
            raw_result['metric_4cls_seq_val'].write_result(f)

