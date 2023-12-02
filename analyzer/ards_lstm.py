import numpy as np
import tools
import os
from tools import logger as logger
from .container import DataContainer
from tools.data import DynamicDataGenerator, LabelGenerator_cls, label_func_min, unroll, map_func
from datasets.derived_ards_dataset import MIMICIV_ARDS_Dataset
from models.ards_lstm_model import ArdsLSTMModel
from torch.utils.data.dataloader import DataLoader
import os
from os.path import join as osjoin
import torch
from tools.data import Collect_Fn
from tools.logging import SummaryWriter

class ArdsLSTMTrainer():
    def __init__(self, params:dict, dataset:MIMICIV_ARDS_Dataset, generator:DynamicDataGenerator) -> None:
        self.params = params
        self.dataset = dataset
        self.generator = generator
        self.train_dataloader = DataLoader(dataset=self.dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=Collect_Fn)
        self.valid_dataloader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=False, collate_fn=Collect_Fn)
        self.test_dataloader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=False, collate_fn=Collect_Fn)
    
    def load_checkpoint(self, p_checkpoint:str):
        result = torch.load(p_checkpoint)
        self.model = result['model'].to(self.params['device'])

    def save(self, save_path):
        torch.save({
            'model': self.model
        }, save_path)

    def cal_label_weight(self, phase:str, dataset:MIMICIV_ARDS_Dataset, generator:DynamicDataGenerator):
        dataset.mode(phase)
        result = None
        dl = DataLoader(dataset=dataset, batch_size=2000, shuffle=False, collate_fn=Collect_Fn)
        for batch in dl:
            data_dict = generator(batch['data'], batch['length'])
            mask = tools.make_mask((batch['data'].shape[0], batch['data'].shape[2]), batch['length'])
            Y_gt:np.ndarray = data_dict['label']
            weight = np.sum(unroll(Y_gt, mask=mask), axis=0)
            result = weight if result is None else result + weight
        result = 1 / result
        result = result / result.min()
        logger.info(f'Label weight: {result}')
        return result

    def train(self, fold_idx:int, summary_writer:SummaryWriter, cache_dir:str):
        self.record = {}
        # create model
        self.model = ArdsLSTMModel(
            in_channels=len(self.generator.avail_idx),
            n_cls=len(self.params['centers']),
            hidden_size=self.params['hidden_size']
        ).to(self.params['device']) # TODO add sample weight
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.params['epoch'], eta_min=self.params['lr']*0.1)
        train_cls_weight = self.cal_label_weight('train', self.dataset, self.generator)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none', weight=torch.as_tensor(train_cls_weight, device=self.params['device']))
        for epoch in range(self.params['epoch']):
            self.record[f'{fold_idx}_epoch_train_loss'], self.record[f'{fold_idx}_epoch_valid_loss'] = 0, 0
            for phase in ['train', 'valid']:
                self.dataset.mode(phase)
                for batch in self.train_dataloader:
                    data_dict = self.generator(batch['data'], batch['length'])
                    mask = torch.as_tensor(
                        tools.make_mask((batch['data'].shape[0], batch['data'].shape[2]), batch['length']),
                        dtype=bool, device=self.params['device']
                    )
                    X = torch.as_tensor(data_dict['data'], dtype=torch.float32, device=self.params['device'])
                    Y_gt:torch.Tensor = torch.as_tensor(data_dict['label'], dtype=torch.float32, device=self.params['device'])
                    Y_pred:torch.Tensor = self.model(X)
                    loss = torch.sum(self.criterion(Y_pred.permute((0, 2, 1)), Y_gt.permute((0, 2, 1))) * mask) / torch.sum(mask)
                    if phase == 'train':
                        self.opt.zero_grad()
                        loss.backward()
                        self.opt.step()
                    self.record[f'{fold_idx}_epoch_{phase}_loss'] += loss.detach().cpu().item() * batch['data'].shape[0] / len(self.dataset)
                summary_writer.add_scalar(f'{fold_idx}_epoch_{phase}_loss', self.record[f'{fold_idx}_epoch_{phase}_loss'], global_step=epoch)
            self.scheduler.step()
            if (not 'best_valid_loss' in self.record) or (self.record[f'{fold_idx}_epoch_valid_loss'] < self.record['best_valid_loss']):
                self.record['best_valid_loss'] = self.record[f'{fold_idx}_epoch_valid_loss']
                logger.info(f'Save model at epoch {epoch}, valid loss={self.record[f"{fold_idx}_epoch_valid_loss"]}')
                self.save(osjoin(cache_dir, 'model_best.pth'))
        logger.info('Done')

    def predict(self):
        self.dataset.mode('test')
        result = {'Y_gt':[], 'Y_pred':[], 'mask':[]}
        with torch.no_grad():
            for batch in self.train_dataloader:
                data_dict = self.generator(batch['data'], batch['length'])
                mask = tools.make_mask((batch['data'].shape[0], batch['data'].shape[2]), batch['length'])
                X = torch.as_tensor(data_dict['data'], dtype=torch.float32, device=self.params['device'])
                Y_gt:torch.Tensor = torch.as_tensor(data_dict['label'], dtype=torch.float32, device=self.params['device'])
                Y_pred:torch.Tensor = torch.softmax(self.model(X), dim=-1)
                result['Y_pred'].append(Y_pred.cpu().numpy())
                result['Y_gt'].append(Y_gt.cpu().numpy())
                result['mask'].append(mask)
        return {
            'Y_pred': np.concatenate(result['Y_pred'], axis=0),
            'Y_gt': np.concatenate(result['Y_gt'], axis=0),
            'mask': np.concatenate(result['mask'], axis=0)
        }
        

class ArdsLSTMAnalyzer:
    def __init__(self, params:dict, container:DataContainer) -> None:
        self.params = params
        self.paths = params['paths']
        self.dataset = MIMICIV_ARDS_Dataset()
        self.dataset.load_version(params['dataset_version'])
        self.model_name = self.params['analyzer_name']
        self.target_idx = self.dataset.idx_dict['PF_ratio']

    def run(self):
        # step 1: init variables
        out_dir = os.path.join(self.paths['out_dir'], self.model_name)
        tools.reinit_dir(out_dir, build=True)

        metric_2cls = [tools.DichotomyMetric() for _ in range(len(self.params['class_names']))]
        metric_4cls = tools.MultiClassMetric(class_names=self.params['class_names'], out_dir=out_dir)

        generator = DynamicDataGenerator(
            window_points=self.params['window'],
            n_fea=len(self.dataset.total_keys),
            label_generator=LabelGenerator_cls(
                centers=self.params['centers']
            ),
            label_func=label_func_min,
            target_idx=self.target_idx,
            limit_idx=[self.dataset.fea_idx(id) for id in self.params['limit_feas']],
            forbidden_idx=[self.dataset.fea_idx(id) for id in self.params['forbidden_feas']]
        )
        feature_names = [self.dataset.fea_label(idx) for idx in generator.avail_idx]
        print(f'Available features: {feature_names}')
        summary_writer = SummaryWriter()
        # step 2: train and predict
        for fold_idx, _ in enumerate(self.dataset.enumerate_kf()):
            trainer = ArdsLSTMTrainer(params=self.params, dataset=self.dataset, generator=generator)
            cache_dir = osjoin(out_dir, f'fold_{fold_idx}')
            tools.reinit_dir(cache_dir, build=True)
            trainer.train(fold_idx=fold_idx, summary_writer=summary_writer, cache_dir=cache_dir)
            out_dict = trainer.predict()
            out_dict['Y_pred'] = unroll(out_dict['Y_pred'],  out_dict['mask'])
            out_dict['Y_gt'] = unroll(out_dict['Y_gt'], out_dict['mask'])
            
            metric_4cls.add_prediction(out_dict['Y_pred'], out_dict['Y_gt']) # 去掉mask外的数据
            for idx, map_dict in zip([0,1,2,3], [{0:0,1:1,2:1,3:1}, {0:0,1:1,2:0,3:0}, {0:0,1:0,2:1,3:0}, {0:0,1:0,2:0,3:1}]): # TODO 这里写错了
                metric_2cls[idx].add_prediction(map_func(out_dict['Y_pred'], map_dict)[:, 1], map_func(out_dict['Y_gt'], map_dict)[:, 1])
        
        metric_4cls.confusion_matrix(comment=self.model_name)
        for idx in range(4):
            metric_2cls[idx].plot_curve(curve_type='roc', title=f'ROC for {self.params["class_names"][idx]}', save_path=osjoin(out_dir, f'roc_cls_{idx}.png'))
        
        for phase in ['train', 'valid']:
            summary_writer.plot(tags=[f'{fold}_epoch_{phase}_loss' for fold in range(5)], 
                            k_fold=True, log_y=True, title=f'{phase} loss for ards LSTM', out_path=osjoin(out_dir, f'{phase}_loss.png'))
        with open(os.path.join(out_dir, 'result.txt'), 'w') as fp:
            print('Overall performance:', file=fp)
            metric_4cls.write_result(fp)