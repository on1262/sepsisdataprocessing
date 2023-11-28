import numpy as np
import tools
import os
from tools import logger as logger
from .container import DataContainer
from tools.data import DynamicDataGenerator, LabelGenerator_cls, label_func_max, map_func
from datasets.derived_vent_dataset import MIMICIV_Vent_Dataset
from models.vent_lstm_model import VentLSTMModel
from torch.utils.data.dataloader import DataLoader
import os
from os.path import join as osjoin
import torch
from tools.data import Collect_Fn
from tools.logging import SummaryWriter


class VentLSTMTrainer():
    def __init__(self, params:dict, dataset:MIMICIV_Vent_Dataset, generator:DynamicDataGenerator) -> None:
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

    def cal_label_weight(self, phase:str, dataset:MIMICIV_Vent_Dataset, generator:DynamicDataGenerator):
        dataset.mode(phase)
        result = None
        dl = DataLoader(dataset=self.dataset, batch_size=1, shuffle=False, collate_fn=Collect_Fn)
        for batch in dl:
            data_dict = self.generator(batch['data'], batch['length'])
            mask = tools.make_mask((batch['data'].shape[0], batch['data'].shape[2]), batch['length']).flatten()
            Y_gt:np.ndarray = data_dict['label']
            if result is None:
                result = np.sum(Y_gt[0, mask, :], axis=0)
            else:
                result += np.sum(Y_gt[0, mask, :], axis=0)
        result = 1 / result
        result = result / result.sum()
        logger.info(f'Label weight: {result}')
        return result


    def train(self, summary_writer:SummaryWriter, cache_dir:str):
        self.record = {}
        # create model
        self.model = VentLSTMModel(
            in_channels=len(self.generator.avail_idx),
            n_cls=len(self.params['centers']),
            hidden_size=self.params['hidden_size']
        ).to(self.params['device']) # TODO add sample weight
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])
        train_cls_weight = self.cal_label_weight('train', self.dataset, self.generator)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none', weight=torch.as_tensor(train_cls_weight, device=self.params['device']))
        for epoch in range(self.params['epoch']):
            self.record['epoch_train_loss'], self.record['epoch_valid_loss'] = 0, 0
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
                    self.record[f'epoch_{phase}_loss'] += loss.detach().cpu().item() * batch['data'].shape[0] / len(self.dataset)
                summary_writer.add_scalar(f'{phase}_loss', self.record[f'epoch_{phase}_loss'], global_step=epoch)
            if (not 'best_valid_loss' in self.record) or (self.record['epoch_valid_loss'] < self.record['best_valid_loss']):
                self.record['best_valid_loss'] = self.record['epoch_valid_loss']
                logger.info(f'Save model at epoch {epoch}, valid loss={self.record["epoch_valid_loss"]}')
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
                Y_pred:torch.Tensor = self.model(X)
                result['Y_pred'].append(Y_pred.cpu().numpy())
                result['Y_gt'].append(Y_gt.cpu().numpy())
                result['mask'].append(mask)
        return {
            'Y_pred': np.concatenate(result['Y_pred'], axis=0),
            'Y_gt': np.concatenate(result['Y_gt'], axis=0),
            'mask': np.concatenate(result['mask'], axis=0)
        }
        

class VentLSTMAnalyzer:
    def __init__(self, params:dict, container:DataContainer) -> None:
        self.params = params
        self.paths = params['paths']
        self.dataset = MIMICIV_Vent_Dataset()
        self.dataset.load_version(params['dataset_version'])
        self.model_name = self.params['analyzer_name']
        self.target_idx = self.dataset.idx_dict['vent_status']

    def run(self):
        # step 1: init variables
        out_dir = os.path.join(self.paths['out_dir'], self.model_name)
        tools.reinit_dir(out_dir, build=True)
        # metric_2cls = tools.DichotomyMetric()
        metric_3cls = tools.MultiClassMetric(class_names=self.params['class_names'], out_dir=out_dir)
        metric_2cls = [tools.DichotomyMetric() for _ in range(3)]

        generator = DynamicDataGenerator(
            window_points=self.params['window'],
            n_fea=len(self.dataset.total_keys),
            label_generator=LabelGenerator_cls(
                centers=self.params['centers']
            ),
            label_func=label_func_max, # predict most severe ventilation in each bin
            target_idx=self.target_idx,
            limit_idx=[],
            forbidden_idx=[self.dataset.idx_dict[id] for id in ['hosp_expire', 'vent_status']]
        )
        summary_writer = SummaryWriter()
        # step 2: train and predict
        for fold_idx, _ in enumerate(self.dataset.enumerate_kf()):
            trainer = VentLSTMTrainer(params=self.params, dataset=self.dataset, generator=generator)
            cache_dir = osjoin(out_dir, f'fold_{fold_idx}')
            tools.reinit_dir(cache_dir, build=True)
            trainer.train(summary_writer=summary_writer, cache_dir=cache_dir)
            out_dict = trainer.predict()
            
            metric_3cls.add_prediction(out_dict['Y_pred'], out_dict['Y_gt'], out_dict['mask'])
            for idx, map_dict in zip([0,1,2], [{0:0,1:1,2:1}, {0:0,1:1,2:0}, {0:0,1:0,2:1}]): # TODO 这里写错了
                metric_2cls[idx].add_prediction(map_func(out_dict['Y_pred'], map_dict)[:, 1], map_func(out_dict['Y_gt'], map_dict)[:, 1])
            # metric_2cls.add_prediction(map_func(Y_pred)[..., 1].flatten(), map_func(Y_test)[..., 1].flatten())
        
        metric_3cls.confusion_matrix(comment=self.model_name)
        for idx in range(3):
            metric_2cls[idx].plot_roc(f'ROC for {self.params["class_names"][idx]}', save_path=osjoin(out_dir, f'roc_cls_{idx}.png'))
        summary_writer.plot(tags=['train_loss'], k_fold=True, log_y=True, title='Train loss for vent LSTM', out_path=osjoin(out_dir, 'train_loss.png'))
        summary_writer.plot(tags=['valid_loss'], k_fold=True, log_y=True, title='Valid loss for vent LSTM', out_path=osjoin(out_dir, 'valid_loss.png'))
        with open(os.path.join(out_dir, 'result.txt'), 'w') as fp:
            print('Overall performance:', file=fp)
            metric_3cls.write_result(fp)