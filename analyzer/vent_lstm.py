import numpy as np
import tools
import os
from tools import logger as logger
from .container import DataContainer
from tools.data import DynamicDataGenerator, LabelGenerator_cls, label_func_max
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

    def train(self, summary_writer:SummaryWriter, cache_dir:str):
        self.record = {}
        self.dataset.mode('train')
        # create model
        self.model = VentLSTMModel(
            in_channels=len(self.generator.avail_idx),
            n_cls=len(self.params['centers']),
            hidden_size=self.params['hidden_size']
        ).to(self.params['device']) # TODO add sample weight
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        for epoch in self.params['epoch']:
            self.record['epoch_train_loss'], self.record['epoch_valid_loss'] = 0, 0
            for phase in ['train', 'valid']:
                for batch in self.train_dataloader:
                    data_dict = self.generator(batch['data'], batch['length'])
                    mask = torch.as_tensor(
                        tools.make_mask((batch['data'].shape[0], batch['data'].shape[2]), batch['length']),
                        dtype=bool, device=self.params['device']
                    )
                    X, Y_gt = data_dict['data'].to(self.params['device']), data_dict['label'].to(self.params['device'])
                    Y_pred = self.model(X)
                    loss = torch.sum(self.criterion(Y_pred, Y_gt) * mask) / torch.sum(mask)
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
                X, Y_gt = data_dict['data'].to(self.params['device']), data_dict['label'].to(self.params['device'])
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
        generator = DynamicDataGenerator(
            window_points=self.params['window'],
            n_fea=len(self.dataset.total_keys),
            label_generator=LabelGenerator_cls(
                centers=self.params['centers']
            ),
            label_func=label_func_max, # predict most severe ventilation in each bin
            target_idx=self.target_idx,
            limit_idx=[],
            forbidden_idx=[]
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
            # metric_2cls.add_prediction(map_func(Y_pred)[..., 1].flatten(), map_func(Y_test)[..., 1].flatten())
        
        metric_3cls.confusion_matrix(comment=self.model_name)
        # metric_2cls.plot_roc(title=f'{self.model_name} model ROC (4->2 cls)', save_path=os.path.join(out_dir, f'{self.model_name}_ROC.png'))
        summary_writer.plot(tags=['train_loss'], k_fold=True, log_y=True, title='Train loss for vent LSTM', out_path=osjoin(out_dir, 'train_loss.png'))
        summary_writer.plot(tags=['valid_loss'], k_fold=True, log_y=True, title='Valid loss for vent LSTM', out_path=osjoin(out_dir, 'valid_loss.png'))
        with open(os.path.join(out_dir, 'result.txt'), 'w') as fp:
            print('Overall performance:', file=fp)
            metric_3cls.write_result(fp)