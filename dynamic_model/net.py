import torch.nn as nn
import numpy as np
import torch
import tools
from tools import logger as logger
from torch.utils.data.dataloader import DataLoader
import os
from tqdm import tqdm


class LSTMPredictor(nn.Module):
    def __init__(self, dev:str, in_channels:int, hidden_size=128, out_mean=0, dp=0) -> None:
        super().__init__()
        self.device = torch.device(dev)
        self.norm = nn.BatchNorm1d(num_features=in_channels)
        self.ebd = nn.Linear(in_features=in_channels, out_features=in_channels)
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, batch_first=True, dropout=dp, device=self.device)
        self.den = nn.Linear(in_features=hidden_size, out_features=1, device=self.device)
        self.c_0 = nn.Parameter(torch.zeros((1, 1, hidden_size), device=self.device), requires_grad=True)
        self.h_0 = nn.Parameter(torch.zeros((1, 1, hidden_size), device=self.device), requires_grad=True)
        self.out_mean = out_mean

    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        # x: (batch, feature, time)
        # mask: (batch, time)
        x = self.norm(x)
        # x: (batch, feature, time)
        x = self.ebd(x.transpose(1,2))
        # x: (batch, time, feature)
        x, _ = self.lstm(x, (self.h_0.expand(-1, x.size(0), -1).contiguous(), self.c_0.expand(-1, x.size(0), -1).contiguous()))
        # x: (batch, time, hidden_size)
        x = self.den(x).squeeze(2)
        # x: (batch, time)
        # time=T位置是T+1的预测值
        return ((x+1) * mask) * self.out_mean # 这是为了减少权重变化的幅度

class QuantileLoss(nn.Module):
    def __init__(self, alpha:float):
        super().__init__()
        self.alpha = alpha

    def forward(self, _pred:torch.Tensor, _gt:torch.Tensor, _mask:torch.Tensor) -> torch.Tensor:
        pred, gt, mask = _pred[:, :-1], _gt[:, 1:], _mask[:, 1:] # T=1对齐
        grt = (pred <= gt)
        res = torch.abs(pred - gt) * mask
        return torch.sum(self.alpha * res * grt + (1-self.alpha) * res * torch.logical_not(grt)) / torch.sum(mask)

class Dataset():
    def __init__(self, params:dict, data:np.ndarray, mask:np.ndarray, train_index, valid_index, test_index) -> None:
        self.mode = 'train' # train/valid/test
        self.ts_mode = params['ts_mode']
        self.train_index = train_index
        self.valid_index = valid_index
        self.test_index = test_index
        self.device = torch.device(params['device'])
        self.data = torch.as_tensor(data, dtype=torch.float32, device='cpu') # size: (sample, [sta_fea, dyn_fea], time_idx)
        self.target_name = params['target']
        self.fea_names = params['fea_names']
        self.target_idx = [False if name != self.target_name else True for name in self.fea_names].index(True)
        self.mask = torch.as_tensor(mask, dtype=torch.bool)
        self.target_mean = self._cal_mean()
        logger.info(f'Target mean={self.target_mean:.2f}')
    
    def _cal_mean(self):
        target_mean = 0
        for idx in range(self.data.size(0)):
            target_mean += torch.mean(self.data[idx, self.target_idx, self.mask[idx,:]])
        return target_mean / self.data.size(0)
    def set_mode(self, mode:str):
        self.mode = mode
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.data[self.train_index[idx], :,:], self.mask[self.train_index[idx], :]
        elif self.mode == 'valid':
            return self.data[self.valid_index[idx], :,:], self.mask[self.valid_index[idx], :]
        else:
            return self.data[self.test_index[idx], :,:], self.mask[self.test_index[idx], :]

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_index)
        elif self.mode == 'valid':
            return len(self.valid_index)
        else:
            return len(self.test_index)

def collate_fn(data:list):
    return torch.stack([d[0] for d in data], dim=0), torch.stack([d[1] for d in data], dim=0)

class Trainer():
    def __init__(self, params:dict, dataset:Dataset) -> None:
        self.params = params
        self.device = torch.device(self.params['device'])
        self.cache_path = params['cache_path']
        tools.reinit_dir(self.cache_path, build=True)
        self.quantile = None
        if 'quantile' in self.params.keys():
            self.quantile = self.params['quantile']
            self.models = [LSTMPredictor(params['device'], params['in_channels'], out_mean=dataset.target_mean) for _ in range(len(self.quantile))]
            self.criterions = [QuantileLoss(alpha) for alpha in self.quantile]
            self.opts = [torch.optim.Adam(params=self.models[idx].parameters(), lr=params['lr']) for idx in range(len(self.quantile))]
        else:
            self.models = [LSTMPredictor(params['device'], params['in_channels'], out_mean=dataset.target_mean)]
            self.criterions = [QuantileLoss(alpha=params['alpha'])] # TODO maybe other loss
            self.opts = [torch.optim.Adam(params=self.models[0].parameters(), lr=params['lr'])]
        self.dataset = dataset
        self.target_idx = dataset.target_idx
        self.train_dataloader = DataLoader(dataset=self.dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_fn)
        self.valid_dataloader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
        self.test_dataloader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        self.register_vals = {'train_loss':[], 'valid_loss':[]}
    
    def plot_loss(self, out_path:str):
        data = np.asarray([self.register_vals['train_loss'], self.register_vals['train_loss']])
        tools.plot_loss(data=data, title='Loss for LSTM model', legend=['train', 'valid'], out_path=out_path)
    

    def train(self):
        if self.quantile is not None:
            for idx,q in enumerate(self.quantile):
                logger.info(f'Training quantile={q}')
                self._train(idx)
        else:
            self._train(0)
    
    def _train(self, model_idx):
        cache_path = os.path.join(self.cache_path, str(model_idx))
        tools.reinit_dir(cache_path)
        self.models[model_idx] = self.models[model_idx].to(self.device)
        with tqdm(total=self.params['epochs']) as tq:
            tq.set_description(f'Training, Epoch')
            best_epoch = 0
            best_valid_loss = np.inf
            for epoch in range(self.params['epochs']):
                register_vals = {'train_loss':0, 'valid_loss':0}
                # train phase
                self.dataset.set_mode('train')
                self.models[model_idx].train()
                for data in self.train_dataloader:
                    x, mask = data[0].to(self.device), data[1].to(self.device)
                    pred = self.models[model_idx](x, mask)
                    loss = self.criterions[model_idx](pred, x[:,self.target_idx, :], mask)
                    self.opts[model_idx].zero_grad()
                    loss.backward()
                    self.opts[model_idx].step()
                    register_vals['train_loss'] += loss.detach().cpu().item() * x.shape[0]
                register_vals['train_loss'] /= len(self.dataset) # 避免最后一个batch导致计算误差
                # validation phase
                self.dataset.set_mode('valid')
                self.models[model_idx].eval()
                with torch.no_grad():
                    for data in self.valid_dataloader:
                        x, mask = data[0].to(self.device), data[1].to(self.device)
                        pred = self.models[model_idx](x, mask)
                        loss = self.criterions[model_idx](pred, x[:,self.target_idx, :], mask)
                        register_vals['valid_loss'] += loss.detach().cpu().item() * x.shape[0]
                register_vals['valid_loss'] /= len(self.dataset)
                tq.set_postfix(valid_loss=register_vals['valid_loss'],
                    train_loss=register_vals['train_loss'])
                tq.update(1)
                if register_vals['valid_loss'] < best_valid_loss:
                    best_valid_loss = register_vals['valid_loss']
                    best_epoch = epoch
                if self.quantile is not None and model_idx == (len(self.quantile)-1) / 2:
                    self.register_vals['train_loss'].append(register_vals['train_loss'])
                    self.register_vals['valid_loss'].append(register_vals['valid_loss'])
                elif self.quantile is None:
                    self.register_vals['train_loss'].append(register_vals['train_loss'])
                    self.register_vals['valid_loss'].append(register_vals['valid_loss'])
                torch.save(self.models[model_idx].state_dict(), os.path.join(cache_path, f'{epoch}.pt'))
        best_path = os.path.join(cache_path, f'{best_epoch}.pt')
        self.models[model_idx].load_state_dict(torch.load(best_path, map_location=self.device))
        logger.info(f'Load best model from {best_path} valid loss={best_valid_loss}')

    def predict(self, mode='test'):
        if self.quantile is not None:
            preds = []
            for idx,q in enumerate(self.quantile):
                logger.info(f'Testing quantile={q}')
                pred = self._predict(idx, mode)
                preds.append(pred)
            return torch.stack(preds, dim=0)
        else:
            return self._predict(0)
    
    def _predict(self, model_idx, mode):
        assert(mode in ['test', 'train', 'valid'])
        self.dataset.set_mode(mode)
        self.models[model_idx] = self.models[model_idx].to(self.device).eval()
        register_vals = {'test_loss':0, 'pred':[], 'gt':[]}
        with tqdm(total=len(self.test_dataloader)) as tq:
            tq.set_description(f'Testing, data={mode}')
            with torch.no_grad():
                for idx, data in enumerate(self.test_dataloader):
                    x, mask = data[0].to(self.device), data[1].to(self.device)
                    pred = self.models[model_idx](x, mask)
                    register_vals['pred'].append(pred.detach().clone().cpu())
                    # register_vals['gt'].append(x[:, self.target_idx, :].detach().clone().cpu())
                    loss = self.criterions[model_idx](pred, x[:,self.target_idx, :], mask)
                    register_vals['test_loss'] += loss.detach().cpu().item()
                    tq.set_postfix(loss=register_vals['test_loss'] / (idx+1))
                    tq.update(1)
        pred = torch.concat(register_vals['pred'], dim=0)[:, :-1]
        pred = torch.concat([-torch.ones(size=(pred.size(0), 1)), pred], dim=1) # 第一列没有预测
        return pred
