import torch.nn as nn
import numpy as np
import torch
import tools
from tools import logger as logger
from torch.utils.data.dataloader import DataLoader
import os
from tqdm import tqdm
import torchinfo


class LSTMPredictor(nn.Module):
    def __init__(self, dev:str, in_channels:int, out_channels:int,  hidden_size=128, out_mean=0, dp=0) -> None:
        super().__init__()
        self.device = torch.device(dev)
        self.norm = nn.BatchNorm1d(num_features=in_channels)
        self.ebd = nn.Linear(in_features=in_channels, out_features=in_channels)
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, batch_first=True, dropout=dp, device=self.device)
        self.den = nn.Linear(in_features=hidden_size, out_features=out_channels, device=self.device)
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
        x = self.den(x)
        # x: (batch, time, out_channels)
        # time=T位置是T+1的预测值
        x = ((x+1) * mask.unsqueeze(-1)) * self.out_mean # 这是为了减少权重变化的幅度
        return torch.permute(x, (2, 0, 1)) # (out_channels, batch, time)

class QuantileLoss(nn.Module):
    def __init__(self, alpha:list, punish:float=None):
        super().__init__()
        self.alpha = alpha
        self.punish = punish
        self.relu = nn.ReLU()

    def forward(self, _pred:torch.Tensor, _gt:torch.Tensor, _mask:torch.Tensor) -> torch.Tensor:
        pred, gt, mask = _pred[:, :, :-1], _gt[:, 1:], _mask[:, 1:] # T=1对齐
        quantile_loss = None
        # calculate quantile loss
        for idx in range(pred.size(0)):
            grt = (pred[idx,...] <= gt)
            res = torch.abs(pred[idx,...] - gt) * mask
            idx_loss =  torch.sum(self.alpha[idx] * res * grt + (1-self.alpha[idx]) * res * torch.logical_not(grt)) / torch.sum(mask)
            quantile_loss = idx_loss if quantile_loss is None else idx_loss + quantile_loss
        if self.punish is not None:
            punish_loss = None
            q_idx = round((pred.size(0)-1)/2)
            for idx in range(1, q_idx+1, 1):
                idx_loss = torch.sum(
                    (self.relu(pred[q_idx-idx,...] - pred[q_idx,...]) + self.relu(pred[q_idx,...] - pred[q_idx+idx,...])) * mask) / torch.sum(mask)
                punish_loss = idx_loss if punish_loss is None else idx_loss + punish_loss
            return quantile_loss / pred.size(0) + self.punish * punish_loss / q_idx
        else:
            return quantile_loss / pred.size(0)

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
        self.quantile = self.params['quantile']
        self.quantile_idx = round(len(self.quantile[:-1]) / 2)
        self.model = LSTMPredictor(params['device'], params['in_channels'], out_channels=len(self.quantile), out_mean=dataset.target_mean)
        self.criterion = QuantileLoss(alpha=self.quantile, punish=params.get('punish'))
        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=params['lr'])
        self.dataset = dataset
        self.target_idx = dataset.target_idx
        self.train_dataloader = DataLoader(dataset=self.dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_fn)
        self.valid_dataloader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
        self.test_dataloader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        self.register_vals = {'train_loss':[], 'valid_loss':[]}
    
    def plot_loss(self, out_path:str):
        data = np.asarray([self.register_vals['train_loss'], self.register_vals['train_loss']])
        tools.plot_loss(data=data, title='Loss for LSTM model', legend=['train', 'valid'], out_path=out_path)
    
    def summary(self):
        torchinfo.summary(self.model)

    def train(self):
        cache_path = os.path.join(self.cache_path)
        tools.reinit_dir(cache_path)
        self.model = self.model.to(self.device)
        with tqdm(total=self.params['epochs']) as tq:
            tq.set_description(f'Training, Epoch')
            best_epoch = 0
            best_valid_loss = np.inf
            for epoch in range(self.params['epochs']):
                register_vals = {'train_loss':0, 'valid_loss':0}
                # train phase
                self.dataset.set_mode('train')
                self.model.train()
                for data in self.train_dataloader:
                    x, mask = data[0].to(self.device), data[1].to(self.device)
                    pred = self.model(x, mask)
                    loss = self.criterion(pred, x[:, self.target_idx, :], mask)
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    register_vals['train_loss'] += loss.detach().cpu().item() * x.shape[0]
                register_vals['train_loss'] /= len(self.dataset) # 避免最后一个batch导致计算误差
                # validation phase
                self.dataset.set_mode('valid')
                self.model.eval()
                with torch.no_grad():
                    for data in self.valid_dataloader:
                        x, mask = data[0].to(self.device), data[1].to(self.device)
                        pred = self.model(x, mask)
                        loss = self.criterion(pred, x[:, self.target_idx, :], mask)
                        register_vals['valid_loss'] += loss.detach().cpu().item() * x.shape[0]
                register_vals['valid_loss'] /= len(self.dataset)
                tq.set_postfix(valid_loss=register_vals['valid_loss'],
                    train_loss=register_vals['train_loss'])
                tq.update(1)
                if register_vals['valid_loss'] < best_valid_loss:
                    best_valid_loss = register_vals['valid_loss']
                    best_epoch = epoch
                self.register_vals['train_loss'].append(register_vals['train_loss'])
                self.register_vals['valid_loss'].append(register_vals['valid_loss'])
                torch.save(self.model.state_dict(), os.path.join(cache_path, f'{epoch}.pt'))
        best_path = os.path.join(cache_path, f'{best_epoch}.pt')
        self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        logger.info(f'Load best model from {best_path} valid loss={best_valid_loss}')
    
    def predict(self, mode):
        assert(mode in ['test', 'train', 'valid'])
        self.dataset.set_mode(mode)
        self.model = self.model.to(self.device).eval()
        register_vals = {'test_loss':0, 'pred':[], 'gt':[]}
        with tqdm(total=len(self.test_dataloader)) as tq:
            tq.set_description(f'Testing, data={mode}')
            with torch.no_grad():
                for idx, data in enumerate(self.test_dataloader):
                    x, mask = data[0].to(self.device), data[1].to(self.device)
                    pred = self.model(x, mask)
                    register_vals['pred'].append(pred.detach().clone().cpu())
                    # register_vals['gt'].append(x[:, self.target_idx, :].detach().clone().cpu())
                    loss = self.criterion(pred, x[:, self.target_idx, :], mask)
                    register_vals['test_loss'] += loss.detach().cpu().item()
                    tq.set_postfix(loss=register_vals['test_loss'] / (idx+1))
                    tq.update(1)
        pred = torch.concat(register_vals['pred'], dim=1)[:, :, :-1]
        pred = torch.concat([-torch.ones(size=(pred.size(0), pred.size(1), 1)), pred], dim=-1) # 第一列没有预测
        if self.quantile is None:
            pred = pred[0,...]
        return pred
