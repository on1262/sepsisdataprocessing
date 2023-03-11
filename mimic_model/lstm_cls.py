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
    def __init__(self, dev:str, in_channels:int, hidden_size=128, out_mean=0, dp=0) -> None:
        super().__init__()
        self.device = torch.device(dev)
        self.norm = nn.BatchNorm1d(num_features=in_channels)
        self.ebd = nn.Linear(in_features=in_channels, out_features=in_channels)
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, batch_first=True, dropout=dp, device=self.device)
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
        x = (x + 1) / 2
        return torch.permute(x, (2, 0, 1)) # (out_channels, batch, time)


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

def Collect_Fn(data_list:list):
    result = {}
    result['data'] = torch.as_tensor(np.stack([d['data'] for d in data_list], axis=0), dtype=torch.float32)
    result['length'] = np.asarray([d['length'] for d in data_list], dtype=np.int32)
    return result

class Trainer():
    def __init__(self, params:dict, dataset:Dataset) -> None:
        self.params = params
        self.device = torch.device(self.params['device'])
        self.cache_path = params['cache_path']
        tools.reinit_dir(self.cache_path, build=True)
        self.model = LSTMPredictor(params['device'], params['in_channels'], out_mean=dataset.target_mean)
        self.criterion = None
        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=params['lr'])
        self.dataset = dataset
        self.target_idx = dataset.target_idx
        self.train_dataloader = DataLoader(dataset=self.dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=Collect_Fn)
        self.valid_dataloader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=True, collate_fn=Collect_Fn)
        self.test_dataloader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=False, collate_fn=Collect_Fn)

        self.register_vals = {'train_loss':[], 'valid_loss':[]}
    
    def get_loss(self):
        data = {
            'train': np.asarray(self.register_vals['train_loss']), 
            'valid': np.asarray(self.register_vals['valid_loss'])
        }
        return data
    
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
                loss_vals = {'train_loss':0, 'valid_loss':0}
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
                    loss_vals['train_loss'] += loss.detach().cpu().item() * x.shape[0]
                loss_vals['train_loss'] /= len(self.dataset) # 避免最后一个batch导致计算误差
                # validation phase
                self.dataset.set_mode('valid')
                self.model.eval()
                with torch.no_grad():
                    for data in self.valid_dataloader:
                        x, mask = data[0].to(self.device), data[1].to(self.device)
                        pred = self.model(x, mask)
                        loss = self.criterion(pred, x[:, self.target_idx, :], mask)
                        loss_vals['valid_loss'] += loss.detach().cpu().item() * x.shape[0]
                loss_vals['valid_loss'] /= len(self.dataset)
                tq.set_postfix(valid_loss=loss_vals['valid_loss'],
                    train_loss=loss_vals['train_loss'])
                tq.update(1)
                if loss_vals['valid_loss'] < best_valid_loss:
                    best_valid_loss = loss_vals['valid_loss']
                    best_epoch = epoch
                self.register_vals['train_loss'].append(loss_vals['train_loss'])
                self.register_vals['valid_loss'].append(loss_vals['valid_loss'])
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
        return pred
