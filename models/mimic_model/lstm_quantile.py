import torch.nn as nn
import numpy as np
import torch
import tools
from tools import logger as logger
from torch.utils.data.dataloader import DataLoader
from .lstm_reg import RegLabelGenerator
import os
from tqdm import tqdm
import torchinfo
import emd

class LSTMQuantileModel(nn.Module):
    '''带分位点的回归模型'''
    def __init__(self, dev:str, in_channels:int, n_quantile:int, target_norm:dict, hidden_size=128, dp=0, emd_params=None) -> None:
        '''
        target_norm: {'mean':mean, 'std':std} 用于输出的归一化还原
        emd_params: {'target_idx':idx, 'max_imfs':int} 设置emd, 如果禁用则为None
            target_idx: 需要emd的曲线的idx
            max_imfs: 一条曲线将拆为多条曲线
        '''
        super().__init__()
        # check emd settings
        if emd_params is not None:
            in_channels = in_channels - 1 + emd_params['max_imfs']
        self.device = torch.device(dev)
        self.norm = nn.BatchNorm1d(num_features=in_channels)
        self.ebd = nn.Linear(in_features=in_channels, out_features=in_channels)
        self.den = nn.Linear(in_features=hidden_size, out_features=n_quantile)
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, batch_first=True, dropout=dp, device=self.device)
        self.c_0 = nn.Parameter(torch.zeros((1, 1, hidden_size), device=self.device), requires_grad=True)
        self.h_0 = nn.Parameter(torch.zeros((1, 1, hidden_size), device=self.device), requires_grad=True)
        self.target_mean = target_norm['mean']
        self.target_std = target_norm['std']

    def forward(self, x:torch.Tensor):
        '''
        x: (batch, feature, time)
        mask: (batch, time)
        '''
        # mask: (batch, time)
        x = self.norm(x)
        # x: (batch, feature, time)
        x = self.ebd(x.transpose(1,2))
        # x: (batch, time, feature) out带有tanh
        x, _ = self.lstm(x, (self.h_0.expand(-1, x.size(0), -1).contiguous(), self.c_0.expand(-1, x.size(0), -1).contiguous()))
        # x: (batch, time, hidden_size)
        x = self.den(x)
        # x: (batch, time, quantile)
        return self.target_mean + x * self.target_std 

def Collect_Fn(data_list:list):
    result = {}
    result['data'] = torch.as_tensor(np.stack([d['data'] for d in data_list], axis=0), dtype=torch.float32)
    result['length'] = np.asarray([d['length'] for d in data_list], dtype=np.int32)
    return result


class LSTMQuantileTrainer():
    def __init__(self, params:dict, dataset) -> None:
        self.params = params
        self.paths = params['paths']
        self.device = torch.device(self.params['device'])
        self.cache_path = self.paths['lstm_quantile_cache']
        tools.reinit_dir(self.cache_path, build=True)
        if params['enable_emd'] == True:
            self.emd_params = {'target_idx': dataset.target_idx, 'max_imfs':self.params['max_imfs']}
        else:
            self.emd_params = None
        self.model = LSTMQuantileModel(
            params['device'], params['in_channels'], n_quantile=len(params['quantile_taus']), target_norm=dataset.norm_dict[dataset.target_name],
            hidden_size=params['hidden_size'], emd_params=self.emd_params)
        self.criterion = QuantileRegressionLoss(taus=params['quantile_taus'], punish_coeff=params['punish_coeff'])
        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=params['lr'])
        self.dataset = dataset
        self.target_idx = dataset.target_idx
        self.generator = RegLabelGenerator(window=self.params['window']) # 生成标签
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

    def _batch_forward(self, data):
        np_data = np.asarray(data['data']) # (batch, n_fea, seq_len)
        seq_lens = data['length']
        # processing
        if self.emd_params is not None:
            emd_in = np_data[:, self.emd_params['target_idx'], :] # (batch, seq_len)
            emd_out = np.zeros((emd_in.shape[0], self.emd_params['max_imfs'], emd_in.shape[1]))
            for idx in range(emd_in.shape[0]):
                out = emd.sift.sift(emd_in[idx, :],  max_imfs=self.emd_params['max_imfs'])
                if out.shape[-1] < self.emd_params['max_imfs']: # 不足则填充0
                    out = np.concatenate([out, np.zeros((out.shape[0], self.emd_params['max_imfs']-out.shape[-1]))], axis=-1)
                emd_out[idx, :, :] = out.T
            np_data = np.concatenate([np_data[:, :-1, :], emd_out], axis=1)
        mask = tools.make_mask((np_data.shape[0], np_data.shape[2]), seq_lens)
        mask[:, -1] = False
        mask, labels = self.generator(np_data, mask)
        mask, labels = torch.as_tensor(mask, device=self.device), torch.as_tensor(labels, device=self.device)
        x = torch.as_tensor(np_data, device=self.device, dtype=torch.float32)
        pred = self.model(x)
        loss = self.criterion(pred, labels, torch.as_tensor(mask, device=self.device))
        return pred, loss

    def train(self):
        cache_path = os.path.join(self.cache_path)
        tools.reinit_dir(cache_path, build=True)
        self.model = self.model.to(self.device)
        
        with tqdm(total=self.params['epochs']) as tq:
            tq.set_description(f'Training, Epoch')
            best_epoch = 0
            best_valid_loss = np.inf
            for epoch in range(self.params['epochs']):
                loss_vals = {'train_loss':0, 'valid_loss':0}
                # train phase
                self.dataset.mode('train')
                self.model.train()
                for data in self.train_dataloader:
                    pred, loss = self._batch_forward(data)
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    loss_vals['train_loss'] += loss.detach().cpu().item() * pred.size(0)
                loss_vals['train_loss'] /= len(self.dataset) # 避免最后一个batch导致计算误差
                # validation phase
                self.dataset.mode('valid')
                self.model.eval()
                with torch.no_grad():
                    for data in self.valid_dataloader:
                        pred, loss = self._batch_forward(data)
                        loss_vals['valid_loss'] += loss.detach().cpu().item() * pred.size(0)
                loss_vals['valid_loss'] /= len(self.dataset)
                tq.set_postfix(valid_loss=loss_vals['valid_loss'], train_loss=loss_vals['train_loss'])
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
        self.dataset.mode(mode)
        self.model = self.model.to(self.device).eval()
        register_vals = {'test_loss':0, 'pred':[], 'gt':[]}
        with tqdm(total=len(self.test_dataloader)) as tq:
            tq.set_description(f'Testing, data={mode}')
            with torch.no_grad():
                for idx, data in enumerate(self.test_dataloader):
                    pred, loss = self._batch_forward(data)
                    register_vals['pred'].append(pred.detach().clone().cpu())
                    register_vals['test_loss'] += loss.detach().cpu().item()
                    tq.set_postfix(loss=register_vals['test_loss'] / (idx+1))
                    tq.update(1)
        pred = torch.concat(register_vals['pred'], dim=0)
        return pred



class QuantileRegressionLoss(nn.Module):
    def __init__(self, taus:list, punish_coeff:float):
        super().__init__()
        self.taus = taus
        self.n_quantile = len(taus)
        self.punish_coeff = punish_coeff
        self.relu = nn.ReLU()

    def forward(self, pred:torch.Tensor, gt:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        '''
        pred: (batch, seq_len, n_quantile)
        gt: (batch, seq_len)
        mask: (batch, seq_len)
        '''
        assert(mask.size() == gt.size() and mask.size() == pred.size()[:-1] and pred.size(-1) == self.n_quantile)
        quantile_loss = None
        # calculate quantile loss
        for idx in range(pred.size(-1)):
            grt = (pred[...,idx] <= gt)
            res = torch.abs(pred[...,idx] - gt) * mask
            idx_loss =  torch.sum(self.taus[idx] * res * grt + (1-self.taus[idx]) * res * torch.logical_not(grt)) / torch.sum(mask)
            quantile_loss = idx_loss if quantile_loss is None else idx_loss + quantile_loss
        if self.punish_coeff > 0:
            punish_loss = None
            q_idx = round((pred.size(-1)-1)/2)
            for idx in range(1, q_idx+1, 1):
                idx_loss = torch.sum(
                    (self.relu(pred[...,q_idx-idx] - pred[...,q_idx]) + self.relu(pred[...,q_idx] - pred[...,q_idx+idx])) * mask) / torch.sum(mask)
                punish_loss = idx_loss if punish_loss is None else idx_loss + punish_loss
            return quantile_loss / pred.size(-1) + self.punish_coeff * punish_loss / q_idx
        else:
            return quantile_loss / pred.size(-1)
