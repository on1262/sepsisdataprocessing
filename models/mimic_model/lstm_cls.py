import torch.nn as nn
import numpy as np
import torch
import tools
from tools import logger as logger
from torch.utils.data.dataloader import DataLoader
import os
from tqdm import tqdm
import torchinfo


class LSTMClsModel(nn.Module):
    '''带预测窗口的多分类判别模型'''
    def __init__(self, dev:str, in_channels:int, n_cls=4, hidden_size=128, dp=0) -> None:
        super().__init__()
        self.device = torch.device(dev)
        self.norm = nn.BatchNorm1d(num_features=in_channels)
        self.ebd = nn.Linear(in_features=in_channels, out_features=in_channels)
        self.den = nn.Linear(in_features=hidden_size, out_features=n_cls)
        self.sf = nn.Softmax(dim=-1)
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, batch_first=True, dropout=dp, device=self.device)
        self.c_0 = nn.Parameter(torch.zeros((1, 1, hidden_size), device=self.device), requires_grad=True)
        self.h_0 = nn.Parameter(torch.zeros((1, 1, hidden_size), device=self.device), requires_grad=True)

    def forward(self, x:torch.Tensor):
        '''
        给出某个时间点对未来一个窗口内是否发生ARDS的概率
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
        x = self.sf(self.den(x))
        return x # (batch, time, n_cls)

def Collect_Fn(data_list:list):
    result = {}
    result['data'] = torch.as_tensor(np.stack([d['data'] for d in data_list], axis=0), dtype=torch.float32)
    result['length'] = np.asarray([d['length'] for d in data_list], dtype=np.int32)
    return result


class LSTMClsTrainer():
    def __init__(self, params:dict, dataset) -> None:
        self.params = params
        self.paths = params['paths']
        self.device = torch.device(self.params['device'])
        self.cache_path = self.paths['lstm_cls_cache']
        tools.reinit_dir(self.cache_path, build=True)
        self.model = LSTMClsModel(params['device'], params['in_channels'])
        self.criterion = ClassificationLoss(len(self.params['centers']))
        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=params['lr'])
        self.dataset = dataset
        self.target_idx = dataset.target_idx
        self.generator = Cls4LabelGenerator(window=self.params['window'], centers=self.params['centers'], smoothing_band=self.params['smoothing_band'])
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
        np_data = np.asarray(data['data'])
        seq_lens = data['length']
        mask = tools.make_mask((np_data.shape[0], np_data.shape[2]), seq_lens)
        mask[:, -1] = False
        labels = torch.as_tensor(self.generator(np_data[:, self.target_idx, :], mask), device=self.device)
        x = data['data'].to(self.device)
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


class Cls4LabelGenerator():
    '''从data: (batch, seq_lens)生成每个时间点在预测窗口内的ARDS标签'''
    def __init__(self, window=16, centers=list(), smoothing_band=50) -> None:
        assert(len(centers) == 4)
        self.window = window # 向前预测多少个点内的ARDS
        self.centers = centers # 判断ARDS的界限
        self.band = smoothing_band # 平滑程度, 不能超过两个center之间的距离

    
    def __call__(self, data:np.ndarray, mask:np.ndarray) -> np.ndarray:
        '''
        data不能正则化
        mask, data: (batch, seq_lens)
        return: (batch, seq_lens, n_cls)
        '''
        assert(len(data.shape) == 2 and len(mask.shape) == 2)
        result = np.zeros(data.shape + (len(self.centers),), dtype=np.float32)
        data_max = data.max()
        for idx in range(data.shape[1]-1): # 最后一个格子预测一格
            stop = min(data.shape[1], idx+self.window)
            mat = mask[:, idx+1:stop] * data[:, idx+1:stop] + (1-mask[:, idx+1:stop]) * data_max # 对于有效的result, 至少有一个格子是有效的
            mat_min = np.min(mat, axis=1) # (batch,)
            result[:, idx, :] = tools.label_smoothing(self.centers, mat_min, band=50)
        return (result * mask[..., None]).astype(np.int32)
  

class ClassificationLoss(nn.Module):
    def __init__(self, n_cls:int) -> None:
        '''
        forecast window: 向前预测的窗口
        '''
        super().__init__()
        self.n_cls = n_cls
        self.criterion = nn.CrossEntropyLoss(reduction='none') # input: (N,C,d1,...dk)

    def forward(self, pred:torch.Tensor, labels:torch.Tensor, mask:torch.Tensor):
        '''
            pred, labels: (batch, seq_len, n_cls)
            mask: (batch, seq_len)
            labels可以是软标签
        '''
        assert(pred.size() == labels.size())
        if len(mask.size()) + 1 == len(pred.size()):
            mask = mask[..., None]
        pred, labels = pred*mask, labels*mask
        # 创建标签矩阵
        loss = torch.sum(self.criterion(pred.permute(0, 2, 1), labels.permute(0, 2, 1))) / torch.sum(mask)
        return loss