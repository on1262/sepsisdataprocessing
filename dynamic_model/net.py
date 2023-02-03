import torch.nn as nn
import numpy as np
import torch
import tools
from tools import logger as logger
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


class LSTMPredictor(nn.Module):
    def __init__(self, dev:str, in_channels:int, hidden_size=128, dp=0) -> None:
        self.device = torch.device(dev)
        self.norm = nn.BatchNorm1d(num_features=in_channels)
        self.ebd = nn.Embedding(num_embeddings=in_channels, embedding_dim=in_channels)
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, batch_first=True, dropout=dp, device=self.device)
        self.den = nn.Linear(in_features=hidden_size, out_features=1, device=self.device)
        self.c_0 = nn.Parameter(torch.zeros((1, hidden_size), device=self.device), requires_grad=True)
        self.h_0 = nn.Parameter(torch.zeros((1, hidden_size), device=self.device), requires_grad=True)

    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        x = self.ebd(self.norm(x))
        x, _ = self.lstm(x, (self.h_0, self.c_0))
        x = self.den(x)
        return x * mask

class QuantileLoss(nn.Module):
    def __init__(self, alpha:float):
        self.alpha = alpha

    def forward(self, pred:torch.Tensor, gt:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        grt = (pred <= gt)
        res = torch.abs(pred - gt) * mask
        return torch.mean(self.alpha * res * grt + (1-self.alpha) * res * (1-grt))

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
        self.mask = mask

    def mode(self, mode:str):
        self.mode = mode
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.data[self.train_index[idx], :,:], self.mask[self.train_index[idx], :]
        elif self.mode == 'valid':
            return self.data[self.valid_index[idx], :,:], self.mask[self.train_index[idx], :]
        else:
            return self.data[self.test_index[idx], :,:], self.mask[self.train_index[idx], :]

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
        self.quantile = None
        if 'quantile' in self.params.keys():
            self.quantile = self.params['quantile']
            self.models = [LSTMPredictor(params['device'], params['in_channels']) for _ in range(len(self.quantile))]
            self.criterions = [QuantileLoss(alpha) for alpha in self.quantile]
            self.opts = [torch.optim.Adam(params=self.models[idx].parameters(), lr=params['lr']) for idx in range(len(self.quantile))]
        else:
            self.models = [LSTMPredictor(params['device'], params['in_channels'])]
            self.criterions = [QuantileLoss(alpha=params['alpha'])]
            self.opts = [torch.optim.Adam(params=self.model.parameters(), lr=params['lr'])]
        self.dataset = dataset
        self.train_dataloader = DataLoader(dataset=self.dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_fn)
        self.valid_dataloader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
        self.test_dataloader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    def train(self):
        if self.quantile is not None:
            for idx,q in enumerate(self.quantile):
                logger.info('Training quantile={q}')
                self._train(self, idx)
        else:
            self._train(self, 0)
    
    def _train(self, idx):
        self.models[idx] = self.models[idx].to(self.device)
        for epoch in range(self.params['epochs']):
            register_vals = {'train_loss':0, 'valid_loss':0}
            # train phase
            self.dataset.mode('train')
            with tqdm(total=len(self.train_dataloader)) as tq:
                tq.set_description(f'Training, Epoch={epoch}')
                for idx, data in enumerate(self.train_dataloader):
                    x, mask = data[0].to(self.device), data[1].to(self.device)
                    pred = self.models[idx](x, mask)
                    loss = self.criterions[idx](pred, x, mask)

                    self.opts[idx].zero_grad()
                    loss.backward()
                    self.opts[idx].step()
                    register_vals['train_loss'] += loss
                    tq.set_postfix(loss=register_vals['train_loss'] / (idx+1))
                    tq.update(1)
            # validation phase
            self.dataset.mode('valid')
            with tqdm(total=len(self.valid_dataloader)) as tq:
                tq.set_description(f'Validation, Epoch={epoch}')
                for idx, data in enumerate(self.valid_dataloader):
                    x, mask = data[0].to(self.device), data[1].to(self.device)
                    pred = self.models[idx](x, mask)
                    loss = self.criterions[idx](pred, x, mask)
                    register_vals['valid_loss'] += loss
                    tq.set_postfix(loss=register_vals['valid_loss'] / (idx+1))
                    tq.update(1)

    def test(self):
        if self.quantile is not None:
            preds = []
            for idx,q in enumerate(self.quantile):
                logger.info('Testing quantile={q}')
                preds.append(self._test(self, idx))
                return torch.stack(preds, dim=0)
        else:
            return self._test(self, 0)
    
    def _test(self, idx):
        self.dataset.mode('test')
        register_vals = {'test_loss':0, 'pred':[]}
        with tqdm(total=len(self.test_dataloader)) as tq:
            tq.set_description(f'Testing')
            for idx, data in enumerate(self.test_dataloader):
                x, mask = data[0].to(self.device), data[1].to(self.device)
                pred = self.models[idx](x, mask)
                register_vals['pred'].append(pred.detach().clone().to('cpu'))
                loss = self.criterion(pred, x, mask)
                register_vals['test_loss'] += loss
                tq.set_postfix(loss=register_vals['valid_loss'] / (idx+1))
                tq.update(1)
        return torch.stack(register_vals['pred'], dim=0)
