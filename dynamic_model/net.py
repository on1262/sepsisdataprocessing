import torch.nn as nn
import numpy as np
import torch
import tools


class LSTMPredictor(nn.Module):
    def __init__(self, dev:str, in_channels:int, hidden_size=128, dp=0) -> None:
        self.device = torch.device(dev)
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, batch_first=True, dropout=dp, device=self.device)
        self.dense = nn.Linear(in_features=hidden_size, out_features=1, device=self.device)
        self.c_0 = nn.Parameter(torch.zeros((1, hidden_size), device=self.device), requires_grad=True)
        self.h_0 = nn.Parameter(torch.zeros((1, hidden_size), device=self.device), requires_grad=True)

    def forward(self, x:torch.Tensor):
        x, _ = self.lstm(x, (self.h_0, self.c_0))
        x = self.dense(x)
        return x

class QuantileLoss(nn.Module):
    def __init__(self, alpha:float):
        self.alpha = alpha

    def forward(self, pred:torch.Tensor, gt:torch.Tensor):
        mask = (pred <= gt)
        res = torch.abs(pred - gt)
        return torch.mean(self.alpha * res * mask + (1-self.alpha) * res * (1-mask))

class Dataset():
    def __init__(self, params:dict, data:np.ndarray, mask:np.ndarray, train_index, valid_index, test_index) -> None:
        self.mode = 'train' # train/valid/test
        self.train_index = train_index
        self.valid_index = valid_index
        self.test_index = test_index
        # TODO TS和插值在此进行
        self.data = data
        self.device = params['device']
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

def collate_fn():
    pass

class Trainer():
    def __init__(self, params:dict, dataset:Dataset) -> None:
        self.params = params
        self.model = LSTMPredictor(params['device'], params['in_channels'])
        self.criterion = QuantileLoss(alpha=params['alpha'])
        self.opt = torch.optim.Adam(params=self.model.parameters, lr=params['lr'])
        self.dataset = dataset
    
    def train(self):
        # for epoch in self.params['epochs']:
        #     # train phase


    
    

    