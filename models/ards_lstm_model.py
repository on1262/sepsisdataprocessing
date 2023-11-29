import torch.nn as nn
import numpy as np
import torch
import tools
from tools import logger as logger

class ArdsLSTMModel(nn.Module):
    '''带预测窗口的多分类判别模型'''
    def __init__(self, in_channels:int, n_cls, hidden_size=128) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_cls = n_cls
        self.hidden_size = hidden_size

        self.norm = nn.BatchNorm1d(num_features=in_channels)
        self.den = nn.Linear(in_features=hidden_size, out_features=n_cls)
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, batch_first=True)
    
    def forward(self, x:torch.Tensor):
        # x: (batch, feature, time)
        x = self.norm(x)
        # x: (batch, feature, time)
        x = x.transpose(1,2)
        # x: (batch, time, feature) out带有tanh
        x, _ = self.lstm(x)
        # x: (batch, time, hidden_size)
        x = self.den(x)
        # x: (batch, time, n_cls)
        return x