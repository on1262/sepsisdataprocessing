import torch
import torch.nn as nn
import numpy as np

class LSTMModel(nn.Module):
    '''用于MIMIC-IV的时序预测, 采用自回归的方式, 会同时预测所有feature的结果'''
    def __init__(self, n_fea, n_hidden=128):
        self.n_hidden = n_hidden
        self.n_fea = n_fea
        self.cell = nn.LSTMCell(input_size=n_fea, hidden_size=n_hidden) # input: (batch, n_fea)
        self.den = nn.Linear(in_features=n_hidden, out_features=n_fea)
        self.init_ch = nn.parameter.Parameter(data=torch.zeros(size=(1, 2*n_hidden)), requires_grad=True)

    def auto_reg(self, pred_point:int, x:torch.Tensor, start_len:torch.Tensor):
        '''
            pred_point: 自回归向前预测多少点
            x: (batch, n_fea, seq_len) 已知数据
            start_len: 约定开始预测的点, 例如已知两个点(0,1), 则start_len[idx]=2
                注意pred_point + max(seqs_len) <= x.shape[-1] 需要成立, 否则不会开辟新空间
        '''
        assert(pred_point + torch.max(start_len) <= x.shape[-1])
        with torch.no_grad():
            return self.forward(x=x, available_len=start_len, predict_len=start_len+pred_point)


    def forward(self, x:torch.Tensor, available_len:np.ndarray, predict_len:np.ndarray=None):
        '''
            x: (batch, n_fea, seq_len)
            available_len: (batch,) 每行可用数据的时间点个数
            predict_len: (batch,) 每行需要预测的时间点个数+可用数据的时间点个数
                None: 和available_len相同, 默认每个点给一个预测值
                其他情况: 比如某行取值为2, 则基于第0,1个元素给出预测, 预测目标是第1,2个时间点的值. 
                如果大于available_len的对应行, 那么进行auto_regression预测
            输出和输入的shape相同
        '''
        if predict_len is None:
            predict_len = available_len.copy()
        x = x.permute(2, 0, 1) # ->(seq_len, batch, n_fea)
        c,h = None, None
        for idx in range(x.size(0)):
            select = idx < predict_len
            if idx == 0:
                c,h = self.init_ch[:, :self.n_hidden].expand(sum(select), 1), self.init_ch[:, self.n_hidden:].expand(sum(select), 1)
            else:
                c,h = c_1[predict_len[predict_len > idx-1][select]], h_1[predict_len[predict_len > idx-1][select]]
            (h_1, c_1) = self.cell(x[idx, select, :], (h,c))
            x[idx, select, :] = self.den(h_1) # h_1: (select, hidden_size)
            auto_reg = (idx+1 >= available_len) * (idx+1 < predict_len) # 下一轮会被select但是以及在available_len之外
            if np.any(auto_reg) and idx+1 < x.size(0):
                x[idx+1, auto_reg, :] = x[idx, auto_reg, :]
        return x

class AutoRegLoss(nn.Module):
    '''
        用于自回归的loss, 需要对model输出错开一个时间点对齐
        target_idx: target(PaO2/FiO2)所在的feature序号
        target_coeff: 对target进行loss加强的系数
    '''
    def __init__(self, target_idx, target_coeff=10) -> None:
        self.t_idx = target_idx
        self.t_coeff = target_coeff

    def _cal_loss(self, pred, gt, mask):
        return torch.sqrt(torch.sum(torch.square(pred - gt) * mask) / torch.sum(mask))

    def forward(self, pred:torch.Tensor, gt:torch.Tensor, mask:torch.Tensor):
        '''
            pred, gt, mask: (batch, n_fea, seq_len)
            return:
                all_loss, target_loss_item
        '''
        pred = pred[:,:, :-1]
        gt = gt[:,:,1:]
        mask = mask[:, :,:-1]
        target_loss = self._cal_loss(self, pred[:, self.t_idx, :], gt[:, self.t_idx, :], mask[:, self.t_idx, :])
        all_loss = self._cal_loss(self, pred, gt, mask)
        return self.t_coeff * target_loss + all_loss, target_loss.detach().clone().item()
        



