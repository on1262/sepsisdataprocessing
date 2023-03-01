import torch
import torch.nn as nn
import numpy as np
import tools
import torchinfo
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import os
from tools import logger as logger


def Collect_Fn(data_list:list):
    result = {}
    result['data'] = torch.as_tensor(np.stack([d['data'] for d in data_list], axis=0))
    result['length'] = torch.as_tensor([d['length'] for d in data_list], dtype=torch.long)
    return result


class LSTMTrainer():
    def __init__(self, params:dict, dataset) -> None:
        self.params = params
        self.device = torch.device(self.params['device'])
        self.cache_path = params['cache_path']
        tools.reinit_dir(self.cache_path, build=True)
        # self.quantile = None
        # self.quantile = self.params['quantile']
        # self.quantile_idx = round(len(self.quantile[:-1]) / 2)
        self.model = LSTMModel(n_fea=params['in_channels'], n_hidden=params['hidden_size'])
        self.criterion = AutoRegLoss(target_idx=params['target_idx'], target_coeff=params['target_coeff'])
        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=params['lr'])
        self.dataset = dataset
        self.target_idx = params['target_idx']
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
    
    def _make_mask(self, m_shape:tuple, available_lens) -> np.ndarray:
        mask = np.zeros(m_shape, dtype=bool)
        for idx in range(m_shape[0]):
            mask[idx, :available_lens[idx]] = True
        return mask


    def summary(self):
        torchinfo.summary(self.model)

    def _train_forward(self, batch):
        x, available_lens = batch['data'].to(self.device), batch['length']
        pred = self.model(x, available_lens, available_lens)
        loss = self.criterion(pred=pred[:, :-1], gt=x[:, self.target_idx, 1:], mask=self._make_mask(x.size(), available_lens)[:, 1:])
        return pred, loss

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
                self.dataset.mode('train')
                self.model.train()
                for batch in self.train_dataloader:
                    _, loss = self._train_forward(self, batch)
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    loss_vals['train_loss'] += loss.detach().cpu().item() * x.shape[0]
                loss_vals['train_loss'] /= len(self.dataset) # 避免最后一个batch导致计算误差
                # validation phase
                self.dataset.mode('valid')
                self.model.eval()
                with torch.no_grad():
                    for batch in self.valid_dataloader:
                        _, loss = self._train_forward(self, batch)
                        loss_vals['valid_loss'] += loss.detach().cpu().item() * x.shape[0]
                loss_vals['valid_loss'] /= len(self.dataset)
                # tqdm
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
        self.dataset.mode(mode)
        self.model = self.model.to(self.device).eval()
        register_vals = {'test_loss':0, 'pred':[]}
        with tqdm(total=len(self.test_dataloader)) as tq:
            tq.set_description(f'Testing, data={mode}')
            with torch.no_grad():
                for idx, batch in enumerate(self.test_dataloader):
                    pred, loss = self._train_forward(self, batch)
                    register_vals['pred'].append(pred.detach().clone().cpu())
                    register_vals['test_loss'] += loss.detach().cpu().item()
                    tq.set_postfix(loss=register_vals['test_loss'] / (idx+1))
                    tq.update(1)
        pred = torch.concat(register_vals['pred'], dim=1)[:, :, :-1]
        pred = torch.concat([-torch.ones(size=(pred.size(0), pred.size(1), 1)), pred], dim=-1) # 第一列没有预测
        return pred


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
        用于自回归的loss, pred和gt位置对齐
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
        



