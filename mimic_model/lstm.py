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
    result['data'] = torch.as_tensor(np.stack([d['data'] for d in data_list], axis=0), dtype=torch.float32)
    result['length'] = np.asarray([d['length'] for d in data_list], dtype=np.int32)
    return result


class LSTMTrainer():
    def __init__(self, params:dict, dataset) -> None:
        self.params = params
        self.device = torch.device(self.params['device'])
        self.cache_path = params['cache_path']
        tools.reinit_dir(self.cache_path, build=True)
        self.target_idx = params['target_idx']
        self.target_std = params['target_std'] # 用于还原真实的std
        self.model = LSTMModel(n_fea=params['in_channels'], n_hidden=params['hidden_size'])
        self.criterion = AutoRegLoss(target_idx=params['target_idx'], out_std=self.target_std, target_coeff=params['target_coeff'])
        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=params['lr'])
        self.dataset = dataset
        
        
        self.train_dataloader = DataLoader(dataset=self.dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=Collect_Fn)
        self.valid_dataloader = DataLoader(dataset=self.dataset, batch_size=256, shuffle=True, collate_fn=Collect_Fn)
        self.test_dataloader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=False, collate_fn=Collect_Fn)
        self.register_vals = {'train_loss':[], 'valid_loss':[]}
    
    def get_loss(self):
        data = {
            'train': np.asarray(self.register_vals['train_loss']), 
            'valid': np.asarray(self.register_vals['valid_loss'])
        }
        return data
    
    def _make_mask(self, m_shape:tuple, start_lens=None, stop_lens=None) -> np.ndarray:
        '''
            mask: (batch, n_fea, seq_lens)
        '''
        mask = np.zeros(m_shape, dtype=bool)
        if start_lens is None:
            start_lens = np.zeros((m_shape[0],), dtype=np.int32)
        for idx in range(m_shape[0]):
            mask[idx, :, start_lens[idx]:stop_lens[idx]] = True
        return torch.as_tensor(mask, dtype=bool, device=self.device)


    def summary(self):
        torchinfo.summary(self.model)

    def _train_forward(self, batch:dict):
        '''
        pred: (batch, n_fea, seq_lens)
        '''
        x, available_lens = batch['data'].to(self.device), batch['length']
        
        pred = self.model(x, available_lens, available_lens, auto_reg_flag=False)
        loss_tensor, loss_tar_num = self.criterion(pred=pred, gt=x, mask=self._make_mask(x.size(), None, available_lens))
        return pred, loss_tensor, loss_tar_num

    def _test_forward(self, batch:dict, start_points:list, pred_point=16):
        '''
        对每个start_point, 自回归预测之后长度为pred_point的区间, 并停止预测
        比如start=2, pred=3, 则利用[0,1]预测[2,3,4]
        return:
            preds: (len(start_points), batch, seq_lens) 第一维代表每种不同start的情况
            losses: list(loss), 不同start_point的loss
        '''
        preds, losses = [], []
        for sp in start_points:
            x, available_lens = batch['data'].to(self.device), batch['length']
            start_lens = sp*np.ones(available_lens.shape, dtype=np.int32)
            pred = self.model.auto_reg(x, start_lens=start_lens, pred_point=pred_point)
            loss = self.criterion(pred=pred, gt=x, mask=self._make_mask(x.size(), start_lens, start_lens+pred_point))
            preds.append(pred[:, self.target_idx, :])
            losses.append(loss)
        return torch.stack(preds, dim=0), losses


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
                    _, loss, loss_num = self._train_forward(batch)
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    loss_vals['train_loss'] += loss_num * batch['data'].size(0)
                loss_vals['train_loss'] /= len(self.dataset) # 避免最后一个batch导致计算误差
                # validation phase
                self.dataset.mode('valid')
                self.model.eval()
                with torch.no_grad():
                    for batch in self.valid_dataloader:
                        _, _, loss_num = self._train_forward(batch)
                        loss_vals['valid_loss'] += loss_num * batch['data'].size(0)
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
    
    def predict(self, mode, start_points:list, pred_point=16):
        '''
        给出[start_point+1, start_point+pred_point]的预测值
        pred: (points, batch, pred_point) 和gt对齐
        '''
        assert(mode in ['test', 'train', 'valid'])
        self.dataset.mode(mode)
        self.model = self.model.to(self.device).eval()
        register_vals = {'gt':[], 'pred':[]}
        with tqdm(total=len(self.test_dataloader)) as tq:
            tq.set_description(f'Testing, data={mode}')
            with torch.no_grad():
                for _, batch in enumerate(self.test_dataloader):
                    preds, _ = self._test_forward(batch, start_points=start_points, pred_point=pred_point)
                    # 这里按照每个点的时间排序
                    register_vals['pred'].append(preds.detach().clone().cpu())
                    # register_vals['test_loss'].append([loss.detach().cpu().item() for loss in losses])
                    # tq.set_postfix(loss=register_vals['test_loss'] / (idx+1))
                    tq.update(1)
        pred = torch.concat(register_vals['pred'], dim=1)
        result = torch.empty((len(start_points), pred.size(1), pred_point), device=pred.device)
        for idx, s_idx in enumerate(start_points):
            result[idx, :, :] = pred[idx, :, s_idx:s_idx+pred_point] # 和gt对齐
        return result


class LSTMModel(nn.Module):
    '''用于MIMIC-IV的时序预测, 采用自回归的方式, 会同时预测所有feature的结果'''
    def __init__(self, n_fea, n_hidden=128):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_fea = n_fea
        self.cell = nn.LSTMCell(input_size=n_fea, hidden_size=n_hidden) # input: (batch, n_fea)
        self.den = nn.Linear(in_features=n_hidden, out_features=n_fea)
        self.init_ch = nn.parameter.Parameter(data=torch.zeros(size=(1, 2*n_hidden), dtype=torch.float32), requires_grad=True)

    def auto_reg(self, x:torch.Tensor, start_lens:np.ndarray, pred_point:int):
        '''
            pred_point: 自回归向前预测多少点
            x: (batch, n_fea, seq_len) 已知数据
            start_len: 约定开始预测的点, 例如已知两个点(0,1), 则start_len[idx]=2
                注意pred_point + max(seqs_len) <= x.shape[-1] 需要成立, 否则不会开辟新空间
        '''
        assert(pred_point + np.max(start_lens) <= x.shape[-1])
        with torch.no_grad():
            return self.forward(x=x, available_len=start_lens, predict_len=start_lens+pred_point)


    def forward(self, x:torch.Tensor, available_len:np.ndarray, predict_len:np.ndarray=None, auto_reg_flag=True):
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
        out = torch.zeros(size=x.size(), dtype=torch.float32, device=x.device)
        c,h = self.init_ch[:, :self.n_hidden].expand(x.shape[1], -1), self.init_ch[:, self.n_hidden:].expand(x.shape[1], -1)
        max_idx = min(predict_len.max(), x.size(0))
        for idx in range(max_idx):
            (h, c) = self.cell(x[idx, :, :], (h,c))
            out[idx, :, :] = self.den(h) # h_1: (batch, hidden_size)
            if auto_reg_flag:
                reg_mat = (idx+1 >= available_len) * (idx+1 < predict_len) # 下一轮会被select但是以及在available_len之外
                if np.any(reg_mat) and idx+1 < x.size(0):
                    assert(auto_reg_flag)
                    x[idx+1, reg_mat, :] = out[idx, reg_mat, :].detach()
        return out.permute(1, 2, 0) # ->(batch, n_fea, seq_len) 存在时间上的错位

class AutoRegLoss(nn.Module):
    '''
        用于自回归的loss, pred和gt位置对齐
        target_idx: target(PaO2/FiO2)所在的feature序号
        target_coeff: 对target进行loss加强的系数
    '''
    def __init__(self, target_idx, out_std, target_coeff=10) -> None:
        super().__init__()
        self.t_idx = target_idx
        self.t_coeff = target_coeff
        self.out_std=out_std

    def _cal_loss(self, pred, gt, mask):
        return torch.sqrt(torch.sum(torch.square((pred-gt)*mask)) / torch.sum(mask))

    def forward(self, pred:torch.Tensor, gt:torch.Tensor, mask:torch.Tensor):
        '''
            pred, gt, mask: (batch, n_fea, seq_len)
            return:
                all_loss, target_loss_item
        '''
        pred = pred[:,:, :-1]
        gt = gt[:,:,1:]
        mask = mask[:, :,:-1]
        target_loss = self._cal_loss(pred[:, self.t_idx, :], gt[:, self.t_idx, :], mask[:, self.t_idx, :])
        all_loss = self._cal_loss(pred, gt, mask)
        return self.t_coeff * target_loss + all_loss, self.out_std * target_loss.detach().clone().item()
        



