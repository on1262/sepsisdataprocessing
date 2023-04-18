import torch.nn as nn
import numpy as np
import torch
import tools
from tools import logger as logger
from torch.utils.data.dataloader import DataLoader
import os
from tqdm import tqdm
from .utils import Collect_Fn, DynamicLabelGenerator
import torchinfo

class LSTMBalancedModel(nn.Module):
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


class LSTMBalancedTrainer():
    def __init__(self, params:dict, dataset) -> None:
        self.params = params
        self.paths = params['paths']
        self.device = torch.device(self.params['device'])
        self.cache_path = self.paths['lstm_balanced_cache']
        tools.reinit_dir(self.cache_path, build=True)
        self.model = LSTMBalancedModel(params['device'], params['in_channels'])
        self.n_cls = len(self.params['centers'])
        self.criterion = BalancedClsLoss(self.n_cls)
        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=params['lr'])
        self.dataset = dataset
        self.available_idx = params['available_idx']
        self.generator = DynamicLabelGenerator(window=self.params['window'], centers=self.params['centers'], smoothing_band=self.params['smoothing_band'])
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

    def load_model(self, model_path):
        logger.info('Load LSTM cls model from:' + model_path)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))


    def _batch_forward(self, data, return_logits=False):
        np_data = np.asarray(data['data'][:, self.available_idx, :])
        seq_lens = data['length']
        mask = tools.make_mask((np_data.shape[0], np_data.shape[2]), seq_lens)
        mask[:, -1] = False
        mask, labels = self.generator(np_data, mask)
        mask, labels = torch.as_tensor(mask, device=self.device), torch.as_tensor(labels, device=self.device)
        x = data['data'][:, self.available_idx, :].to(self.device)
        pred = self.model(x)
        loss = self.criterion(pred, labels, torch.as_tensor(mask, device=self.device))
        if return_logits:
            logits_dict = {}
            label_gt = torch.argmax(labels, dim=-1)
            for idx in range(self.n_cls):
                logits_dict[idx] = [torch.sum(label_gt==idx).detach().cpu().item(), torch.sum((label_gt==idx)*pred[...,idx]).detach().cpu().item()]
            return pred, loss, logits_dict
        else:
            return pred, loss

    def train(self):
        cache_path = os.path.join(self.cache_path)
        tools.reinit_dir(cache_path, build=True)
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        
        with tqdm(total=self.params['epochs']) as tq:
            tq.set_description(f'Training, Epoch')
            best_epoch = 0
            best_valid_metric = 0
            for epoch in range(self.params['epochs']):
                loss_vals = {'train_loss':0, 'valid_loss':0}
                # train phase
                acc_dict = {} # class:[n_gt,n_success_pred]
                self.dataset.mode('train')
                self.model.train()
                for data in self.train_dataloader:
                    pred, loss, a_dict = self._batch_forward(data, return_logits=True)
                    for key in a_dict:
                        if key not in acc_dict:
                            acc_dict[key] = a_dict[key]
                        else:
                            acc_dict[key] = [acc_dict[key][0]+a_dict[key][0], acc_dict[key][1]+a_dict[key][1]]
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    loss_vals['train_loss'] += loss.detach().cpu().item() * pred.size(0)
                loss_vals['train_loss'] /= len(self.dataset) # 避免最后一个batch导致计算误差
                accs = np.asarray([acc_dict[idx][1] / acc_dict[idx][0] for idx in range(self.n_cls)])
                self.criterion.update_accs(accs)
                tr_acc = ','.join([f'{acc:.3f}' for acc in accs])
                # validation phase
                acc_dict = {} # class:[n_gt,n_success_pred]
                self.dataset.mode('valid')
                self.model.eval()
                with torch.no_grad():
                    for data in self.valid_dataloader:
                        pred, loss, a_dict = self._batch_forward(data, return_logits=True)
                        for key in a_dict:
                            if key not in acc_dict:
                                acc_dict[key] = a_dict[key]
                            else:
                                acc_dict[key] = [acc_dict[key][0]+a_dict[key][0], acc_dict[key][1]+a_dict[key][1]]
                        loss_vals['valid_loss'] += loss.detach().cpu().item() * pred.size(0)
                # update accs by valid data
                accs = np.asarray([acc_dict[idx][1] / acc_dict[idx][0] for idx in range(self.n_cls)])
                mean_acc = np.mean(accs)
                # val_acc = ','.join([f'{acc:.3f}' for acc in accs])
                coeff = ','.join([f'{c:.3f}' for c in self.criterion.get_coeff()])
                loss_vals['valid_loss'] /= len(self.dataset)
                tq.set_postfix(v_loss=loss_vals['valid_loss'], t_loss=loss_vals['train_loss'], acc=mean_acc, tr_acc=tr_acc, coeff=coeff)
                tq.update(1)
                if mean_acc >= best_valid_metric:
                    best_valid_metric = mean_acc
                    best_epoch = epoch
                self.register_vals['train_loss'].append(loss_vals['train_loss'])
                self.register_vals['valid_loss'].append(loss_vals['valid_loss'])
                torch.save(self.model.state_dict(), os.path.join(cache_path, f'{epoch}.pt'))
        best_path = os.path.join(cache_path, f'{best_epoch}.pt')
        self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        logger.info(f'Load best model from {best_path} valid loss={best_valid_metric}')
    
    def predict(self, mode, warm_step=30):
        assert(mode in ['test', 'train', 'valid'])
        self.dataset.mode(mode)
        self.model = self.model.to(self.device).eval()
        register_vals = {'test_loss':0, 'pred':[], 'gt':[]}
        with tqdm(total=len(self.test_dataloader)) as tq:
            tq.set_description(f'Testing, data={mode}')
            with torch.no_grad():
                for idx, data in enumerate(self.test_dataloader):
                    if warm_step is not None:
                        new_data = torch.concat([torch.expand_copy(data['data'][:, :, 0][..., None], (-1,-1,warm_step)), data['data']], dim=-1)
                        pred, loss = self._batch_forward({'data':new_data, 'length':data['length']})
                        pred = pred[:,warm_step:,:]
                    else:
                        pred, loss = self._batch_forward(data)
                    register_vals['pred'].append(pred.detach().clone().cpu())
                    register_vals['test_loss'] += loss.detach().cpu().item()
                    tq.set_postfix(loss=register_vals['test_loss'] / (idx+1))
                    tq.update(1)
        pred = torch.concat(register_vals['pred'], dim=0)
        return pred


class BalancedClsLoss(nn.Module):
    '''提供动态预测的分类loss'''
    def __init__(self, n_cls:int, target_weight=None) -> None:
        '''
        forecast window: 向前预测的窗口
        '''
        super().__init__()
        self.n_cls = n_cls
        # input: (N,C,d1,...dk)
        self.logits = np.zeros((4,)) # 准确率记录
        self.weight = np.ones((4,)) # 目标准确率的权重修正
        self.coeff = torch.ones((4,))/4
    
    def update_accs(self, new_accs):
        '''更新准确率, 越频繁越好'''
        self.logits = np.asarray(new_accs)
        self.coeff = self.cal_coeff() # 更新coeff
    
    def cal_coeff(self):
        '''按照当前准确率计算各分类loss权重'''
        target_logits = np.mean(self.logits) * (self.weight * self.n_cls / np.sum(self.weight)) # 当前性能对应的目标性能
        delta = (self.logits - target_logits)
        coeff = torch.sigmoid(-10*torch.as_tensor(delta)) # 只推进不满足target的部分
        coeff = coeff / torch.sum(coeff)
        return coeff

    def get_coeff(self):
        return np.asarray(self.coeff)
    
    def forward(self, pred:torch.Tensor, labels:torch.Tensor, mask:torch.Tensor):
        '''
            pred, labels: (batch, seq_len, n_cls)
            mask: (batch, seq_len)
            labels可以是软标签, 但不建议使用(因为交叉熵的原因)
        '''
        assert(pred.size() == labels.size())
        if len(mask.size()) + 1 == len(pred.size()): 
            mask = mask[..., None] # mask->(batch, seq_len, 1)
        pred, labels = pred.permute(0, 2, 1), labels.permute(0, 2, 1) # permute: ->(batch, n_cls, seq_len)
        # 只有label中置为1的样本分类对结果有贡献
        loss = (-torch.log(pred))*(pred < 0.6)*labels*(mask.permute(0,2,1)) # ->(batch, n_cls, seq_len)
        loss = torch.sum(torch.sum(loss, dim=2), dim=0) / torch.sum(mask) # -> (n_cls,) average
        coeff = self.coeff.to(loss.device) # ->(n_cls,)
        loss = torch.sum(loss * coeff)
        return loss