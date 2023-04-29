import torch.nn as nn
import numpy as np
import torch
import tools
from tools import logger as logger
from torch.utils.data.dataloader import DataLoader
import os
from tqdm import tqdm
from .utils import Collect_Fn, DynamicLabelGenerator, DropoutLabelGenerator
import torchinfo
from .custom_lstm import LSTMLayer


    
class LSTMOriginalModel(nn.Module):
    '''带预测窗口的多分类判别模型'''
    def __init__(self, in_channels:int, n_cls=4, hidden_size=128) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_cls = n_cls
        self.hidden_size = hidden_size

        self.norm = nn.BatchNorm1d(num_features=in_channels)
        self.ebd = nn.Linear(in_features=in_channels, out_features=in_channels)
        self.den = nn.Linear(in_features=hidden_size, out_features=n_cls)
        self.sf = nn.Softmax(dim=-1)
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, batch_first=True)
        self.c_0 = nn.Parameter(torch.zeros((1, 1, hidden_size)), requires_grad=True)
        self.h_0 = nn.Parameter(torch.zeros((1, 1, hidden_size)), requires_grad=True)
        self.explainer_mode = False
        self.explainer_time_thres = 0

    def set_explainer_mode(self, value):
        self.explainer_mode = value
    
    def forward(self, x:torch.Tensor):
        '''
        给出某个时间点对未来一个窗口内是否发生ARDS的概率
        x: (batch, feature, time)
        '''
        # mask: (batch, time)
        x = self.norm(x)
        # x: (batch, feature, time)
        x = self.ebd(x.transpose(1,2))
        # x: (batch, time, feature) out带有tanh
        h_0, c_0 = self.h_0.expand(-1, x.size(0), -1).contiguous(), self.c_0.expand(-1, x.size(0), -1).contiguous()
        x, _ = self.lstm(x, (h_0, c_0))
        # x: (batch, time, hidden_size)
        x = self.den(x)
        if self.explainer_mode:
            return x[:,self.explainer_time_thres,3] # (batch,) 计算给定时刻的梯度贡献
        else:
            return self.sf(x) # (batch, time, n_cls)

class LSTMOriginalExplainerWrapper(nn.Module):
    '''替换原生LSTM层, 用于可解释性分析'''
    def __init__(self, model:LSTMOriginalModel, explainer_time_thres) -> None:
        super().__init__()
        self.model = model
        custom_lstm = LSTMLayer()
        # initalize LSTM weights
        custom_lstm.weight_ih = model.lstm.weight_ih_l0
        custom_lstm.weight_hh = model.lstm.weight_hh_l0
        custom_lstm.bias_ih = model.lstm.bias_ih_l0
        custom_lstm.bias_hh = model.lstm.bias_hh_l0
        self.model.lstm = custom_lstm
        self.model.set_explainer_mode(True)
        self.model.explainer_time_thres = explainer_time_thres

    def forward(self, x:torch.Tensor):
        data = torch.split(x, 100, dim=0)
        return torch.concat([self.model.forward(x) for x in data], dim=0)
         

class LSTMOriginalTrainer():
    def __init__(self, params:dict, dataset) -> None:
        self.params = params
        self.paths = params['paths']
        self.device = torch.device(self.params['device'])
        self.cache_path = self.paths['lstm_original_cache']
        self.model = LSTMOriginalModel(params['in_channels'])
        # self.rebalance_trainer = RebalanceTrainer(self.params)
        self.n_cls = len(self.params['centers'])
        self.criterion = OriginalClsLoss(self.n_cls, params['weight'])
        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=params['lr'])
        self.dataset = dataset
        self.available_idx = params['available_idx']
        self.dropout_generator = None # 应对robust metric
        self.generator = DynamicLabelGenerator(window=self.params['window'], centers=self.params['centers'], smoothing_band=self.params['smoothing_band'])
        self.train_dataloader = DataLoader(dataset=self.dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=Collect_Fn)
        self.valid_dataloader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=False, collate_fn=Collect_Fn)
        self.test_dataloader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=False, collate_fn=Collect_Fn)
        self.register_vals = {'train_loss':[], 'valid_loss':[]}
        # pre-train status
        self.trained = False
        if self.params['pretrained'] == True:
            self.trained = True
            load_path = self.load_model(None, load_best=True)
            logger.info(f'Load pretrained model from {load_path}')
            self.model = self.model.to(self.device)
            self.criterion = self.criterion.to(self.device)

    def get_loss(self):
        data = {
            'train': np.asarray(self.register_vals['train_loss']), 
            'valid': np.asarray(self.register_vals['valid_loss'])
        }
        return data
    
    def create_wrapper(self, shap_time_thres):
        return LSTMOriginalExplainerWrapper(self.model, shap_time_thres).to(self.device)

    def summary(self):
        torchinfo.summary(self.model)

    def reinit_cache(self, index):
        cache_path = os.path.join(self.cache_path, str(index))
        tools.reinit_dir(cache_path, build=True)
    
    def load_model(self, epoch, load_best=False):
        '''
        读取模型state_dict
        如果load_latest=True, epoch将被无视
        返回full path
        '''
        model_dir = os.path.join(self.cache_path, str(self.params['kf_index']))
        if load_best:
            model_path = tools.find_best(model_dir)
            self.model = torch.load(model_path, map_location=self.device)
        else:
            model_path = os.path.join(model_dir, f'{epoch}.pt')
            self.model = torch.load(model_path, map_location=self.device)
        return model_path

    def save_model(self, name):
        torch.save(self.model, os.path.join(self.cache_path, str(self.params['kf_index']), f'{name}.pt'))

    def _batch_forward(self, data, return_set=set(), addi_params:dict=None):
        for p in return_set:
            assert(p in {'logit', 'mask', 'label', 'acc'})
        data['data'] = data['data'][:, self.available_idx, :]
        np_data = np.asarray(data['data'])
        if addi_params is not None:
            if 'dropout' in addi_params.keys():
                _, dp_data = self.dropout_generator(np_data)
                data['data'] = torch.as_tensor(dp_data, dtype=torch.float32)
        seq_lens = data['length']
        mask = tools.make_mask((np_data.shape[0], np_data.shape[2]), seq_lens)
        mask[:, -1] = False
        mask, labels = self.generator(np_data, mask)
        mask, labels = torch.as_tensor(mask, device=self.device), torch.as_tensor(labels, device=self.device)
        x = data['data'].to(self.device)
        pred = self.model(x)
        loss = self.criterion(pred, labels, torch.as_tensor(mask, device=self.device))
        result = {'pred':pred, 'loss':loss}
        if 'logit' in return_set:
            acc_dict = {}
            label_gt = torch.argmax(labels, dim=-1)
            for idx in range(self.n_cls):
                acc_dict[idx] = [torch.sum(label_gt==idx).detach().cpu().item(), torch.sum((label_gt==idx)*pred[...,idx]).detach().cpu().item()]
            result['logit'] = acc_dict
        if 'acc' in return_set:
            acc_dict = {}
            label_gt = torch.argmax(labels, dim=-1)
            pred_argmax = torch.argmax(pred, dim=-1)
            for idx in range(self.n_cls):
                acc_dict[idx] = [torch.sum((label_gt==idx)*mask).detach().cpu().item(), torch.sum(mask*(label_gt==idx)*(pred_argmax==idx)).detach().cpu().item()]
            result['acc'] = acc_dict
        if 'mask' in return_set:
            result['mask'] = mask
        if 'label' in return_set:
            result['label'] = labels
        return result

    def train(self, addi_params:dict=None):
        if self.trained:
            return
        self.trained = True # 不能训练多次
        self.reinit_cache(self.params['kf_index'])
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        # 更新dropout
        if addi_params is not None:
            if 'dropout' in addi_params.keys():
                self.dropout_generator = DropoutLabelGenerator(dropout=addi_params['dropout'],miss_table=tools.generate_miss_table(self.dataset.idx_dict))
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
                    result = self._batch_forward(data, return_set={'acc'}, addi_params=addi_params)
                    pred, loss, a_dict = result['pred'], result['loss'], result['acc']
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
                tr_acc = ','.join([f'{acc:.3f}' for acc in accs])
                # validation phase
                acc_dict = {} # class:[n_gt,n_success_pred]
                self.dataset.mode('valid')
                self.model.eval()
                with torch.no_grad():
                    for data in self.valid_dataloader:
                        result = self._batch_forward(data, return_set={'acc'})
                        pred, loss, a_dict = result['pred'], result['loss'], result['acc']
                        for key in a_dict:
                            if key not in acc_dict:
                                acc_dict[key] = a_dict[key]
                            else:
                                acc_dict[key] = [acc_dict[key][0]+a_dict[key][0], acc_dict[key][1]+a_dict[key][1]]
                        loss_vals['valid_loss'] += loss.detach().cpu().item() * pred.size(0)
                # update accs by valid data
                accs = np.asarray([acc_dict[idx][1] / acc_dict[idx][0] for idx in range(self.n_cls)])
                mean_acc = np.mean(accs)
                val_acc = ','.join([f'{acc:.3f}' for acc in accs])
                loss_vals['valid_loss'] /= len(self.dataset)
                tq.set_postfix(v_loss=loss_vals['valid_loss'], t_loss=loss_vals['train_loss'], acc=mean_acc, tr_acc=tr_acc, val_acc=val_acc)
                tq.update(1)
                if mean_acc >= best_valid_metric:
                    best_valid_metric = mean_acc
                    best_epoch = epoch
                    self.save_model(f'best')
                self.register_vals['train_loss'].append(loss_vals['train_loss'])
                self.register_vals['valid_loss'].append(loss_vals['valid_loss'])
        best_path = self.load_model(None, load_best=True)
        logger.info(f'Load best model from {best_path} valid acc={best_valid_metric} epoch={best_epoch}')
    
    def predict(self, mode, warm_step=30, addi_params:dict=None):
        assert(self.trained == True)
        assert(mode in ['test', 'train', 'valid'])
        self.dataset.mode(mode)
        self.model = self.model.to(self.device).eval()
        register_vals = {'test_loss':0, 'pred':[], 'gt':[]}
        # 更新dropout
        if addi_params is not None:
            if 'dropout' in addi_params.keys():
                self.dropout_generator = DropoutLabelGenerator(dropout=addi_params['dropout'], miss_table=tools.generate_miss_table(self.dataset.idx_dict))
        with tqdm(total=len(self.test_dataloader)) as tq:
            tq.set_description(f'Testing, data={mode}')
            with torch.no_grad():
                for idx, data in enumerate(self.test_dataloader):
                    if warm_step is not None:
                        new_data = torch.concat([torch.expand_copy(data['data'][:, :, 0][..., None], (-1,-1,warm_step)), data['data']], dim=-1)
                        result = self._batch_forward({'data':new_data, 'length':data['length']}, addi_params=addi_params)
                        pred, loss = result['pred'][:,warm_step:,:], result['loss']
                    else:
                        result = self._batch_forward(data, addi_params=addi_params)
                        pred, loss = result['pred'], result['loss']
                    register_vals['pred'].append(pred.detach().clone())
                    register_vals['test_loss'] += loss.detach().cpu().item()
                    tq.set_postfix(loss=register_vals['test_loss'] / (idx+1))
                    tq.update(1)
        pred = torch.concat(register_vals['pred'], dim=0)
        return pred.cpu()


class OriginalClsLoss(nn.Module):
    '''提供动态预测的分类loss'''
    def __init__(self, n_cls:int, target_weight=None) -> None:
        '''forecast window: 向前预测的窗口'''
        super().__init__()
        self.n_cls = n_cls
        # input: (N,C,d1,...dk)
        self.weight = torch.as_tensor(target_weight) # 目标准确率的权重修正
    
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
        # 可以加入focal loss: torch.pow(1-pred, 5)
        loss = -torch.log(pred)*labels*(mask.permute(0,2,1)) # ->(batch, n_cls, seq_len)
        loss = torch.sum(torch.sum(loss, dim=2), dim=0) / (torch.sum(mask)) # -> (n_cls,) average
        self.weight = self.weight.to(loss.device) # ->(n_cls,)
        loss = torch.sum(loss * self.weight)
        return 5*loss
