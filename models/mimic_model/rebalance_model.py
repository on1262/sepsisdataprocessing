import torch.nn as nn
import numpy as np
import torch
import tools
from tools import logger as logger
from tqdm import tqdm
import os

class RebalanceTrainer:
    def __init__(self, params) -> None:
        self.params = params
        self.n_cls = len(params['centers'])
        self.device = torch.device(params['device'])
        self.cache_path = params['paths']['rebalance_cache_path']
        tools.reinit_dir(self.cache_path)
        self.model = None

    def load_model(self, model_path):
        logger.info('Load LSTM cls model from:' + model_path)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
    
    def train(self, in_dict:dict):
        '''
        in_dict: {'train':{'x', 'label','mask'}, 'valid':{'x', 'label','mask'}}
        train_x, valid_x: (batch, seq_len, n_cls)
        mask: (batch, seq_len)
        train_label, valid_label: (batch, seq_len, n_cls) 必须是one-hot
        '''
        self.model = RebalanceModel(n_cls=self.n_cls).to(self.device)
        for phase in in_dict.keys():
            for key in in_dict[phase].keys():
                in_dict[phase][key] = in_dict[phase][key].to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        best_epoch = 0
        best_valid_acc = 0
        epochs = self.params['rebalance_epochs']
        for epoch in range(epochs):
            # train
            opt.zero_grad()
            out_x, soft_accs, loss = self.model(in_dict['train']['x'], in_dict['train']['label'], in_dict['train']['mask'])
            t_label = torch.argmax(in_dict['train']['label'],dim=-1)
            t_mask = in_dict['train']['mask']
            pred = torch.argmax(out_x, dim=-1)
            real_acc = [torch.sum((t_label==idx)*(pred==idx)*t_mask).detach().cpu().item()/torch.sum(t_mask*(t_label==idx)).detach().cpu().item() for idx in range(4)]
            if (epoch+1) % (epochs//10) == 0 or epoch == epochs-1 or epoch == 0:
                log_meanacc = np.mean(real_acc)
                log_loss = loss.detach().clone().cpu()
                # log_softacc = np.asarray(soft_accs.detach().clone().cpu())
                print(f'epoch={epoch}, loss={log_loss}, tr_real_acc={real_acc}, tr_mean={log_meanacc}')
            loss.backward()
            opt.step()
            # save model
            torch.save(self.model.state_dict(), os.path.join(self.cache_path, f'{epoch}.pt'))
            # valid
            with torch.no_grad():
                v_label = torch.argmax(in_dict['valid']['label'],dim=-1)
                v_mask = in_dict['valid']['mask']
                out_x, _, _ = self.model(in_dict['valid']['x'], in_dict['valid']['label'], in_dict['valid']['mask'])
                pred = torch.argmax(out_x, dim=-1)
                real_acc = np.mean(
                    [torch.sum((v_label==idx)*(pred==idx)*v_mask).detach().cpu().item()/torch.sum((v_label==idx)*v_mask).detach().cpu().item() for idx in range(4)])
                if real_acc > best_valid_acc:
                    best_valid_acc = real_acc
                    best_epoch = epoch
        # load best model
        best_path = os.path.join(self.cache_path, f'{best_epoch}.pt')
        self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        logger.info(f'Load best model from {best_path} valid mean acc={best_valid_acc}')

    def predict(self, test_x):
        '''输出改进后的x'''
        with torch.no_grad():
            out_x = self.model(test_x, None, None)
        return out_x


class RebalanceModel(nn.Module):
    '''对训练好的model输出logits进行再优化, 最大化平均acc'''
    def __init__(self, n_cls) -> None:
        super().__init__()
        self.coeff = nn.Parameter(torch.ones((n_cls,)), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros((n_cls,)), requires_grad=True)
        self.n_cls = n_cls

    def forward(self, x:torch.Tensor, label:torch.Tensor, mask:torch.Tensor):
        '''
        x: (batch, seq_len, n_cls) 最后一维表示概率, 都是正数
        label: (batch, seq_len, n_cls) one-hot矢量
        mask: (batch, seq_len) or None
        '''
        x = torch.softmax(self.bias + x * torch.abs(self.coeff), dim=-1)
        if label is None:
            return x
        x_max = torch.max(x.detach(), dim=-1).values[...,None]
        if mask is None:
            mask = 1
        else:
            mask = mask[..., None]
        invalid_x = label*x*(1/x.detach())*(x<x_max)*mask # 这个计算是完全准确的
        soft_accs = 1 - (torch.sum(torch.sum(invalid_x, dim=0), dim=0) / torch.sum(torch.sum(label*mask, dim=0), dim=0)) # softaccs大于0
        acc_loss = torch.mean(soft_accs) # loss需要使得mean_acc尽可能大
        loss = acc_loss # 降低loss会使得acc_loss升高, balance_loss降低
        return x, soft_accs, loss


if __name__ == '__main__':
    '''测试RebalanceModel是否能达到效果'''
    data = torch.zeros((200,20,4))
    label = torch.zeros((200,20,4))
    label[1:100,:,0] = 1 # 49.5%
    data[1:50,:,3] = 0.6 + 0.2*torch.randn((49,20))
    data[:,:,0] = 1
    label[100:150,:,1] = 1 # 25%
    data[100:150,:,1] = 0.8 # 25%
    label[150:,:,2] = 1 # 25%
    data[150:,:,2] = 0.5
    label[0:1,:,3] = 1 # 0.5%
    data[0:1,:,3] = 0.6 + 0.05*torch.randn((1,20)) # 25%
    label = label.to('cuda:1')
    data = data.to('cuda:1')
    label_gt = torch.argmax(label,dim=-1)
    model = RebalanceModel(n_cls=4).to('cuda:1')
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    epochs = 5000
    for epoch in range(epochs):
        opt.zero_grad()
        x, soft_accs, loss = model(data,label)
        pred = torch.argmax(x, dim=-1)
        real_acc = [torch.sum((label_gt==idx)*(pred==idx)).detach().cpu().item()/torch.sum(label_gt==idx).detach().cpu().item() for idx in range(4)]
        if (epoch+1) % (epochs//10) == 0 or epoch == epochs-1 or epoch == 0:
            print(f'loss={loss.detach().clone().cpu()}, real_acc={real_acc}, mean={np.mean(real_acc)}, soft_acc={np.asarray(soft_accs.detach().clone().cpu())}')
        loss.backward()
        opt.step()
    print('coeff:', model.coeff)
    print('bias:', model.bias)

