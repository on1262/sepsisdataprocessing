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
from .lstm_original import OriginalClsLoss
from catboost import CatBoostClassifier, Pool


class LSTMCascadeModel(nn.Module):
    '''Catboost初始化初态, 动态预测模型'''
    def __init__(self, catboost_model:CatBoostClassifier, in_channels:int, n_cls=4, hidden_size=128) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_cls = n_cls
        self.hidden_size = hidden_size
        self.catboost_model = catboost_model
        self.norm = nn.BatchNorm1d(num_features=in_channels)
        self.ebd = nn.Linear(in_features=in_channels, out_features=in_channels)
        self.den = nn.Linear(in_features=hidden_size, out_features=n_cls)
        self.lin_init = nn.Linear(in_features=n_cls, out_features=hidden_size)
        self.sf = nn.Softmax(dim=-1)
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, batch_first=True)
        self.h_0 = torch.zeros((1, 1, hidden_size))
        self.explainer_mode = False
        self.explainer_time_thres = 0

    def set_explainer_mode(self, value):
        self.explainer_mode = value
    
    def init_predict(self, x:np.ndarray):
        pool_test = Pool(data=x)
        return self.catboost_model.predict(pool_test, prediction_type='Probability')
    
    def tensor2numpy(self, x):
        return x.detach().clone().cpu().numpy()

    def forward(self, x:torch.Tensor):
        '''
        给出某个时间点对未来一个窗口内是否发生ARDS的概率
        x: (batch, feature, time)
        '''
        c_0 = torch.as_tensor(self.init_predict(self.tensor2numpy(x.clone()[:, :, 0])), dtype=torch.float32, device=x.device) # (batch, n_cls)
        c_0 = self.lin_init(c_0)[np.newaxis, ...] # (1, batch, hidden)
        # mask: (batch, time)
        x = self.norm(x)
        # x: (batch, feature, time)
        x = self.ebd(x.transpose(1,2))
        # x: (batch, time, feature) out带有tanh
        h_0 = self.h_0.to(x.device).expand(-1, x.size(0), -1).contiguous()
        x, _ = self.lstm(x, (h_0, c_0))
        # x: (batch, time, hidden_size)
        x = self.den(x)
        if self.explainer_mode:
            return x[:,self.explainer_time_thres,3] # (batch,) 计算给定时刻的梯度贡献
        else:
            return self.sf(x) # (batch, time, n_cls)


class LSTMCascadeTrainer():
    def __init__(self, params:dict, dataset) -> None:
        self.params = params
        self.paths = params['paths']
        self.device = torch.device(self.params['device'])
        self.cache_path = self.paths['lstm_cascade_cache']
        self.model = None
        # self.rebalance_trainer = RebalanceTrainer(self.params)
        self.n_cls = len(self.params['centers'])
        self.criterion = OriginalClsLoss(self.n_cls, params['weight'])
        self.opt = None
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
    
    # def create_wrapper(self, shap_time_thres):
    #     return LSTMOriginalExplainerWrapper(self.model, shap_time_thres).to(self.device)

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

    def _extract_data(self):
        result = {}
        for phase in ['train', 'valid', 'test']:
            self.dataset.mode(phase)
            data = []
            seq_lens = []
            for _, batch in enumerate(self.dataset):
                data.append(batch['data'])
                seq_lens.append(batch['length'])
            data = np.stack(data, axis=0)
            mask = tools.make_mask((data.shape[0], data.shape[-1]), seq_lens)
            mask, label = self.generator(data, mask) # e.g. [phase]['X']
            result[phase] = {'mask':mask[:,0], 'X':data[:,:,0], 'Y':label[:,0,:]}
        self.dataset.mode('all')
        return result
    
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

    def train_gbdt(self, addi_params:dict=None):
        p_catboost_cache = os.path.join(self.cache_path, 'catboost')
        tools.reinit_dir(p_catboost_cache, build=True) # 这是catboost输出loss的文件夹
        self.data_dict = self._extract_data()
        model = CatBoostClassifier(
            train_dir=p_catboost_cache, # 读取数据
            iterations=self.params['gbdt']['iterations'],
            depth=self.params['gbdt']['depth'],
            loss_function=self.params['gbdt']['loss_function'],
            learning_rate=self.params['gbdt']['learning_rate'],
            verbose=0,
            class_weights=self.params['weight'],
            use_best_model=True
        )
        train_X = self.data_dict['train']['X'][self.data_dict['train']['mask']]
        train_Y = self.data_dict['train']['Y'][self.data_dict['train']['mask']]
        valid_X = self.data_dict['valid']['X'][self.data_dict['valid']['mask']]
        valid_Y = self.data_dict['valid']['Y'][self.data_dict['valid']['mask']]
        if addi_params is not None:
            if 'dropout' in addi_params.keys():
                # 这个dropout是gbdt+LSTM训练时都要的
                self.dropout_generator = DropoutLabelGenerator(dropout=addi_params['dropout'], miss_table=self.dataset.miss_table())
                _, train_X = self.dropout_generator(train_X)
                _, valid_X = self.dropout_generator(valid_X)
        pool_train = Pool(train_X, np.argmax(train_Y, axis=-1))
        pool_valid = Pool(valid_X, np.argmax(valid_Y, axis=-1))
        model.fit(pool_train, eval_set=pool_valid, use_best_model=True)
        return model

    def train(self, addi_params:dict=None):
        if self.trained:
            return
        self.trained = True # 不能训练多次
        self.reinit_cache(self.params['kf_index'])
        gbdt_model = self.train_gbdt(addi_params=addi_params)
        self.model = LSTMCascadeModel(gbdt_model, self.params['in_channels'])
        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=self.params['lr'])
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
                self.dropout_generator = DropoutLabelGenerator(dropout=addi_params['dropout'], miss_table=self.dataset.miss_table())
        with tqdm(total=len(self.test_dataloader)) as tq:
            tq.set_description(f'Testing, data={mode}')
            with torch.no_grad():
                for idx, data in enumerate(self.test_dataloader):
                    if warm_step > 0:
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

