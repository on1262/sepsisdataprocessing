import torch
import torch.nn as nn
import numpy as np

class net_v1(nn.Module):
    def __init__(self, in_fea, hidden=100) -> None:
        super().__init__()
        self.den1 = nn.Linear(in_fea, hidden)
        self.den2 = nn.Linear(hidden, hidden)
        self.den3 = nn.Linear(hidden, 2)
        self.dp = nn.Dropout(0.10)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.relu = nn.ReLU()
    
    def forward(self, x) -> torch.Tensor:
        x = self.relu(self.norm1(self.dp(self.den1(x))))
        x = self.relu(self.norm2(self.dp(self.den2(x))))
        x = self.den3(x)
        return x

class NNTrainer():
    def __init__(self, device:str, in_fea:int, hidden=100) -> None:
        self.device = torch.device(device)
        self.in_fea = in_fea
        self.net = net_v1(in_fea=in_fea, hidden=hidden)
        self.sf = torch.nn.Softmax()
    
    def train(self, hyper_params:dict, X_train:np.ndarray, Y_train:np.ndarray, X_valid:np.ndarray, Y_valid:np.ndarray, X_test:np.ndarray):
        epochs = hyper_params['epoch']
        lr = hyper_params['lr']
        print(f'NN fitting, params={hyper_params}')
        optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        cri_train = nn.CrossEntropyLoss(reduction='mean')
        cri_test = nn.CrossEntropyLoss(reduction='mean')
        X_train = torch.as_tensor(X_train, dtype=torch.float32).to(self.device)
        Y_train = torch.as_tensor(Y_train, dtype=torch.long).to(self.device)
        X_valid = torch.as_tensor(X_valid, dtype=torch.float32).to(self.device)
        Y_valid = torch.as_tensor(Y_valid, dtype=torch.long).to(self.device)
        X_test = torch.as_tensor(X_test, dtype=torch.float32).to(self.device)
        self.net = self.net.to(self.device)
        self.net = self.net.train()
        loss_down = []
        for epoch in range(epochs):
            self.net = self.net.eval()
            with torch.no_grad():
                valid_loss_item = cri_test(self.net(X_valid), Y_valid).item()
                loss_down.append(valid_loss_item)
            self.net = self.net.train()
            train_loss = cri_train.forward(self.net(X_train), Y_train)
            optim.zero_grad()
            train_loss.backward()
            optim.step()
            print(f'Epoch={epoch} train loss={train_loss.item()}, valid loss={valid_loss_item}')
        self.net = self.net.eval()
        out = self.sf(self.net(X_test)).detach().cpu().numpy()
        print('NN fit done.')
        return out[:,1], loss_down
            




        
