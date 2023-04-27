import torch
import torch.nn as nn
from typing import List, Tuple
from torch import Tensor

class LSTMLayer(nn.Module):
    def __init__(self):
        super(LSTMLayer, self).__init__()
        self.weight_ih = None
        self.weight_hh = None
        self.bias_ih = None
        self.bias_hh = None

    def cell_forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        gates = (torch.matmul(input, self.weight_ih.t()) + self.bias_ih +
                 torch.matmul(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 2)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        return hy, (hy, cy)

    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        # input: (batch, n_fea, seq_len)

        inputs = input.transpose(0,1).unbind(0)
        outputs = torch.jit.annotate(List[Tensor], []) #  (batch, hidden)
        for i in range(len(inputs)):
            out, state = self.cell_forward(inputs[i], state)
            outputs += [out]
        return torch.concat(outputs, dim=0).transpose(0,1), state

