import torch
import torch.nn as nn
import numpy as np
import tools
import torchinfo
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import os
from tools import logger as logger
from .lstm import Collect_Fn

class BaselineNearest():
    def __init__(self, params, dataset) -> None:
        self.dataset = dataset
        self.target_idx = params['target_idx']
        self.loader = DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=False, collate_fn=Collect_Fn)

    def predict(self, mode:str, start_points:list, pred_point:int):
        self.dataset.mode(mode)
        result = np.empty((len(start_points), len(self.dataset), pred_point))
        for idx, sp in enumerate(start_points):
            for batch in self.loader:
                for pp in range(pred_point):
                    result[idx, :, pp] = batch['data'][:, self.target_idx, sp]
        return result

