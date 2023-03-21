import torch
import torchinfo
import numpy as np
from datasets.mimic_dataset import MIMICDataset, Subject, Admission, Config # 这个未使用的import是pickle的bug
import models.mimic_model as mimic_model
from sklearn.model_selection import KFold

import tools
import os
from tqdm import tqdm
from tools import logger as logger
from .container import DataContainer


class Catboost2ClsAnalyzer:
    '''静态模型, 预测长期是否有ARDS'''
    def __init__(self, params, container:DataContainer) -> None:
        self.params = params
        self.dataset = container.dataset
        self.container = container
        self.model_name = 'catboost_2cls'


    def run(self):
        pass

