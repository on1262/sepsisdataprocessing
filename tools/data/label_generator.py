from typing import Any
import torch
import numpy as np
import tools
import compress_pickle as pickle
from os.path import exists, join as osjoin
from abc import abstractmethod

class LabelGenerator():
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def __call__(self, slice) -> Any:
        pass

class LabelGenerator_cls(LabelGenerator):
    def __init__(self, centers:list, soft_label=False, smooth_band=0) -> None:
        super().__init__()
        self.centers = centers
        self.soft_label = soft_label
        self.smooth_band = smooth_band

    def __call__(self, target:np.ndarray) -> Any:
        if self.soft_label:
            return self.label_cls(self.centers, target, smooth_band=self.smooth_band)
        else:
            return self.label_cls(self.centers, target, smooth_band=0)

    def label_cls(self, centers:list, nums:np.ndarray, smooth_band:int):
        '''
        centers: centers of each class, needs to be increasing, n_cls = len(centers)
        nums: input(in_shape,) can be arbitrary
        band: linear smoothing between two classes, band is the total width to be smoothed.
            When the inputs are outside the band (near the center or over the sides), they are hard labels, only inside the band they are soft labels.
        return: (... , len(centers)) where (...) = nums.shape
        '''
        num_classes = len(centers)
        smoothed_labels = np.zeros((nums.shape + (num_classes,)))
        for i in range(num_classes-1):
            center_i = centers[i]
            center_j = centers[i+1]
            lower = 0.5*(center_i + center_j) - smooth_band/2
            upper = 0.5*(center_i + center_j) + smooth_band/2
            hard_i = np.logical_and(nums >= center_i, nums <= lower)
            hard_j = np.logical_and(nums < center_j, nums > upper)
            mask = np.logical_and(nums > lower, nums <= upper)
            if smooth_band > 0 and mask.any():
                diff = (nums - center_i) / (center_j - center_i)
                smooth_i = 1 - diff
                smooth_j = diff
                smoothed_labels[..., i][mask] = smooth_i[mask]
                smoothed_labels[..., i+1][mask] = smooth_j[mask]
            smoothed_labels[..., i][hard_i] = 1
            smoothed_labels[..., i+1][hard_j] = 1
        smoothed_labels[..., 0][nums <= centers[0]] = 1
        smoothed_labels[..., -1][nums >= centers[-1]] = 1
        return smoothed_labels

class LabelGenerator_origin(LabelGenerator):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, target) -> Any:
        return target[..., None]
