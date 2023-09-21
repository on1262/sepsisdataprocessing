import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def label_smoothing(centers:list, nums:np.ndarray, band=50):
    '''
    标签平滑
    centers: 每个class的中心点, 需要是递增的
    nums: 输入(in_shape,) 可以是任意的
    band: 在两个class之间进行线性平滑, band是需要平滑的总宽度
        当输入在band外时, 是硬标签, 只有在band内才是软标签
    '''
    num_classes = len(centers)
    smoothed_labels = np.zeros((nums.shape + (num_classes,)))
    for i in range(num_classes-1):
        center_i = centers[i]
        center_j = centers[i+1]
        lower = 0.5*(center_i + center_j) - band/2
        upper = 0.5*(center_i + center_j) + band/2
        mask = np.logical_and(nums >= lower, nums <= upper)
        hard_i = np.logical_and(nums >= center_i, nums < lower)
        hard_j = np.logical_and(nums < center_j, nums > upper)
        if mask.any():
            diff = (nums - center_i) / (center_j - center_i)
            smooth_i = 1 - diff
            smooth_j = diff
            smoothed_labels[..., i][mask] = smooth_i[mask]
            smoothed_labels[..., i+1][mask] = smooth_j[mask]
        smoothed_labels[..., i][hard_i] = 1
        smoothed_labels[..., i+1][hard_j] = 1
    smoothed_labels[..., 0][nums <= centers[0]] = 1
    smoothed_labels[..., -1][nums > centers[-1]] = 1
    return smoothed_labels
    
if __name__ == '__main__':
    centers = [0,100,200,300,500]
    nums = np.asarray([-1, 50, 100, 110, 150, 180, 420, 600])
    labels = label_smoothing(centers, nums, band=50)
    print(labels)