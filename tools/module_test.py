import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

def plot_stack_proportion(data:dict[str, tuple], out_path=None):
    plt.figure(figsize=(15, 8))

    names = list(data.keys())
    style = [to_rgb(f'C{idx}') for idx in range(10)]
    plt.barh(names, [0 for _ in names])
    idx = 0
    for k_idx, (key, (x, label)) in enumerate(data.items()):
        x_sum = 0
        for idx in range(len(x)):
            color = np.asarray(style[k_idx % 10])
            color = np.clip(color * (1 - idx/len(x)) + 1.0*(idx/len(x)), 0, 0.95)
            plt.barh([key], x[idx], left=x_sum, color=tuple(color))
            label_wid = len(label[idx])*0.005
            if x[idx] > label_wid:
                plt.annotate(label[idx], (x_sum + x[idx]*0.5 - label_wid*0.5, k_idx), fontsize=10)
            x_sum += x[idx]
    
    plt.xlim(left=0, right=1)
    plt.savefig(out_path)
        
if __name__ == '__main__':
    data = {
        'A': ([0.5, 0.25, 0.25], ['A1', 'A2', 'Others']),
        'B': ([0.5, 0.25, 0.15, 0.05, 0.05], ['B1', 'B2', 'B3', 'B4', 'Others']),
        'C': ([0.7, 0.25, 0.04, 0.01], ['C1', 'C2', 'C3', 'Others'])
    }
    plot_stack_proportion(data, 'test.png')

def interp(fx:np.ndarray, fy:np.ndarray, x_start:float, interval:float, n_bins:int, missing=-1, fill_bin=['avg', 'latest']):
    # fx, (N,), N >= 1, sample time for each data point (irrgular)
    # fy: same size as fx, sample value for each data point
    # x: dim=1
    assert(fx.shape[0] == fy.shape[0] and len(fx.shape) == len(fy.shape) and len(fx.shape) == 1 and fx.shape[0] >= 1)
    assert(interval > 0 and n_bins > 0)
    assert(fill_bin in ['avg', 'latest'])
    result = np.ones((n_bins)) * missing
    
    for idx in range(n_bins):
        t_bin_start = x_start + (idx - 1) * interval
        t_bin_end = x_start + idx * interval
        valid_mask = np.logical_and(fx > t_bin_start, fx <= t_bin_end) # (start, end]
        if np.any(valid_mask): # have at least one point
            if fill_bin == 'avg':
                result[idx] = np.mean(fy[valid_mask])
            elif fill_bin == 'latest':
                result[idx] = fy[valid_mask][-1]
            else:
                assert(0)
        else: # no point in current bin
            if idx == 0:
                result[idx] = missing
            else:
                result[idx] = result[idx-1] # history is always available
    return result

if __name__ == '__main__':
    pass