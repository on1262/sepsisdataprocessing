import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb


class SummaryWriter:
    '''A simple version of summary writer that mimics tensorboard'''
    def __init__(self) -> None:
        self.clear()

    def add_scalar(self, tag, value, global_step):
        self.data[tag] = self.data[tag] + [(value, global_step)] if tag in self.data else [(value, global_step)]

    def clear(self):
        self.data = {}
    
    def plot(self, tags:[list, str], k_fold=False, log_y=False, title='Title', out_path:str=None):
        plt.figure(figsize=(12, 8))
        plt.title(title)
        tags:list = [tags] if isinstance(tags, str) else tags
        for tag in tags:
            tag_data = np.asarray(self.data[tag])
            color = f'C{tags.index(tag)}' if not k_fold else 'grey'
            alpha = 1.0 if not k_fold else 0.5
            plt.plot(tag_data[:, 1], tag_data[:, 0], '-o', color=color, alpha=alpha)
        if k_fold: # tags are different folds. Plot mean and std bar
            ax = plt.gca()
            data = np.asarray([self.data[tag] for tag in tags])
            x_ticks = data.mean(axis=0)[:, 1]
            mean_y = data.mean(axis=0)[:, 0] # (n_steps,)
            std_data = data[:, :, 0].std(axis=0) # (n_steps,)
            ax.fill_between(x_ticks, mean_y + std_data * 0.5, mean_y - std_data * 0.5, alpha=0.5, linewidth=0)
            plt.plot(x_ticks, mean_y, '-o', color='C0')
        if log_y:
            plt.yscale('log')
        if out_path is not None:
            plt.savefig(out_path)
        plt.close()


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
    wrt = SummaryWriter()
    for fold in range(5):
        for idx in range(10):
            wrt.add_scalar(f'tr_{fold}', np.random.rand(), idx)
    wrt.plot([f'tr_{fold}' for fold in range(5)], k_fold=True, log_y=False, title='test', out_path='test_plot/test_writer.png')


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