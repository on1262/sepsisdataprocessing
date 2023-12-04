from loguru import logger
import numpy as np
import matplotlib.pyplot as plt

logger.info('Import logger from loguru')


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
