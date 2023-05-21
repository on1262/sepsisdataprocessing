import numpy as np
import matplotlib.pyplot as plt
import math
import os
import pandas as pd

def plot_curve(csv_path, out_dir):
    '''绘制缺失率和性能关系曲线'''
    df = pd.read_csv(csv_path, encoding='utf-8')
    missrates = np.linspace(0, 1, 11)
    legends = []
    for idx in range(len(df)):
        metrics = np.asarray(df.iloc[idx, 0:11])
        plt.plot(missrates, metrics, '+-', color=f'C{idx}')
        auc = df['auc'][idx]
        name = df['name'][idx]
        legends.append(f'{name}, AUC={auc:.3f}')
    plt.title('Performance with missrate')
    plt.legend(legends)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Missing rate")
    plt.ylabel("Metric")
    save_path = os.path.join(out_dir, 'metric_comp.png')
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    plot_curve('metric_out.csv', '.')
