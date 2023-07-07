import numpy as np
import matplotlib.pyplot as plt
import math
import os
import pandas as pd
import seaborn as sns

def plot_curve(csv_path, out_dir):
    '''绘制缺失率和性能关系曲线'''
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
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

def plot_histogram():
    gbl_missrate = np.asarray([1,0.998,0.997,0.678,0.651,0.632,0.63,0.628,0.614,0.57,0.555,0.555,0.545,0.536,0.519,0.484,0.484,0.471,0.466,0.466,0.466,0.466,0.466,0.463,0.461,0.458,0.455,0.451,0.451,0.45,0.45,0.45,0.45,0.419,0.417,0.408,0.401,0.397,0.396,0.394,0.39,0.389,0.388,0.387,0.386,0.386,0.386,0.386,0.386,0.386,0.384,0.382,0.341,0.341,0.341,0.341,0.341,0.34,0.339,0.308,0.225,0.174,0.166,0.142,0.084,0.08,0.058,0.027,0.016,0.011,0.011,0.011,0.011,0.01,0.01,0.01,0.01,0.009,0.009,0.009,0.009,0.008,0.001,0.001,0,0,0,0,0,0,0,0,0])
    sns.set_theme(style="ticks")
    sns.histplot(gbl_missrate, bins=20)
    plt.title('Global missing rate distribution')
    plt.xlabel('Missing rate')
    plt.xlim([0,1])
    plt.ylabel('Count')
    plt.savefig('global_missrate.png')
    plt.close()

if __name__ == '__main__':
    # plot_histogram()
    plot_curve('metric_out.csv', '.')
