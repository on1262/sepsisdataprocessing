import seaborn as sns
from matplotlib import pyplot as plt
import tools
import numpy as np
import pandas as pd

def plot_missrate_comp():
    processed_row = tools.load_pkl('outputs/feature_explore[ards@origin]/row_missrate.pkl').flatten()
    processed_col = tools.load_pkl('outputs/feature_explore[ards@origin]/col_missrate.pkl').flatten()
    raw_row = tools.load_pkl('outputs/feature_explore[raw@version]/row_missrate.pkl').flatten()
    raw_col = tools.load_pkl('outputs/feature_explore[raw@version]/col_missrate.pkl').flatten()
    row_data = np.concatenate([processed_row, raw_row], axis=0)
    col_data = np.concatenate([processed_col, raw_col], axis=0)
    for data, label in zip([row_data, col_data], ['row', 'col']):
        df = pd.DataFrame(data, columns=['data'])
        df['source'] = 'raw'
        lens = len(processed_row) if label == 'row' else len(processed_col)
        df.loc[:lens, 'source'] = 'processed'
        sns.histplot(df, x='data', hue='source', bins=20, stat='proportion', common_norm=False, shrink=0.95, element='bars', edgecolor=None)
        plt.xlabel(f'{label} missrate')
        plt.savefig(f'test_plot/{label}_missrate.png')
        plt.close()



if __name__ == '__main__':
    plot_missrate_comp()