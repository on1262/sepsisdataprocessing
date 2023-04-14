from analyzer import Analyzer
from datasets import MIMICDataset

analyzer_params = [
    #'feature_explore',
    #'nearest_4cls',
    #'LSTM_original'
    #'nearest_reg',
    #'LSTM_reg',
    # 'LSTM_quantile',
    'catboost_4cls'
    # 'ensemble_4cls'
]

if __name__ == '__main__':
    dataset = MIMICDataset()
    analyzer = Analyzer(analyzer_params, dataset)
