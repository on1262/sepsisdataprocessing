from analyzer import Analyzer
from datasets import MIMICDataset

analyzer_params = [
    #'feature_explore',
    #'nearest_4cls',
    #'LSTM_4cls'
    #'nearest_reg',
    #'LSTM_reg',
    # 'LSTM_quantile',
    # 'catboost_2cls'
    'ensemble_4cls'
]

if __name__ == '__main__':
    dataset = MIMICDataset()
    analyzer = Analyzer(analyzer_params, dataset)
