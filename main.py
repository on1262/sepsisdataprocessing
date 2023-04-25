from analyzer import Analyzer
from datasets import MIMICDataset

analyzer_params = [
    #'feature_explore',
    #'nearest_4cls',
    #'LSTM_balanced',
    'LSTM_original',
    #'catboost_4cls',
    #'random_forest',
    #'catboost_forest',
]

if __name__ == '__main__':
    dataset = MIMICDataset()
    if len(analyzer_params) > 0:
        analyzer = Analyzer(analyzer_params, dataset)
