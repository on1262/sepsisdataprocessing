from analyzer import Analyzer
from datasets import MIMICDataset

analyzer_params = [
    #'nearest_4cls',
    #'LSTM_4cls'
    #'nearest_reg',
    #'LSTM_reg',
    'catboost_2cls'
]

if __name__ == '__main__':
    dataset = MIMICDataset()
    analyzer = Analyzer(analyzer_params, dataset)
