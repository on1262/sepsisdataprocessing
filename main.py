from analyzer import Analyzer
from datasets import MIMICDataset

analyzer_params = [
    #'feature_explore',
    #'nearest_4cls',
    'LSTM_balanced',
    #'LSTM_original',
    #'LSTM_original_pf_dp',
    #'LSTM_original_pf', pf没有测试过
    #'LSTM_original_dp',
    #'LSTM_original',
    #'LSTM_cascade',
    #'catboost_dyn',
    #'catboost_4cls',
    #'random_forest',
    #'catboost_forest',
    #'feature_explore'
]

if __name__ == '__main__':
    dataset = MIMICDataset()
    if len(analyzer_params) > 0:
        analyzer = Analyzer(analyzer_params, dataset)
    print('Done')