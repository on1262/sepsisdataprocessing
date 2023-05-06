from analyzer import Analyzer
from datasets import MIMICDataset

analyzer_params = [
    #'feature_explore',
    #'nearest_4cls',
    'LSTM_original_dp2',
    # 'LSTM_original_dp4'
    # 'LSTM_original_dp6'
    # 'LSTM_original_dp8'
    'LSTM_original',
    #'LSTM_cascade',
    #'catboost_dyn',
    #'catboost_dyn_dp2',
    #'catboost_dyn_dp4',
    #'catboost_dyn_dp6',
    #'catboost_dyn_dp8',
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