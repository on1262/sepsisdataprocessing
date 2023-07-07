from analyzer import Analyzer
from datasets import MIMICDataset

analyzer_params = [
    #'feature_explore',
    #'nearest_4cls',
    #'LSTM_original',
    #'LSTM_original_no_warm'
    #'LSTM_original_dp2',
    #'LSTM_original_dp4',
    #'LSTM_original_dp6',
    #'LSTM_original_dp8'
    #'LSTM_cascade',
    #'LSTM_extend_cascade',
    #'catboost_dyn',
    #'catboost_dyn_dp2',
    #'catboost_dyn_dp4',
    #'catboost_dyn_dp6',
    #'catboost_dyn_dp8',
    'catboost_4cls',
    #'catboost_4cls_dp2',
    #'catboost_4cls_dp4',
    #'catboost_4cls_dp6',
    #'catboost_4cls_dp8',
    #'catboost_4cls_full',
    #'catboost_4cls_full_50perc',
    #'catboost_4cls_full_25perc',
    #'catboost_4cls_9fea',
    #'catboost_4cls_1fea',
    #'random_forest',
    #'logistic_reg',
    #'catboost_forest',
    #'catboost_forest_dp2',
    #'catboost_forest_dp4',
    #'holt_winters'
]

if __name__ == '__main__':
    dataset = MIMICDataset()
    if len(analyzer_params) > 0:
        analyzer = Analyzer(analyzer_params, dataset)
    print('Done')