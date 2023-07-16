from analyzer import Analyzer
from datasets import MIMICIVDataset

analyzer_params = [
    'feature_explore',
    'nearest_4cls',
]

if __name__ == '__main__':
    dataset = MIMICIVDataset()
    if len(analyzer_params) > 0:
        analyzer = Analyzer(analyzer_params, dataset)
    print('Done')