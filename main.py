from analyzer import Analyzer
from datasets import MIMICIVDataset
import yaml

with open('./launch_list.yml', 'r', encoding='utf-8') as fp:
    analyzer_params = yaml.load(fp, Loader=yaml.SafeLoader)['analyzer_params']

if __name__ == '__main__':
    dataset = MIMICIVDataset()
    if len(analyzer_params) > 0:
        analyzer = Analyzer(analyzer_params, dataset)
    print('Done')