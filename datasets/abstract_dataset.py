from torch.utils.data import Dataset
import abc


class AbstractDataset(Dataset, metaclass=abc.ABCMeta):
    '''抽象的Dataset, 要求所有dataset提供一个类似的接口'''
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def register_split(self, train_index:list, valid_index:list, test_index:list):
        '''向dataset注册数据集子集的划分, 每个index对应一个list, 存储对应样本的下标'''
        raise NotImplementedError

    @abc.abstractmethod
    def mode(self, mode=['train', 'valid', 'test', 'all']):
        '''切换数据集子集, 切换后影响get_item和__len__'''
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self, idx):
        raise NotImplementedError
