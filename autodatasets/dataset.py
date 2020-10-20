from typing import Optional, Union
import pandas as pd
from . import data_reader

_DATASETS = dict()

def add_dataset(name: str, func, args=[]):
    _DATASETS[name] = (func, args)

def get_dataset(name: str):
    return _DATASETS[name][0](*_DATASETS[name][1])

def list_datasets():
    return list(_DATASETS.keys())

class BaseDataset(object):
    """The base dataset."""
    def __init__(self, name: Optional[str] = None,
                 root: Optional[Union[str, data_reader.Reader]] = None,
                 train_df: Optional[pd.DataFrame] = None,
                 valid_df: Optional[pd.DataFrame] = None,
                 test_df: Optional[pd.DataFrame] = None):
        self._name = name
        self._reader = root
        if isinstance(root, str):
            self._reader = data_reader.create_reader(root)
        if train_df is None and valid_df is None and test_df is None:
            raise ValueError('At least one of train_df, valid_df and test_df should be specified.')
        self._train_df = train_df
        self._valid_df = valid_df
        self._test_df = test_df

    def get_dfs(self):
        dfs = dict()
        if self._train_df is not None: dfs['train'] = self._train_df
        if self._valid_df is not None: dfs['valid'] = self._valid_df
        if self._test_df is not None: dfs['test'] = self._test_df
        return dfs

    @property
    def train_df(self):
        return self._train_df
