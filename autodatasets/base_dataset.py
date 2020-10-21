from typing import Optional, Union
import pandas as pd
from . import data_reader
from . import data_downloader
import pathlib

class BaseDataset(object):
    """The base dataset."""
    def __init__(self,
                 reader: Optional[data_reader.Reader] = None,
                 train_df: Optional[pd.DataFrame] = None,
                 valid_df: Optional[pd.DataFrame] = None,
                 test_df: Optional[pd.DataFrame] = None):
        self._reader = reader
        if train_df is None and valid_df is None and test_df is None:
            raise ValueError('At least one of train_df, valid_df and test_df should be specified.')
        self._train_df = train_df
        self._valid_df = valid_df
        self._test_df = test_df

    TYPE = ''
    _DATASETS = dict()

    @classmethod
    def add(cls, entry, *args):
        """Add a dataset.

        :param entry: Either a string name or a callable function to construct the dataset.
        :param args:
        """
        if isinstance(entry, str):
            if len(args) != 2:
                raise ValueError('xxx')
            cls._DATASETS[(cls.TYPE, entry)] = (args[0], args[1])
            return
        cls._DATASETS[(cls.TYPE, entry.__name__)] = (entry, [])

    @classmethod
    def get(cls, name: str):
        """Get a dataset by name."""
        with data_downloader.NameContext(name):
            key = (cls.TYPE, name)
            return cls._DATASETS[key][0](*cls._DATASETS[key][1])

    @classmethod
    def list(cls):
        """List all added datasets."""
        return [name for typ, name in cls._DATASETS if typ == cls.TYPE]

    @classmethod
    def from_df_func(cls, datapath: str, train_fn=None, valid_fn=None, test_fn=None):
        """Create a dataset from dataframe functions.

        :param cls:
        :param datapath: A remote URL or a local datapath.
        """
        if not pathlib.Path(datapath).exists():
            datapath = data_downloader.download(datapath)
        reader = data_reader.create_reader(datapath)
        get_df = lambda fn: fn(reader) if fn else None
        return cls(reader, get_df(train_fn), get_df(valid_fn), get_df(test_fn))

    def get_dfs(self):
        dfs = dict()
        if self._train_df is not None: dfs['train'] = self._train_df
        if self._valid_df is not None: dfs['valid'] = self._valid_df
        if self._test_df is not None: dfs['test'] = self._test_df
        return dfs

    @classmethod
    def summary_all(cls):
        names = cls.list()
        datasets = [cls.get(name) for name in names]
        summary = pd.DataFrame([ds.summary().loc['train'] for ds in datasets], index=names)
        for name in summary:
            if name.startswith('#'):
                summary[name] = summary[name].astype(int)
        return summary