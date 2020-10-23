from typing import Optional, Union
import pandas as pd
from . import data_reader
from . import data_downloader
import pathlib
import logging

class BaseDataset(object):
    """The base dataset."""
    def __init__(self,
                 reader: Optional[data_reader.Reader] = None,
                 train_df: Optional[pd.DataFrame] = None,
                 valid_df: Optional[pd.DataFrame] = None,
                 test_df: Optional[pd.DataFrame] = None):
        self._reader = reader
        empty_df = lambda df: df is None or len(df) == 0
        if empty_df(train_df) and empty_df(valid_df) and empty_df(test_df):
            logging.error('No example is found as all train_df, valid_df and test_df are empty.')
        self._train_df = train_df
        self._valid_df = valid_df
        self._test_df = test_df

    TYPE = ''
    _DATASETS = dict()

    @property
    def reader(self):
        return self._reader

    @classmethod
    def add(cls, entry, *args):
        """Add a dataset.

        :param entry: Either a string name or a callable function to construct the dataset.
        :param args:
        """
        fn_args = []
        fn_kwargs = {}
        if isinstance(entry, str):
            name = entry
            assert len(args), 'xxx'
            fn = args[0]
            if len(args)>1: fn_args = args[1]
            if len(args)>2: fn_kwargs = args[2]
        else:
            name = entry.__name__.replace('_', '-')
            fn = entry
        cls._DATASETS[(cls.TYPE, name)] = (fn, fn_args, fn_kwargs)

    @classmethod
    def get(cls, name: str):
        """Get a dataset by name."""
        with data_downloader.NameContext(name):
            (fn, fn_args, fn_kwargs) = cls._DATASETS[(cls.TYPE, name)]
            return fn(*fn_args, **fn_kwargs)

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
        if isinstance(datapath, str) or isinstance(datapath, pathlib.Path):
            datapath = [datapath]
        datapath = [(p if pathlib.Path(p).exists() else data_downloader.download(p)) for p in datapath]
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