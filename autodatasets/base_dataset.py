from typing import Optional, Union, Callable
import pandas as pd
from . import data_reader
from . import data_downloader
import pathlib
import logging

class BaseDataset(object):
    """The base dataset."""
    def __init__(self,
                 reader: Optional[data_reader.Reader] = None,
                 df: Optional[pd.DataFrame] = None,
                 name: str = ''):
        self.reader = reader
        self.df = df
        if self.df is None or len(self.df) == 0:
            logging.error('No example is found as `df` is empty.')
            logging.error('You may call `ds.reader.list_files()` to check all files.')
        self.name = name

    TYPE = ''
    _DATASETS = dict()

    def split(self, frac: float, shuffle: bool=True):
        """Split a dataset into two.
        """
        df = self.df.sample(frac=1, random_state=0) if shuffle else self.df
        n = int(len(df) * frac)
        return (self.__class__(self.reader, df.iloc[:n].reset_index(), self.name+'.0'),
                self.__class__(self.reader, df.iloc[n:].reset_index(), self.name+'.1'))

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
            ds = fn(*fn_args, **fn_kwargs)
            ds.name = name
            return ds

    @classmethod
    def list(cls):
        """List all added datasets."""
        return [name for typ, name in cls._DATASETS if typ == cls.TYPE]

    @classmethod
    def from_df_func(cls, datapath: str,
                     df_func: Callable[[data_reader.Reader], pd.DataFrame]) -> 'BaseDataset':
        """Create a dataset from dataframe functions.

        :param cls:
        :param datapath: A remote URL or a local datapath.
        """
        if isinstance(datapath, str) or isinstance(datapath, pathlib.Path):
            datapath = [datapath]
        datapath = [(p if pathlib.Path(p).exists() else data_downloader.download(p, extract=True)) for p in datapath]
        reader = data_reader.create_reader(datapath)
        return cls(reader, df_func(reader))

    def _get_summary_path(self):
        if not self.name: return None
        return pathlib.Path(data_downloader.DATAROOT/self.name/f'{self.TYPE}_summary.pkl')

    @classmethod
    def summary_all(cls, quick=False):
        names = cls.list()
        summaries = []
        for name in names:
            if quick:
                ds = cls(None, pd.DataFrame([{'classname':'fack'}]), name)
            else:
                ds = cls.get(name)
            summaries.append(ds.summary().iloc[0])
        summary = pd.DataFrame(summaries, index=names)
        return summary


class ClassificationDatatset(BaseDataset):
    def __init__(self,
                 reader: Optional[data_reader.Reader] = None,
                 df: Optional[pd.DataFrame] = None,
                 name: str = ''):
        super().__init__(reader, df, name)
        self.classes = sorted(self.df['classname'].unique().tolist())


    def split(self, frac: float, shuffle: bool=True):
        a, b = super().split(frac, shuffle)
        a.classes = self.classes
        b.classes = self.classes
        return a, b
