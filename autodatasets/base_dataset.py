from typing import Optional, Union, Callable, Tuple, Sequence
import pandas as pd
from . import data_reader
from . import data_downloader
import pathlib
import logging

class BaseDataset(object):
    """The base class of a dataset.

    :param df: Depends on the dataset type, it either contains the data (e.g. tabular and text)
        or labels with links to the examples (e.g. images).
    :param reader: An optional reader to read the rest data not in the dataframe.
    :param name: An optional name to retrieve this dataset later.


    :ivar df: The dataframe
    :ivar reader: The data reader
    :ivar name: The string name
    :cvar TYPE: The string type of this dataset, such as ``image_classification``
    """
    def __init__(self,
                 df: pd.DataFrame,
                 reader: Optional[data_reader.Reader] = None,
                 name: Optional[str] = '') -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f'{type(df)} is a not pandas DataFrame')
        self.df = df
        if len(self.df) == 0:
            logging.warning('No example is found as `df` is empty.')
            logging.warning('You may use `ds.reader.list_files()` to check all files.')
        self.reader = reader
        self.name = name

    TYPE = ''
    _DATASETS = dict()

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.df)

    def split(self, frac: float, shuffle: bool=True) -> Tuple['BaseDataset', 'BaseDataset']:
        """Split a dataset into two.

        :param frac: The fraction, in (0, 1), of examples for the first dataset.
        :param shuffle: If True (default), then randomly shuffle the examples before spliting.
        :return: Two datasets, each has the same type as this instance.
        """
        df = self.df.sample(frac=1, random_state=0) if shuffle else self.df
        n = int(len(df) * frac)
        return (self.__class__(df.iloc[:n].reset_index(), self.reader, self.name+'.0'),
                self.__class__(df.iloc[n:].reset_index(), self.reader, self.name+'.1'))

    @classmethod
    def add(cls, entry, *args) -> None:
        """Add a dataset to be retrieved later.

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
    def get(cls, name: str) -> 'BaseDataset':
        """Return the dataset by its name."""
        with data_downloader.NameContext(name):
            (fn, fn_args, fn_kwargs) = cls._DATASETS[(cls.TYPE, name)]
            ds = fn(*fn_args, **fn_kwargs)
            ds.name = name
            return ds

    @classmethod
    def list(cls) -> Sequence[str]:
        """Return the list of names of added datasets."""
        return [name for typ, name in cls._DATASETS if typ == cls.TYPE]

    @classmethod
    def from_df_func(cls, datapath: Optional[Union[str, Sequence[str]]],
                     df_func: Callable[[data_reader.Reader], pd.DataFrame]) -> 'BaseDataset':
        """Create a dataset from a dataframe function.

        :param datapath: A remote URL (data will be downloaded automatically) or a local datapath, or a list of them
        :param df_func: A function takes `self.reader` as its input to return the dataframe.
        """
        if datapath is None:
            reader = None
        else:
            if isinstance(datapath, str) or isinstance(datapath, pathlib.Path):
                datapath = [datapath]
            datapath = [(p if pathlib.Path(p).exists() else data_downloader.download(p, extract=True)) for p in datapath]
            reader = data_reader.create_reader(datapath)
        return cls(df_func(reader), reader)

    def _get_summary_path(self):
        if not self.name: return None
        return pathlib.Path(data_downloader.DATAROOT/self.name/f'{self.TYPE}_summary.pkl')

    @classmethod
    def summary_all(cls, quick: bool=False):
        """Return the summary of all datasets.

        :param quick: If True (default is False), then load saved summary from local disk instead of computing it.
           It often reduce the time. But you should call with ``quick=False`` before to have the summary saved.
        """
        summaries = []
        failed = []
        names = []
        for name in cls.list():
            if quick:
                ds = cls(pd.DataFrame([{'classname':'fack'}]), None, name)
                if not ds._get_summary_path().exists():
                    failed.append(name)
                    continue
            else:
                ds = cls.get(name)
            names.append(name)
            summaries.append(ds.summary().iloc[0])
        summary = pd.DataFrame(summaries, index=names)
        if failed:
            logging.warning(f'Failed to load summary info for {len(failed)} datasets. '
                'It may due to they haven\'t downloaded and preprocessed yet. '
                'You could change `quick=True` to `quick=False` to fix it')
        return summary


class ClassificationDataset(BaseDataset):
    """The base class of a classification dataset.

    Additional variables added besides :py:class:`BaseDataset`

    :ivar classes: The list of unique classes, each one is a string.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 reader: Optional[data_reader.Reader] = None,
                 name: Optional[str] = ''):
        super().__init__(df, reader, name)
        self.classes = sorted(self.df['classname'].unique().tolist())

    def split(self, frac: float, shuffle: bool=True):
        """Split a dataset into two.

        It is similar to :py:func:`BaseDataset.split`, but it guarantees the splitted datasets
        will have the same `classes` as this one.
        """
        a, b = super().split(frac, shuffle)
        a.classes = self.classes
        b.classes = self.classes
        return a, b
