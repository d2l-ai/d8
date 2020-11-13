# This file is generated from core/base_dataset.md automatically through:
#    d2lbook build lib
# Don't edit it directly

#@save_all
#@hide_all
from typing import Optional, Union, Callable, Tuple, Sequence, List, Type, TypeVar
import pandas as pd
from d8 import core
import pathlib
import logging
from matplotlib import pyplot as plt

__all__ = ['BaseDataset', 'ClassificationDataset', 'show_images']

_T = TypeVar("_T", bound='BaseDataset')

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
                 reader: Optional[core.Reader] = None,
                 name: str = '') -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f'{type(df)} is a not pandas DataFrame')
        self.df = df
        if len(self.df) == 0:
            logging.warning('No example is found as `df` is empty.')
            logging.warning('You may use `ds.reader.list_files()` to check all files.')
        self.reader = core.EmptyReader() if reader is None else reader
        self.name = name

    TYPE = ''
    _DATASETS = dict()  # type: ignore

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.df)

    def split(self, frac: Union[float, Sequence[float]], shuffle: bool = True, seed: int = 0) -> List['BaseDataset']:
        """Split a dataset.

        When ``frac`` is a float, it returns two datasets, with the first one has frac*len(self) examples.
        If ``frac`` is a list, then its sum should be less or equal to 1. len(frac)+1 datasets will return.

        :param frac: A fraction, in (0, 1), or a list of fractions.
        :param shuffle: If True (default), then randomly shuffle the examples before spliting.
        :param seed: The random seed (default 0) to shuffle the examples given ``shuffle=True``.
        :return: A list of datasets, each has the same type as this instance.
        """
        df = self.df.sample(frac=1, random_state=seed) if shuffle else self.df
        fracs = core.listify(frac)
        if sum(fracs) >= 1:
            raise ValueError(f'the sum of frac {sum(fracs)} should be less than 1')
        fracs = fracs + [1.0 - sum(fracs)]
        rets = []
        s = 0
        for i, f in enumerate(fracs):
            if f <= 0: raise ValueError(f'frac {f} is not in (0, 1)')
            e = int(sum(fracs[:(i+1)]) * len(df))
            rets.append(self.__class__(df.iloc[s:e].reset_index(), self.reader, f'{self.name}.{i}'))
            s = e
        return rets

    def merge(self, *args: 'BaseDataset') -> 'BaseDataset':
        """Merge with other datasets.

        :param args: One or multiple datasets
        :return: A new dataset with examples merged.
        """
        dfs = [self.df]
        for ds in args:
            if ds.reader != self.reader:
                raise ValueError('You cannot merge with another dataset with a different reader')
            dfs.append(ds.df)
        return self.__class__(pd.concat(dfs, axis=0, ignore_index=True), self.reader, self.name+'.merged')

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
        with core.NameContext(name):
            (fn, fn_args, fn_kwargs) = cls._DATASETS[(cls.TYPE, name)]
            ds = fn(*fn_args, **fn_kwargs)
            ds.name = name
            return ds

    @classmethod
    def list(cls) -> Sequence[str]:
        """Return the list of names of added datasets."""
        return [name for typ, name in cls._DATASETS if typ == cls.TYPE]

    @classmethod
    def create_reader(cls, data_path: Optional[Union[str, Sequence[str]]], name: Optional[str]=None) -> core.Reader:
        def download(data_path):
            return [(p if pathlib.Path(p).exists() else core.download(p, extract=True)) for p in data_path]
        if name:
            with core.NameContext(name):
                data_path = download(core.listify(data_path))
        else:
            data_path = download(core.listify(data_path))
        return core.create_reader(data_path)

    @classmethod
    def from_df_func(cls: Type[_T], data_path: Optional[Union[str, Sequence[str]]],
                     df_func: Callable[[core.Reader], pd.DataFrame]) -> _T:
        """Create a dataset from a dataframe function.

        :param data_path: A remote URL (data will be downloaded automatically) or a local data_path, or a list of them
        :param df_func: A function takes `self.reader` as its input to return the dataframe.
        """
        reader = cls.create_reader(data_path)
        return cls(df_func(reader), reader)

    def summary(self) -> pd.DataFrame:
        """Returns a summary about this dataset."""
        raise NotImplementedError()

    def _get_summary_path(self) -> Optional[pathlib.Path]:
        if not self.name: return None
        return pathlib.Path(core.DATAROOT/self.name/f'{self.TYPE}_summary.pkl')

    @classmethod
    def summary_all(cls, quick: bool=False) -> pd.DataFrame:
        """Return the summary of all datasets.

        :param quick: If True (default is False), then load saved summary from local disk instead of computing it.
           It often reduce the time. But you should call with ``quick=False`` before to have the summary saved.
        """
        summaries = []
        failed = []
        names = []
        for name in cls.list():
            if quick:
                ds = cls(df=pd.DataFrame([{'class_name':'fack'}]), reader=None, name=name)
                path = ds._get_summary_path()
                if not path or not path.exists():
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
        return summary.sort_values(summary.columns[0])


class ClassificationDataset(BaseDataset):
    """The base class of a classification dataset.

    Additional variables added besides :py:class:`BaseDataset`

    :ivar classes: The list of unique classes, each one is a string.
    :ivar label: The column name that stores the labels.
    """
    def __init__(self, df: pd.DataFrame, reader: core.Reader,
                 label: Union[str, int]):
        if isinstance(label, int):
            label = df.columns[label]
        if label not in df.columns:
            raise ValueError(f'Label {label} is not in {df.columns}')
        self.label = label
        df = df[~df[label].isnull()]
        self.classes = sorted(df[label].unique().tolist())
        super().__init__(df, reader)

    def split(self, frac: Union[float, Sequence[float]], shuffle: bool = True, seed: int = 0):
        """Split a dataset into two.

        It is similar to :py:func:`BaseDataset.split`, but it guarantees the splitted datasets
        will have the same `classes` as this one.
        """
        rets = super().split(frac, shuffle, seed)
        for r in rets: r.classes = self.classes  # type: ignore
        return rets

def show_images(images, layout, scale):
    nrows, ncols = layout
    if len(images) != nrows * ncols:
        raise ValueError(f'Cannot layout f{len(images)} images to f{nrows} rows and f{ncols} columns')
    figsize = (ncols * scale, nrows * scale)
    _, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis("off")
    return axes

import unittest
import pandas as pd

class TestBaseDataset(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'file_path':[1,2,3,4,5,6]})
        self.ds = BaseDataset(self.df)

    def test_split(self):
        a, b = self.ds.split(0.5)
        self.assertEqual(len(a), 3)
        self.assertEqual(len(b), 3)
        self.assertEqual(a.df['file_path'].tolist(), [6, 3, 2])

        c, d = self.ds.split(0.5)
        self.assertTrue(c.df.equals(a.df))
        self.assertTrue(b.df.equals(d.df))

        rets = self.ds.split([0.2, 0.3, 0.4])
        self.assertEqual(len(rets), 4)
        self.assertEqual(len(rets[0]), 1)
        self.assertEqual(len(rets[1]), 2)
        self.assertEqual(len(rets[2]), 2)
        self.assertEqual(len(rets[3]), 1)

    def test_merge(self):
        rets = self.ds.split([0.3, 0.4], shuffle=False)
        ds = rets[0].merge(*rets[1:])
        self.assertTrue(ds.df['file_path'].equals(self.ds.df['file_path']))


    def test_add(self):
        @BaseDataset.add
        def test():
            return BaseDataset(self.df)

        BaseDataset.add('test2', BaseDataset, [self.df])

    def test_get(self):
        self.assertTrue(BaseDataset.get('test').df.equals(self.df))
        self.assertTrue(BaseDataset.get('test2').df.equals(self.df))

    def test_list(self):
        self.assertEqual(BaseDataset.list(), ['test', 'test2'])


    def test_from_df_func(self):
        ds = BaseDataset.from_df_func(None, lambda reader: self.df)
        self.assertTrue(ds.df.equals(self.df))

class TestClassificationDataset(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'class_name':[1,2,3,1,2,3]})
        self.ds = ClassificationDataset(self.df)

    def test_split(self):
        a, b = self.ds.split(0.8)
        self.assertEqual(b.df['class_name'].tolist(), [1,2])
        self.assertEqual(a.classes, [1,2,3])
        self.assertEqual(b.classes, [1,2,3])




if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

