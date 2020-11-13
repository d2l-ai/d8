# This file is generated from tabular_classification/dataset.md automatically through:
#    d2lbook build lib
# Don't edit it directly

#@save_all
import pathlib
import pandas as pd
from typing import Union, Sequence, Callable
import fnmatch
import numpy as np
import unittest

from d8 import base_dataset
from d8 import data_reader

class TabularDataset(base_dataset.BaseDataset):
    def __init__(self,
                 df: pd.DataFrame,
                 reader: data_reader.Reader,
                 label: str = -1,
                 name: str = '') -> None:
        super().__init__(df, reader, name)
        if isinstance(label, int):
            label = df.columns[label]
        if label not in df.columns:
            raise ValueError(f'Label {label} is not in {df.columns}')
        self.label = label

    @classmethod
    def from_csv(cls, data_path: Union[str, Sequence[str]], label, names=None, df_func=None) -> 'Dataset':
        header = 0 if names else 'infer'
        reader = cls.create_reader(data_path)
        filenames = [p.replace('#','/').replace('?select=','/').replace('+',' ').split('/')[-1]
                     for p in base_dataset.listify(data_path)]
        dfs = [pd.read_csv(reader.open(f), header=header, names=names) for f in filenames]
        df = dfs[0] if len(dfs) == 1 else pd.concat(dfs, axis=0, ignore_index=True)
        if df_func: df = df_func(df)
        return cls(df, reader, label)

    def summary(self):
        
        numeric_cols = len(self.df.drop(self.label, axis=1).select_dtypes('number').columns)
        return  pd.DataFrame([{'#examples':len(self.df),
                               '#numeric_features':numeric_cols,
                               '#category_features':len(self.df.columns) - 1 - numeric_cols,
                                'size(MB)':self.df.memory_usage().sum()/2**20,}])

class Dataset(TabularDataset):
    TYPE = 'tabular_classification'

    def summary(self):
        path = self._get_summary_path()
        if path and path.exists(): return pd.read_pickle(path)
        s = super().summary()
        s.insert(1, '#classes', self.df[self.label].nunique())
        if path and path.parent.exists(): s.to_pickle(path)
        return s

