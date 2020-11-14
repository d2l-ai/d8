# This file is generated from tabular_classification/dataset.md automatically through:
#    d2lbook build lib
# Don't edit it directly

#@save_all
import fnmatch
import pathlib
from typing import Callable, Sequence, Union

import numpy as np
import pandas as pd

from d8 import core

__all__ = ['read_csv', 'Dataset']

def read_csv(data_path: Union[str, Sequence[str]], columns=None):
    header = 0 if columns else 'infer'
    reader = core.create_reader(data_path)
    filenames = [p.replace('#','/').replace('?select=','/').replace('+',' ').split('/')[-1]
                 for p in core.listify(data_path)]
    dfs = [pd.read_csv(reader.open(f), header=header, names=columns) for f in filenames]
    df = dfs[0] if len(dfs) == 1 else pd.concat(dfs, axis=0, ignore_index=True)
    return df, reader


class Dataset(core.BaseDataset):
    TYPE = 'tabular_classification'

    @classmethod
    def from_csv(cls, data_path: Union[str, Sequence[str]], label_name, columns=None, df_func=None) -> 'Dataset':
        df, reader = read_csv(data_path, columns)
        if df_func: df = df_func(df)
        return cls(df, reader, label_name)

    def _summary(self):
        numeric_cols = len(self.df.drop(self.label_name, axis=1).select_dtypes('number').columns)
        return pd.DataFrame([{'#examples':len(self.df),
                              '#classes':len(self.classes),
                              '#numeric_features':numeric_cols,
                              '#category_features':len(self.df.columns) - 1 - numeric_cols,
                              'size(MB)':self.df.memory_usage().sum()/2**20,}])

import unittest

class TestDataset(unittest.TestCase):
    def test_from_csv(self):
        name = 'titanic_test'
        for fn in (core.DATAROOT/name).glob('*'): fn.unlink()
        Dataset.add(name, Dataset.from_csv,
                     ['https://www.kaggle.com/c/titanic/data?select=train.csv', -1])
        ds = Dataset.get(name)
        self.assertEqual(len(ds.df), 889)
        self.assertEqual(ds.classes, ['C', 'Q', 'S'])





if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

