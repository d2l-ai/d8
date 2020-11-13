# The `Dataset` API

```{.python .input  n=1}
#@save_all
import pathlib
import pandas as pd
from typing import Union, Sequence, Callable
import fnmatch
import numpy as np

from d8 import core
```

```{.python .input  n=2}

def read_csv(data_path: Union[str, Sequence[str]], label, columns=None):
    header = 0 if columns else 'infer'
    reader = core.BaseDataset.create_reader(data_path)
    filenames = [p.replace('#','/').replace('?select=','/').replace('+',' ').split('/')[-1]
                 for p in core.listify(data_path)]
    dfs = [pd.read_csv(reader.open(f), header=header, names=columns) for f in filenames]
    df = dfs[0] if len(dfs) == 1 else pd.concat(dfs, axis=0, ignore_index=True)
    return df, reader


class Dataset(core.ClassificationDataset):
    def __init__(self, df: pd.DataFrame, reader: core.Reader,
                 label: Union[int, str]) -> None:
        super().__init__(df, reader, label)

    TYPE = 'tabular_classification'

    @classmethod
    def from_csv(cls, data_path: Union[str, Sequence[str]], label, columns=None, df_func=None) -> 'Dataset':
        df, reader = read_csv(data_path, label, columns)
        if df_func: df = df_func(df)
        return cls(df, reader, label)

    def summary(self):
        path = self._get_summary_path()
        if path and path.exists(): return pd.read_pickle(path)
        numeric_cols = len(self.df.drop(self.label, axis=1).select_dtypes('number').columns)
        s = pd.DataFrame([{'#examples':len(self.df),
                               '#classes':len(self.classes),
                               '#numeric_features':numeric_cols,
                               '#category_features':len(self.df.columns) - 1 - numeric_cols,
                               'size(MB)':self.df.memory_usage().sum()/2**20,}])
        if path and path.parent.exists(): s.to_pickle(path)
        return s
```

```{.python .input}
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
        
```

```{.python .input}
%load_ext mypy_ipython
%mypy 

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
```
