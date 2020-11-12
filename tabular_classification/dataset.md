# The `Dataset` API

```{.python .input}
import pathlib
import pandas as pd
from matplotlib import pyplot as plt
from typing import Union, Sequence, Callable
import fnmatch
import numpy as np
import unittest

from d8 import base_dataset
from d8 import data_reader
```

```{.python .input}
class TabularDataset(base_dataset.BaseDataset):
    def __init__(self,
                 df: pd.DataFrame,
                 reader: data_reader.Reader,
                 label: str = -1,
                 name: str = '') -> None:
        super().__init__(df, reader, name)
        self.label = label 
        
    @classmethod
    def from_csv(cls, data_path: Union[str, Sequence[str]], label, names=None) -> 'Dataset':
        header = 0 if names else 'infer'
        reader = cls.create_reader(data_path)
        filenames = [p.replace(':','/').split('/')[-1] 
                     for p in base_dataset.listify(data_path)]
        dfs = [pd.read_csv(reader.open(f), header=header, names=names) for f in filenames]
        df = dfs[0] if len(dfs) == 1 else pd.concat(dfs, axis=0, ignore_index=True)        
        return cls(df, reader, label)

class Dataset(TabularDataset):
    TYPE = 'tabular_classification'

_UCI = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'

from_csv_meta = [
    {'name'    : 'iris',
     'url'     : _UCI+'iris/iris.data',
     'label'   : -1,
     'columns' : ['sepal length', 'sepal width', 'petal length', 'petal width','class']},
    {'name'    : 'adult',     
     'url'     : [_UCI+'adult/adult.data', _UCI+'adult/adult.test'],
     'label'   : -1,
     'columns' : ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary']},
    {'name'    : 'titanic',
     'url'     : 'kaggle:titanic:train.csv',
     'label'   : 1},
]
    
    
for x in from_csv_meta:
    Dataset.add(x['name'], Dataset.from_csv, [x['url'], x['label'], x.get('columns', None)])
    
def show(name):
    ds = Dataset.get(name) 
    return ds.df.head()

show('titanic')
```
