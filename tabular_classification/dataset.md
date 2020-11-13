# The `Dataset` API

```{.python .input  n=1}
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

```{.python .input  n=2}
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
        filenames = [p.replace('#','/').split('/')[-1]
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
        s = super().summary()
        s.insert(1, '#classes', self.df[self.label].nunique())
        return s

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
     'url'     : 'kaggle://titanic#train.csv',
     'label'   : 1},
    {'name'    : 'credit-card-fraud',
     'url'     : 'kaggle://mlg-ulb/creditcardfraud#creditcard.csv',
     'label'   : -1},
    {'name'    : 'mushroom',
     'url'     : 'kaggle://uciml/mushroom-classification#mushrooms.csv',
     'label'   : 0},
    {'name'    : 'glass',
     'url'     : 'kaggle://uciml/glass#glass.csv',
     'label'   : -1},
    {'name'    : 'mobile-price',
     'url'     : 'kaggle://iabhishekofficial/mobile-price-classification#train.csv',
     'label'   : 'price_range'},
    {'name'    : 'fetal-health',
     'url'     : 'kaggle://andrewmvd/fetal-health-classification#fetal_health.csv',
     'label'   : -1},
    {'name'    : 'drug',
     'url'     : 'kaggle://prathamtripathi/drug-classification#drug200.csv',
     'label'   : -1},
    {'name'    : 'asteroids',
     'url'     : 'kaggle://shrutimehta/nasa-asteroids-classification#nasa.csv',
     'label'   : -1},
    {'name'    : 'taekwondo',
     'url'     : 'kaggle://ali2020armor/taekwondo-techniques-classification#Taekwondo_Technique_Classification_Stats.csv',
     'label'   : 0},
    {'name'    : 'cs-go',
     'url'     : 'kaggle://christianlillelund/csgo-round-winner-classification#csgo_round_snapshots.csv',
     'label'   : -1},
    {'name'    : 'wine',
     'url'     : 'kaggle://uciml/red-wine-quality-cortez-et-al-2009#winequality-red.csv',
     'label'   : -1},
    {'name'    : 'churn',
     'url'     : 'kaggle://shrutimechlearn/churn-modelling#Churn_Modelling.csv',
     'label'   : -1},
    {'name'    : 'usp_drug',
     'url'     : 'kaggle://danofer/usp-drug-classification#usp_drug_classification.csv',
     'label'   : 0},
    {'name'    : 'rain_au',
     'url'     : 'kaggle://jsphyg/weather-dataset-rattle-package#weatherAUS.csv',
     'label'   : -1},
    {'name'    : 'automobile-customer',
     'url'     : 'kaggle://kaushiksuresh147/customer-segmentation#Train.csv',
     'label'   : -1},
#     {'name'    : 'belgium_population',
#      'url'     : 'kaggle://sameerkulkarni91/belgium-population-classification#BELGIUM_POPULATION_STRUCTURE_2018.csv',
#      'label'   : -1},
    {'name'    : 'loan',
     'url'     : 'kaggle://burak3ergun/loan-data-set#loan_data_set.csv',
     'label'   : -1},
    {'name'    : 'crime',
     'url'     : 'kaggle://abidaaslam/crime#Crime1.csv',
     'label'   : 'Category'},
    {'name'    : 'toddler-autism',
     'url'     : 'kaggle://fabdelja/autism-screening-for-toddlers#Toddler Autism dataset July 2018.csv',
     'label'   : -1},
    #  "Financial Distress" if it is greater than -0.50 the company should be considered as healthy (0). Otherwise, it would be regarded as financially distressed (1).
#     {'name'    : 'financial-distress',
#      'url'     : 'kaggle://shebrahimi/financial-distress#Financial Distress.csv',
#      'label'   : 'Financial Distress'},
    {'name'    : '',
     'url'     : 'kaggle://',
     'label'   : ''},
    {'name'    : '',
     'url'     : 'kaggle://',
     'label'   : ''},
    {'name'    : '',
     'url'     : 'kaggle://',
     'label'   : ''},
    {'name'    : '',
     'url'     : 'kaggle://',
     'label'   : ''},


]


for x in from_csv_meta:
    Dataset.add(x['name'], Dataset.from_csv, [x['url'], x['label'], x.get('columns', None), x.get('df_func', None)])

def show(name):
    ds = Dataset.get(name)
    return ds.df.head()

name = [x['name'] for x in from_csv_meta if x['name']]
ds = Dataset.get(name[-1])
#ds = Dataset.get('adult')
ds.summary()
#show('titanic')
```

```{.json .output n=2}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "/Users/mli/.d8/toddler-autism/Toddler%20Autism%20dataset%20July%202018.csv\n"
 },
 {
  "ename": "FileNotFoundError",
  "evalue": "[Errno 2] No such file or directory: '/Users/mli/.d8/toddler-autism/Toddler Autism dataset July 2018.csv'",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
   "\u001b[0;32m<ipython-input-2-61ff23085e15>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m    135\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    136\u001b[0m \u001b[0mname\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'name'\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mx\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mfrom_csv_meta\u001b[0m \u001b[0;32mif\u001b[0m \u001b[0mx\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'name'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 137\u001b[0;31m \u001b[0mds\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mDataset\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mname\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m-\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    138\u001b[0m \u001b[0;31m#ds = Dataset.get('adult')\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    139\u001b[0m \u001b[0mds\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msummary\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/repos/d8/d8/base_dataset.py\u001b[0m in \u001b[0;36mget\u001b[0;34m(cls, name)\u001b[0m\n\u001b[1;32m    111\u001b[0m         \u001b[0;32mwith\u001b[0m \u001b[0mdata_downloader\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mNameContext\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mname\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    112\u001b[0m             \u001b[0;34m(\u001b[0m\u001b[0mfn\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mfn_args\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mfn_kwargs\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcls\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_DATASETS\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcls\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mTYPE\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mname\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 113\u001b[0;31m             \u001b[0mds\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mfn\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0mfn_args\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mfn_kwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    114\u001b[0m             \u001b[0mds\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mname\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mname\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    115\u001b[0m             \u001b[0;32mreturn\u001b[0m \u001b[0mds\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m<ipython-input-2-61ff23085e15>\u001b[0m in \u001b[0;36mfrom_csv\u001b[0;34m(cls, data_path, label, names, df_func)\u001b[0m\n\u001b[1;32m     18\u001b[0m         filenames = [p.replace('#','/').split('/')[-1]\n\u001b[1;32m     19\u001b[0m                      for p in base_dataset.listify(data_path)]\n\u001b[0;32m---> 20\u001b[0;31m         \u001b[0mdfs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0mpd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mread_csv\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mreader\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mopen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mf\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mheader\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mheader\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnames\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mnames\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mf\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mfilenames\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     21\u001b[0m         \u001b[0mdf\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mdfs\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;32mif\u001b[0m \u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdfs\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;36m1\u001b[0m \u001b[0;32melse\u001b[0m \u001b[0mpd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mconcat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdfs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mignore_index\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     22\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mdf_func\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mdf\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mdf_func\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdf\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m<ipython-input-2-61ff23085e15>\u001b[0m in \u001b[0;36m<listcomp>\u001b[0;34m(.0)\u001b[0m\n\u001b[1;32m     18\u001b[0m         filenames = [p.replace('#','/').split('/')[-1]\n\u001b[1;32m     19\u001b[0m                      for p in base_dataset.listify(data_path)]\n\u001b[0;32m---> 20\u001b[0;31m         \u001b[0mdfs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0mpd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mread_csv\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mreader\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mopen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mf\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mheader\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mheader\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnames\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mnames\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mf\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mfilenames\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     21\u001b[0m         \u001b[0mdf\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mdfs\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;32mif\u001b[0m \u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdfs\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;36m1\u001b[0m \u001b[0;32melse\u001b[0m \u001b[0mpd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mconcat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdfs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mignore_index\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     22\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mdf_func\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mdf\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mdf_func\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdf\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/repos/d8/d8/data_reader.py\u001b[0m in \u001b[0;36mopen\u001b[0;34m(self, path)\u001b[0m\n\u001b[1;32m    146\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    147\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mopen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mpath\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mUnion\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mstr\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mpathlib\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mPath\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 148\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_root\u001b[0m\u001b[0;34m/\u001b[0m\u001b[0mpath\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mopen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'rb'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    149\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    150\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_list_all\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/miniconda3/lib/python3.7/pathlib.py\u001b[0m in \u001b[0;36mopen\u001b[0;34m(self, mode, buffering, encoding, errors, newline)\u001b[0m\n\u001b[1;32m   1174\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_raise_closed\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1175\u001b[0m         return io.open(self, mode, buffering, encoding, errors, newline,\n\u001b[0;32m-> 1176\u001b[0;31m                        opener=self._opener)\n\u001b[0m\u001b[1;32m   1177\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1178\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mread_bytes\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/miniconda3/lib/python3.7/pathlib.py\u001b[0m in \u001b[0;36m_opener\u001b[0;34m(self, name, flags, mode)\u001b[0m\n\u001b[1;32m   1028\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_opener\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mname\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mflags\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmode\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m0o666\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1029\u001b[0m         \u001b[0;31m# A stub for the opener argument to built-in open()\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1030\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_accessor\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mopen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mflags\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmode\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1031\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1032\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_raw_open\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mflags\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmode\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m0o777\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: '/Users/mli/.d8/toddler-autism/Toddler Autism dataset July 2018.csv'"
  ]
 }
]
```
