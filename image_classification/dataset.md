# The `Dataset` class
:label:`image_classification_api`

```eval_rst

.. currentmodule:: d8.image_classification

.. autoclass:: Dataset

```


## Adding and Getting Datasets

The following functions list, add and get these datasets.

```eval_rst

.. autosummary::

   Dataset.list
   Dataset.get
   Dataset.add

```


## Constructing a Dataset

We often construct a new dataset using one of the following class methods.

```eval_rst

.. autosummary::

   Dataset.from_folders
   Dataset.from_label_func
   Dataset.from_df_func

```


## Manipulating a Dataset

These functions manipulate a dataset.

```eval_rst

.. autosummary::

   Dataset.split
   Dataset.merge

```


## Visualizing a Dataset

These functions let you have a peak about a dataset.

```eval_rst

.. autosummary::

   Dataset.show
   Dataset.summary
   Dataset.summary_all

```


## Converting Formats

These functions export a `d8` dataset into dataset formats for various libraries.

```eval_rst

.. autosummary::

   Dataset.to_mxnet

```


## `Dataset`

```eval_rst

.. autoclass:: d8.image_classification.Dataset
   :members:
   :show-inheritance:
   :inherited-members:

```


```{.python .input  n=6}
#@save_all
#@hide_all
import pathlib
import pandas as pd
from matplotlib import pyplot as plt
from typing import Union, Sequence, Callable
import fnmatch
import numpy as np
import unittest

from d8 import base_dataset
```

```{.python .input  n=8}
class Dataset(base_dataset.ClassificationDataset):
    TYPE = 'image_classification'

    def show(self, layout=(2,8)) -> None:
        """Show several random examples with their labels.

        :param layout: A tuple of (number of rows, number of columns).
        """
        nrows, ncols = layout
        max_width=300
        scale = 14 / ncols
        figsize = (ncols * scale, nrows * scale)
        _, axes = plt.subplots(nrows, ncols, figsize=figsize)
        samples = self.df.sample(n=nrows*ncols, random_state=0)
        for ax, (_, sample) in zip(axes.flatten(), samples.iterrows()):
            ax.set_title(sample['class_name'])
            img = self.reader.read_image(sample['file_path'], max_width=max_width)
            ax.imshow(img)
            ax.axis("off")

    def summary(self) -> pd.DataFrame:
        """Returns a summary about this dataset."""
        path = self._get_summary_path()
        if path and path.exists(): return pd.read_pickle(path)
        get_mean_std = lambda col: f'{col.mean():.1f} ± {col.std():.1f}'
        img_df = self.reader.get_image_info(self.df['file_path'])
        summary = pd.DataFrame([{'# images':len(img_df),
                                 '# classes':len(self.classes),
                                 'image width':get_mean_std(img_df['width']),
                                 'image height':get_mean_std(img_df['height']),
                                 'size (GB)':img_df['size (KB)'].sum()/2**20,}])
        if path and path.parent.exists(): summary.to_pickle(path)
        return summary

    def __getitem__(self, idx):
        if idx < 0 or idx > self.__len__():
            raise IndexError(f'index {idx} out of range [0, {self.__len__()})')
        file_path = self.df['file_path'][idx]
        img = self.reader.read_image(file_path)
        return np.array(img), self.df['class_name'][idx]

    def to_mxnet(self):
        """Returns a MXNet dataset instance"""
        import mxnet as mx

        class MXDataset(mx.gluon.data.Dataset):
            def __init__(self, dataset):
                self.data = dataset
                self.label_to_idx = {n:i for i, n in enumerate(self.data.classes)}
                self.classes = dataset.classes

            def __getitem__(self, idx):
                file_path = self.data.df['file_path'][idx]
                img = self.data.reader.read_image(file_path)
                img = mx.nd.array(img)
                label = self.label_to_idx[self.data.df['class_name'][idx]]
                return img, label

            def __len__(self):
                return len(self.data.df)
        return MXDataset(self)

    @classmethod
    def from_folders(cls, data_path: str, folders: Union[str, Sequence[str]]) -> 'Dataset':
        """Create a dataset when images from the same class are stored in the same folder.

        :param data_path: Either a URL or a local path. For the former, data will be downloaded automatically.
        :param folders: The folders containing all example images.
        :return: The created dataset.
        """
        if isinstance(folders, (str, pathlib.Path)): folders = [folders]
        def label_func(file_path):
            for folder in folders:
                if fnmatch.fnmatch(str(file_path.parent.parent), folder):
                    return file_path.parent.name
            return None
        return cls.from_label_func(data_path, label_func)

    @classmethod
    def from_label_func(cls, data_path: str,
                        label_func: Callable[[pathlib.Path], str]) -> 'Dataset':
        """Create a dataset from a function that maps a image path to its class name.

        :param data_path: Either a URL or a local path. For the former, data will be downloaded automatically.
        :param label_func: A function takes an image path (an instance :class:`pathlib.Path`) to return a string class name or a None to skip this image.
        :return: The created dataset.
        :param data_path:
        """
        def get_df(reader):
            entries = []
            for file_path in reader.list_images():
                lbl = label_func(file_path)
                if lbl: entries.append({'file_path':file_path, 'class_name':lbl})
            return pd.DataFrame(entries)
        return cls.from_df_func(data_path, get_df)

```

```{.python .input}
class TestDataset(unittest.TestCase):

    def test_from_folders(self):
        Dataset.add('chessman_test', Dataset.from_folders,
                     ['kaggle://niteshfre/chessman-image-dataset', '*'])
        ds = Dataset.get('chessman_test')
        self.assertEqual(len(ds.df), 552)
        self.assertEqual(ds.classes, ['Bishop', 'King', 'Knight', 'Pawn', 'Queen', 'Rook'])
        items = ds[10]
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].shape[2], 3)
```

```{.python .input}
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
```
