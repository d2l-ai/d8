# This file is generated from image_classification/dataset.md automatically through:
#    d2lbook build lib
# Don't edit it directly

#@save_all
#@hide_all
import pathlib
import pandas as pd
from matplotlib import pyplot as plt
from typing import Union, Sequence, Callable, Optional
import fnmatch
import numpy as np
import unittest

from d8 import core

class Dataset(core.BaseDataset):
    """The class of an image classification dataset."""
    def __init__(self, df: pd.DataFrame, reader: core.Reader):
        super().__init__(df, reader, label_name='class_name')

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
            class_name = sample['class_name']
            if 'confidence' in sample:
                # add confidence to class_name if available
                class_name += f': {float(sample["confidence"]):.2f}'
            ax.set_title(class_name)
            img = self.reader.read_image(sample['file_path'], max_width=max_width)
            ax.imshow(img)
            ax.axis("off")

    def _summary(self) -> pd.DataFrame:
        """Returns a summary about this dataset."""
        get_mean_std = lambda col: f'{col.mean():.1f} ± {col.std():.1f}'
        img_df = self.reader.get_image_info(self.df['file_path'])
        return pd.DataFrame([{'# images':len(img_df),
                                 '# classes':len(self.classes),
                                 'image width':get_mean_std(img_df['width']),
                                 'image height':get_mean_std(img_df['height']),
                                 'size (GB)':img_df['size (KB)'].sum()/2**20,}])

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
    def from_folders(cls, data_path: Union[str, Sequence[str]],
                     folders: Union[str, Sequence[str]]) -> 'Dataset':
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
    def from_label_func(cls, data_path: Union[str, Sequence[str]],
                        label_func: Callable[[pathlib.Path], str]) -> 'Dataset':
        """Create a dataset from a function that maps a image path to its class name.

        :param data_path: Either a URL or a local path. For the former, data will be downloaded automatically.
        :param label_func: A function takes an image path (an instance :class:`pathlib.Path`) to return a string class name or a None to skip this image.
        :return: The created dataset.
        :param data_path:
        """
        reader = core.create_reader(data_path)
        entries = []
        for file_path in reader.list_images():
            lbl = label_func(file_path)
            if lbl: entries.append({'file_path':file_path, 'class_name':lbl})
        df = pd.DataFrame(entries)
        return cls(df, reader)


class TestDataset(unittest.TestCase):

    def test_from_folders(self):
        Dataset.add('chessman_test', Dataset.from_folders,
                     ['https://www.kaggle.com/niteshfre/chessman-image-dataset', '*'])
        ds = Dataset.get('chessman_test')
        self.assertEqual(len(ds.df), 552)
        self.assertEqual(ds.classes, ['Bishop', 'King', 'Knight', 'Pawn', 'Queen', 'Rook'])
        items = ds[10]
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].shape[2], 3)

    def test_from_label_func(self):
        name = 'test-honey-bee'
        Dataset.add(name, Dataset.from_label_func,
                     ['https://www.kaggle.com/jenny18/honey-bee-annotated-images',
                      lambda path: path.name.split('_')[0]])
        ds = Dataset.get(name)
        self.assertEqual(len(ds.df), 5172)
        self.assertEqual(len(ds.classes), 45)





if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
