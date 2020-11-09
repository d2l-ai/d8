import pathlib
import pandas as pd
from matplotlib import pyplot as plt
from typing import Union, Sequence, Callable
import fnmatch
import numpy as np

from .. import base_dataset

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
            ax.set_title(sample['classname'])
            img = self.reader.read_image(sample['filepath'], max_width=max_width)
            ax.imshow(img)
            ax.axis("off")

    def summary(self) -> pd.DataFrame:
        """Returns a summary about this dataset."""
        path = self._get_summary_path()
        if path and path.exists(): return pd.read_pickle(path)
        get_mean_std = lambda col: f'{col.mean():.1f} ± {col.std():.1f}'
        img_df = self.reader.get_image_info(self.df['filepath'])
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
        filepath = self.df['filepath'][idx]
        img = self.reader.read_image(filepath)
        return np.array(img), self.df['classname'][idx]

    def to_mxnet(self):
        """Returns a MXNet dataset instance"""
        import mxnet as mx

        class MXDataset(mx.gluon.data.Dataset):
            def __init__(self, dataset):
                self.data = dataset
                self.label_to_idx = {n:i for i, n in enumerate(self.data.classes)}
                self.classes = dataset.classes

            def __getitem__(self, idx):
                filepath = self.data.df['filepath'][idx]
                img = self.data.reader.read_image(filepath)
                img = mx.nd.array(img)
                label = self.label_to_idx[self.data.df['classname'][idx]]
                return img, label

            def __len__(self):
                return len(self.data.df)
        return MXDataset(self)

    @classmethod
    def from_folders(cls, datapath: str, folders: Union[str, Sequence[str]]) -> 'Dataset':
        """Create a dataset when images from the same class are stored in the same folder.

        :param datapath: Either a URL or a local path. For the former, data will be downloaded automatically.
        :param folders: The folders containing all example images.
        :return: The created dataset.
        """
        if isinstance(folders, (str, pathlib.Path)): folders = [folders]
        def label_func(filepath):
            for folder in folders:
                if fnmatch.fnmatch(str(filepath.parent.parent), folder):
                    return filepath.parent.name
            return None
        return cls.from_label_func(datapath, label_func)

    @classmethod
    def from_label_func(cls, datapath: str,
                        label_func: Callable[[pathlib.Path], str]) -> 'Dataset':
        """Create a dataset from a function that maps a image path to its class name.

        :param datapath: Either a URL or a local path. For the former, data will be downloaded automatically.
        :param label_func: A function takes an image path (an instance :class:`pathlib.Path`) to return a string class name or a None to skip this image.
        :return: The created dataset.
        :param datapath:
        """
        def get_df(reader):
            entries = []
            for filepath in reader.list_images():
                lbl = label_func(filepath)
                if lbl: entries.append({'filepath':filepath, 'classname':lbl})
            return pd.DataFrame(entries)
        return cls.from_df_func(datapath, get_df)

    @classmethod
    def summary_all(cls, quick=False):
        df = super().summary_all(quick)
        return df.sort_values('# images')
