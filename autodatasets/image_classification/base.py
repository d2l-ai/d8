import pathlib
import pandas as pd
from matplotlib import pyplot as plt
import PIL

from .. import data_reader
from .. import data_downloader
from .. import dataset

class Dataset(dataset.BaseDataset):
    def show(self, layout=(2,10), scale=None):
        nrows, ncols = layout
        if not scale: scale = 14 / ncols
        figsize = (ncols * scale, nrows * scale)
        _, axes = plt.subplots(nrows, ncols, figsize=figsize)
        samples = self._train_df.sample(n=nrows*ncols, random_state=0)
        for ax, (_, sample) in zip(axes.flatten(), samples.iterrows()):
            img = PIL.Image.open(self._reader.open(sample['filepath']))
            ax.set_title(sample['classname'])
            ax.imshow(img)
            ax.axis("off")


    def summary(self):
        get_mean_std = lambda col: f'{col.mean():.1f} ± {col.std():.1f}'
        dfs = self.get_dfs()
        summary = []
        for df in dfs.values():
            img_df = data_reader.get_image_info(self._reader, df['filepath'])

            summary.append({'# images':len(img_df),
                    '# classes':df['classname'].nunique(),
                    'image width':get_mean_std(img_df['width']),
                    'image height':get_mean_std(img_df['height']),
                    'size (GB)':img_df['size (KB)'].sum()/2**20,
                })
        return pd.DataFrame(summary, index=dfs.keys())

def dataset_from_folders(datapath: str, train_dir: str=None, valid_dir: str=None, test_dir: str=None):
    """Create a dataset when images from the same class are stored in the same folder.

    :param datapath: Either a URL or a local path. For the former, data will be downloaded automatically.
    :param train_dir: The folder contains all training images.
    :param valid_dir: The folder contains all validation images.
    :param test_dir: The folder contains all test images.
    :return: The created dataset.
    """
    def get_fn(dir):
        if not dir: return None
        return lambda filepath: (filepath.parent.name if filepath.parent.parent == pathlib.Path(dir) else None)
    return dataset_from_label_functions(datapath, get_fn(train_dir), get_fn(valid_dir), get_fn(test_dir))

def dataset_from_label_functions(datapath, train_fn=None, valid_fn=None, test_fn=None):
    if not pathlib.Path(datapath).exists():
        datapath = data_downloader.download(datapath)
    reader = data_reader.create_reader(datapath)
    images = reader.get_images()
    def get_df(fn):
        if not fn: return None
        entries = []
        for filepath in images:
            lbl = fn(filepath)
            if lbl: entries.append({'filepath':filepath, 'classname':lbl})
        return pd.DataFrame(entries)
    return Dataset('xxx', reader, get_df(train_fn), get_df(valid_fn), get_df(test_fn))

TASK = 'image_classification'

def add_dataset(func):
    """Add a dataset."""
    dataset.add_dataset((TASK, func.__name__), func, [])
    return func

def get_dataset(name: str):
    """get datasets.

    :param name: a name
    """
    return dataset.get_dataset((TASK, name))

def list_datasets():
    return [name for task, name in dataset.list_datasets() if task == TASK]

def summary():
    names = list_datasets()
    datasets = [get_dataset(name) for name in names]
    summary = pd.DataFrame([ds.summary().loc['train'] for ds in datasets], index=names)
    for name in summary:
        if name.startswith('#'):
            summary[name] = summary[name].astype(int)
    return summary.sort_values('# images')