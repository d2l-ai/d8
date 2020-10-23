import pathlib
import pandas as pd
from matplotlib import pyplot as plt

from .. import base_dataset

class Dataset(base_dataset.BaseDataset):
    TYPE = 'image_classification'

    def show(self, layout=(2,8), scale=None, max_width=300):
        nrows, ncols = layout
        if not scale: scale = 14 / ncols
        figsize = (ncols * scale, nrows * scale)
        _, axes = plt.subplots(nrows, ncols, figsize=figsize)
        samples = self._train_df.sample(n=nrows*ncols, random_state=0)
        for ax, (_, sample) in zip(axes.flatten(), samples.iterrows()):
            ax.set_title(sample['classname'])
            img = self._reader.read_image(sample['filepath'], max_width=max_width)
            ax.imshow(img)
            ax.axis("off")

    def summary(self):
        get_mean_std = lambda col: f'{col.mean():.1f} ± {col.std():.1f}'
        dfs = self.get_dfs()
        summary = []
        for df in dfs.values():
            img_df = self._reader.get_image_info(df['filepath'])
            summary.append({'# images':len(img_df),
                    '# classes':df['classname'].nunique(),
                    'image width':get_mean_std(img_df['width']),
                    'image height':get_mean_std(img_df['height']),
                    'size (GB)':img_df['size (KB)'].sum()/2**20,
                })
        return pd.DataFrame(summary, index=dfs.keys())

    @classmethod
    def from_folders(cls, datapath: str, train_dir: str=None, valid_dir: str=None, test_dir: str=None):
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
        return cls.from_label_func(datapath, get_fn(train_dir), get_fn(valid_dir), get_fn(test_dir))

    @classmethod
    def from_label_func(cls, datapath, train_fn=None, valid_fn=None, test_fn=None):
        def get_df_fn(fn):
            if not fn: return None
            def get_df(reader):
                entries = []
                for filepath in reader.list_images():
                    lbl = fn(filepath)
                    if lbl: entries.append({'filepath':filepath, 'classname':lbl})
                return pd.DataFrame(entries)
            return get_df
        return cls.from_df_func(datapath, get_df_fn(train_fn), get_df_fn(valid_fn), get_df_fn(test_fn))

def summary():
    names = Dataset.list()
    datasets = [Dataset.get(name) for name in names]
    summary = pd.DataFrame([ds.summary().loc['train'] for ds in datasets], index=names)
    for name in summary:
        if name.startswith('#'):
            summary[name] = summary[name].astype(int)
    return summary.sort_values('# images')