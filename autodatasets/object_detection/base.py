import dataclasses
import pathlib
import pandas as pd
import random
from matplotlib import pyplot as plt
import PIL
from typing import Union, Tuple, Callable, List, Any, Optional
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import logging

from .. import data_downloader
from .. import data_reader
from .. import dataset

@dataclasses.dataclass
class Label:
    filepath: str
    classname: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def project(self):
        self.xmin = max(0, self.xmin)
        self.ymin = max(0, self.ymin)
        self.xmax = min(1.0, self.xmax)
        self.ymax = min(1.0, self.ymax)

    def is_valid(self) -> bool:
        if not (0 <= self.xmin <= 1 and 0 <= self.ymin <= 1 and
                self.xmin <= self.xmax <= 1 and self.ymin <= self.ymax <= 1):
            return False
        return True

def _parse_voc_annotation(xml_fp, image_dir) -> List[Label]:
    root = ET.parse(xml_fp).getroot()
    size = root.find('size')
    filepath = pathlib.Path(image_dir)/root.find('filename').text.strip()
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    labels = []
    for obj in root.iter('object'):
        classname = obj.find('name').text.strip().lower()
        xml_box = obj.find('bndbox')
        xmin = float(xml_box.find('xmin').text) / width
        ymin = float(xml_box.find('ymin').text) / height
        xmax = float(xml_box.find('xmax').text) / width
        ymax = float(xml_box.find('ymax').text) / height
        label = Label(str(filepath), classname, xmin, ymin, xmax, ymax)
        label.project()
        if not label.is_valid():
            logging.warning(f'Invalid label {label}')
        labels.append(label)
    return labels

def parse_voc(reader, image_dir, annotation_dir):
    entries = []
    xmls = reader.get_files(['.xml'], [annotation_dir])
    imgs = set(reader.get_images([image_dir]))
    for xml in xmls:
        labels = _parse_voc_annotation(reader.open(xml), image_dir)
        if labels:
            if not pathlib.Path(labels[0].filepath) in imgs:
                logging.warning(f'Not found {labels[0].filepath }')
            else:
                entries.extend(labels)
    return pd.DataFrame(entries)

DATAROOT = pathlib.Path.home()/'.autodatasets'

class Dataset(dataset.BaseDataset):

    def show(self, layout=(4,2), scale=None):
        nrows, ncols = layout
        if not scale:
            scale = 8 / ncols
        figsize = (ncols * scale, nrows * scale)
        _, axes = plt.subplots(nrows, ncols, figsize=figsize)
        random.seed(0)
        samples = random.sample(list(self._train_df.groupby('filepath')), nrows*ncols)
        classes = list(self._train_df['classname'].unique())
        colors = ['b', 'g', 'r', 'm', 'c']
        class_to_color = {c:colors[i%len(colors)] for i, c in enumerate(classes)}
        for ax, sample in zip(axes.flatten(), samples):
            img = PIL.Image.open(self._reader.open(sample[0]))
            ax.imshow(img, aspect='auto')
            img_width, img_height = img.size
            ax.axis("off")
            for _, row in sample[1].iterrows():
                bbox = plt.Rectangle(
                    xy=(row['xmin']*img_width, row['ymin']*img_height),
                    width=(row['xmax']-row['xmin'])*img_width,
                    height=(row['ymax']-row['ymin'])*img_height,
                    fill=False, edgecolor=class_to_color[row['classname']], linewidth=2)
                ax.add_patch(bbox)
                text_color = 'w' #if color == 'w' else 'w'
                ax.text(bbox.xy[0], bbox.xy[1], row['classname'],
                      va='center', ha='center', fontsize=7, color=text_color,
                      bbox=dict(facecolor=class_to_color[row['classname']],
                                lw=0, alpha=1, pad=2))

    def summary(self):
        dfs = dict()
        get_dfs = lambda df: (df, data_reader.get_image_info(self._reader, df['filepath'].unique()))
        if self._train_df is not None: dfs['train'] = get_dfs(self._train_df)
        if self._valid_df is not None: dfs['valid'] = get_dfs(self._valid_df)
        if self._test_df is not None: dfs['test'] = get_dfs(self._test_df)
        def _get_mean_std(col):
            return f'{col.mean():.1f} ± {col.std():.1f}'

        summary = []
        for lbl_df, img_df in dfs.values():
            merged_df = pd.merge(lbl_df, img_df, on='filepath')
            summary.append({'# images':len(img_df),
                    '# bboxes':len(lbl_df),
                    '# classes':lbl_df['classname'].nunique(),
                    'image width':_get_mean_std(img_df['width']),
                    'image height':_get_mean_std(img_df['height']),
                    'bbox width':_get_mean_std((merged_df['xmax']-merged_df['xmin'])*merged_df['width']),
                    'bbox height':_get_mean_std((merged_df['ymax']-merged_df['ymin'])*merged_df['height']),
                    'size (GB)':img_df['size (KB)'].sum()/2**20,
                })
        return pd.DataFrame(summary, index=dfs.keys())

TASK = 'object_detection'

def add_dataset(name: str, func, args=[]):
    dataset.add_dataset((TASK, name), func, args)

def get_dataset(name: str):
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