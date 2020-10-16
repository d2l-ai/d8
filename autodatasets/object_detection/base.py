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

# TYPE = 'image'
# TASK = 'detection'

@dataclasses.dataclass
class Label:
    filepath: str
    classname: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def is_valid(self) -> bool:
        if not (0 <= self.xmin <= 1 and 0 <= self.ymin <= 1 and
                self.xmin <= self.xmax <= 1 and self.ymin <= self.ymax <= 1):
            return False
        return True


# @dataclasses.dataclass
# class DatasetInfo:
#     name: str
#     # description: str
#     url: str
#     train_fn: Tuple[Callable, Tuple[Any]]
#     valid_fn: Optional[Tuple[Callable, Tuple[Any]]] = None
#     test_fn: Optional[Tuple[Callable, Tuple[Any]]] = None

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


# _MAKEML_DATA_URL = 'https://arcraftimages.s3-accelerate.amazonaws.com/Datasets/'

# DATASETS = [
#     DatasetInfo('raccoon',
#                 'Detect raccoons',
#                 _MAKEML_DATA_URL+'Raccoon/RaccoonPascalVOC.zip',
#                 (parse_voc, ('images', 'annotations'))
#     ),
#     DatasetInfo('paperprototype',
#                 'Detect elements in handraw papers',
#                 _MAKEML_DATA_URL+'PaperPrototype/PaperPrototypePascalVOC.zip',
#                 (parse_voc, ('images', 'annotations'))
#     ),
#     DatasetInfo('tomato',
#                 'Detect tomatos',
#                 _MAKEML_DATA_URL+'Tomato/TomatoPascalVOC.zip',
#                 (parse_voc, ('images', 'annotations'))
#     ),
# ]

DATAROOT = pathlib.Path.home()/'.autodatasets'

# def _get_df(name: str):
#     datasets = {ds.name:ds for ds in DATASETS}
#     if name not in datasets:
#         raise ValueError(f'{name} is not one of {list(datasets.keys())}')
#     ds = datasets[name]
#     filepaths = data_downloader.download(ds.url, ROOT/name)
#     _get_df = lambda fn: fn[0](filepaths, *fn[1]) if fn else None
#     return filepaths, _get_df(ds.train_fn), _get_df(ds.valid_fn), _get_df(ds.test_fn)



class Dataset:
    def __init__(self, name: str, reader: data_reader,
                 train_df: Optional[pd.DataFrame] = None,
                 valid_df: Optional[pd.DataFrame] = None,
                 test_df: Optional[pd.DataFrame] = None):
        self._name = name
        self._reader = reader
        self._train_df = train_df
        self._valid_df = valid_df
        self._test_df = test_df

    # #def __init__(self, name: Union[str, DatasetInfo]):
    #     ds = self._get_ds_info(name)
    #     filepaths = data_downloader.download(ds.url, _ROOT/ds.name)
    #     self._reader = data_reader.create_reader(filepaths)
    #     _get_df = lambda fn: fn[0](self._reader, *fn[1]) if fn else None
    #     self._train_df = _get_df(ds.train_fn)
    #     self._valid_df = _get_df(ds.valid_fn)
    #     self._test_df = _get_df(ds.test_fn)

    # def _get_ds_info(self, name):
    #     if isinstance(name, DatasetInfo):
    #         return name
    #     elif isinstance(name, str):
    #         datasets = {ds.name:ds for ds in DATASETS}
    #         if name not in datasets:
    #             raise ValueError(f'{name} is not one of {list(datasets.keys())}')
    #         return datasets[name]
    #     raise TypeError(f'str or DatasetInfo')

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
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
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
        summary = [{'# images':len(dfs[name][1]),
                '# bounding boxes':len(dfs[name][0]),
                '# classes':dfs[name][0]['classname'].unique().size,
            'size (GB)':dfs[name][1]['img_size(KB)'].sum()/2**20
                } for name in dfs]
        return pd.DataFrame(summary, index=dfs.keys())

_DATASETS = dict()
def add_dataset(name: str, func, args=[]):
    _DATASETS[name] = (func, args)

def get_dataset(name: str):
    return _DATASETS[name][0](*_DATASETS[name][1])

def list_datasets():
    return list(_DATASETS.keys())