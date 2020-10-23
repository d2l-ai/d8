import pathlib
import pandas as pd
import random
from matplotlib import pyplot as plt
import dataclasses
import collections
import PIL
from typing import Union, Tuple, Callable, List, Any, Optional
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import logging

from .. import base_dataset

@dataclasses.dataclass
class Label:
    filepath: str
    classname: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def project_bbox(self):
        self.xmin = max(0, self.xmin)
        self.ymin = max(0, self.ymin)
        self.xmax = min(1.0, self.xmax)
        self.ymax = min(1.0, self.ymax)

    def is_bbox_valid(self) -> bool:
        if not (0 <= self.xmin <= 1 and 0 <= self.ymin <= 1 and
                self.xmin <= self.xmax <= 1 and self.ymin <= self.ymax <= 1):
            return False
        return True

def parse_voc_annotation(xml_fp) -> List[Label]:
    root = ET.parse(xml_fp).getroot()
    filepath = root.find('filename').text.strip()
    size = root.find('size')
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
        label = Label(filepath, classname, xmin, ymin, xmax, ymax)
        label.project_bbox()
        if not label.is_bbox_valid():
            logging.warning(f'Invalid bounding box {label}')
        else:
            labels.append(label)
    return labels

def _parse_voc(reader, image_dir, annotation_dir):
    entries = []
    annotation_dir = str(pathlib.Path(annotation_dir))
    image_dir = str(pathlib.Path(image_dir))
    # don't use .is_relative_to as it requires python >= 3.9
    xmls = [xml for xml in reader.list_files(['.xml'], [annotation_dir])]
    imgs = set([img for img in reader.list_images([image_dir])])
    for xml in xmls:
        labels = parse_voc_annotation(reader.open(xml))
        if labels:
            image_path = pathlib.Path(image_dir)/labels[0].filepath
            if image_path not in imgs:
                logging.warning(f'Not found image {limage_path}')
            else:
                for l in labels: l.filepath = str(image_path)
                entries.extend(labels)
    return pd.DataFrame(entries)


class Dataset(base_dataset.BaseDataset):
    TYPE = 'object_detection'

    def show(self, layout=(2,4), scale=None, max_width=500):
        nrows, ncols = layout
        if not scale: scale = 10 / ncols
        figsize = (ncols * scale, nrows * scale)
        _, axes = plt.subplots(nrows, ncols, figsize=figsize)
        random.seed(0)
        samples = random.sample(list(self._train_df.groupby('filepath')), nrows*ncols)
        classes = list(self._train_df['classname'].unique())
        colors = ['b', 'g', 'r', 'm', 'c']
        class_to_color = {c:colors[i%len(colors)] for i, c in enumerate(classes)}
        for ax, sample in zip(axes.flatten(), samples):
            img = self._reader.read_image(sample[0], max_width=max_width)
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
                ax.text(bbox.xy[0], bbox.xy[1], row['classname'],
                      va='center', ha='center', fontsize=7, color='w',
                      bbox=dict(facecolor=class_to_color[row['classname']],
                                lw=0, alpha=1, pad=2))

    def summary(self):
        dfs = self.get_dfs()
        get_mean_std = lambda col: f'{col.mean():.1f} ± {col.std():.1f}'
        summary = []
        for lbl_df in dfs.values():
            img_df = self._reader.get_image_info(lbl_df['filepath'])
            merged_df = pd.merge(lbl_df, img_df, on='filepath')
            summary.append({'# images':len(img_df),
                    '# bboxes':len(lbl_df),
                    '# classes':lbl_df['classname'].nunique(),
                    'image width':get_mean_std(img_df['width']),
                    'image height':get_mean_std(img_df['height']),
                    'bbox width':get_mean_std((merged_df['xmax']-merged_df['xmin'])*merged_df['width']),
                    'bbox height':get_mean_std((merged_df['ymax']-merged_df['ymin'])*merged_df['height']),
                    'size (GB)':img_df['size (KB)'].sum()/2**20,
                })
        return pd.DataFrame(summary, index=dfs.keys())

    @classmethod
    def from_voc(cls, datapath: str,
                 train_image_dir: str = None, train_annotation_dir: str = None,
                 valid_image_dir: str = None, valid_annotation_dir: str = None,
                 test_image_dir: str = None, test_annotation_dir: str = None):
        def get_df_func(image_dir, annotation_dir):
            if image_dir is None or annotation_dir is None:
                return None
            return lambda reader: _parse_voc(reader, image_dir, annotation_dir)
        return cls.from_df_func(
            datapath,
            get_df_func(train_image_dir, train_annotation_dir),
            get_df_func(valid_image_dir, valid_annotation_dir),
            get_df_func(test_image_dir, test_annotation_dir))

    @classmethod
    def summary_all(cls):
        df = super().summary_all()
        return df.sort_values('# images')