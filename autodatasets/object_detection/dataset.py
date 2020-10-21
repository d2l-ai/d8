import pathlib
import pandas as pd
import random
from matplotlib import pyplot as plt
import collections
import PIL
from typing import Union, Tuple, Callable, List, Any, Optional
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import logging

from .. import base_dataset

Label = collections.namedtuple('Label', ['filepath', 'classname', 'xmin', 'ymin', 'xmax', 'ymax'])

def _project_bbox(label):
    return Label(label.filepath, label.classname, max(0, label.xmin),
        max(0, label.ymin), min(1.0, label.xmax), min(1.0, label.ymax))

def _is_bbox_valid(label):
    if not (0 <= label.xmin <= 1 and 0 <= label.ymin <= 1 and
            label.xmin <= label.xmax <= 1 and label.ymin <= label.ymax <= 1):
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
        label = _project_bbox(Label(str(filepath), classname, xmin, ymin, xmax, ymax))
        if not _is_bbox_valid(label):
            logging.warning(f'Invalid {label}')
        else:
            labels.append(label)
    return labels

def _parse_voc(reader, image_dir, annotation_dir):
    entries = []
    annotation_dir = pathlib.Path(annotation_dir)
    image_dir = pathlib.Path(image_dir)
    xmls = [xml for xml in reader.list_files(['.xml']) if xml.parent == annotation_dir]
    imgs = set([img for img in reader.list_images() if img.parent == image_dir])
    for xml in xmls:
        labels = _parse_voc_annotation(reader.open(xml), image_dir)
        if labels:
            if not pathlib.Path(labels[0].filepath) in imgs:
                logging.warning(f'Not found {labels[0].filepath }')
            else:
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
            img_height, img_width, _ = img.shape
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