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
class BBox:
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

def parse_voc_annotation(xml_fp) -> List[BBox]:
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
        label = BBox(filepath, classname, xmin, ymin, xmax, ymax)
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

class Dataset(base_dataset.ClassificationDataset):
    TYPE = 'object_detection'

    def show(self, layout=(2,4)) -> None:
        """Show several random examples with their labels.

        :param layout: A tuple of (number of rows, number of columns).
        """
        nrows, ncols = layout
        max_width=500
        scale = 10 / ncols
        figsize = (ncols * scale, nrows * scale)
        _, axes = plt.subplots(nrows, ncols, figsize=figsize)
        random.seed(0)
        samples = random.sample(list(self.df.groupby('filepath')), nrows*ncols)
        colors = ['b', 'g', 'r', 'm', 'c']
        class_to_color = {c:colors[i%len(colors)] for i, c in enumerate(self.classes)}
        for ax, sample in zip(axes.flatten(), samples):
            img = self.reader.read_image(sample[0], max_width=max_width)
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
        """Returns a summary about this dataset."""
        path = self._get_summary_path()
        if path and path.exists(): return pd.read_pickle(path)
        get_mean_std = lambda col: f'{col.mean():.1f} ± {col.std():.1f}'
        img_df = self.reader.get_image_info(self.df['filepath'].unique())
        merged_df = pd.merge(self.df, img_df, on='filepath')
        summary = pd.DataFrame([{'# images':len(img_df),
                                 '# bboxes':len(self.df),
                                 '# bboxes / image':get_mean_std(self.df.groupby('filepath')['filepath'].count()),
                                 '# classes':len(self.classes),
                                 'image width':get_mean_std(img_df['width']),
                                 'image height':get_mean_std(img_df['height']),
                                 'bbox width':get_mean_std((merged_df['xmax']-merged_df['xmin'])*merged_df['width']),
                                 'bbox height':get_mean_std((merged_df['ymax']-merged_df['ymin'])*merged_df['height']),
                                 'size (GB)':img_df['size (KB)'].sum()/2**20}])
        if path and path.parent.exists(): summary.to_pickle(path)
        return summary

    @classmethod
    def from_voc(cls, datapath: str,
                 image_folders: str, annotation_folders: str):
        """Create a dataset when data are stored in the VOC format.

        :param datapath: Either a URL or a local path. For the former, data will be downloaded automatically.
        :param folders: The folders containing all example images.
        :return: The created dataset.
        """
        listify = lambda x: x if isinstance(x, (tuple, list)) else [x]

        def get_df_func(image_folders, annotation_folders):
            def df_func(reader):
                dfs = []
                for image_folder, annotation_folder in zip(image_folders, annotation_folders):
                    dfs.append(_parse_voc(reader, image_folder, annotation_folder))
                return pd.concat(dfs, axis=0)
            return df_func
        return cls.from_df_func(datapath, get_df_func(listify(image_folders), listify(annotation_folders)))

    @classmethod
    def summary_all(cls, quick=False):
        df = super().summary_all(quick)
        return df.sort_values('# images')
