# This file is generated from semantic_segmentation/dataset.md automatically through:
#    d2lbook build lib
# Don't edit it directly

from typing import Union, Sequence, Callable, Dict, Tuple, Optional
from d8 import core
import pandas as pd
import pathlib

class Dataset(core.BaseDataset):
    def __init__(self,
                 df: pd.DataFrame,
                 reader: core.Reader,
                 pixel_to_class,
                 name: str = '',):
        super().__init__(df, reader, name)
        self.pixel_to_class = pixel_to_class

    def show(self, layout=(2,3)) -> None:
        nrows, ncols = layout
        max_width = 400
        samples = self.df.sample(n=nrows*ncols, random_state=0)
        images = []
        for (_, sample) in samples.iterrows():
            images.append(self.reader.read_image(sample['file_path'], max_width=max_width))
            images.append(self.reader.read_image(sample['label_file_path'], max_width=max_width))
        core.show_images(images, (nrows, ncols*2), 7.0 / ncols)

    @classmethod
    def from_label_func(cls, data_path: Union[str, Sequence[str]],
                        label_func: Callable[[pathlib.Path], Optional[pathlib.Path]],
                        pixel_to_class_func: Callable[[core.Reader], Dict[Sequence[int], str]]):
        reader = cls.create_reader(data_path)
        all_image_paths = reader.list_images()
        pairs = []
        for p in all_image_paths:
            if label_func(p):
                pairs.append({'file_path':p, 'label_file_path':label_func(p)})
        pixel_to_class = pixel_to_class_func(reader)
        return Dataset(pd.DataFrame(pairs), reader, pixel_to_class)

    def __getitem__(self, idx):
        if idx < 0 or idx > self.__len__():
            raise IndexError(f'index {idx} out of range [0, {self.__len__()})')
        img_path = self.df['file_path'][idx]
        lbl_path = self.df['label_file_path'][idx]
        img = self.reader.read_image(img_path)
        lbl = self.reader.read_image(lbl_path)
        return np.array(img), np.array(lbl)

