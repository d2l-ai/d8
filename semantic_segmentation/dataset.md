# The `Dataset` API


```{.python .input}
%load_ext autoreload
%autoreload 2
```

```{.python .input}
#@save
from typing import Union, Sequence, Callable, Dict, Tuple, Optional
from d8 import base_dataset, data_reader
import pandas as pd
import pathlib
```

```{.python .input}
#@save
class Dataset(base_dataset.BaseDataset):
    def __init__(self,
                 df: pd.DataFrame,
                 reader: data_reader.Reader,
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
        base_dataset.show_images(images, (nrows, ncols*2), 7.0 / ncols)

    @classmethod
    def from_label_func(cls, data_path: Union[str, Sequence[str]],
                        label_func: Callable[[pathlib.Path], Optional[pathlib.Path]],
                        pixel_to_class_func: Callable[[data_reader.Reader], Dict[Sequence[int], str]]):
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
```

```{.python .input}
def camseq01_class_mapping(reader):
    maps = []
    with reader.open('label_colors.txt') as f:
        lines = f.readlines()
    for line in lines:
        line = line.decode().strip().replace('\t\t', '\t')
        if not line: continue
        pixels, label = line.split('\t')
        pixels = tuple([int(p) for p in pixels.split(' ')])
        maps.append((pixels, label))
    return dict(maps)
```

```{.python .input}
#'http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamSeq01/CamSeq01.zip',
Dataset.add('camseq01', Dataset.from_label_func,
            [['kaggle:carlolepelaars/camseq-semantic-segmentation',
              'http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamSeq01/label_colors.txt'],
             lambda p: None if '_L' in str(p) else pathlib.Path(p.stem+'_L'+p.suffix),
             camseq01_class_mapping])

ds = Dataset.get('camseq01')
ds.show()
```
