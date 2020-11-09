# Constructing Datasets
:label:`object_detection_constructing`

```eval_rst

.. currentmodule:: d8.object_detection

```


The :class:`Dataset` class provides multiple class methods to construct an image classification dataset.

```eval_rst

.. autosummary::

   Dataset.from_voc
   Dataset.from_df_func

```


```{.python .input  n=4}
#@hide
%matplotlib inline
%load_ext autoreload
%autoreload 2

from IPython import display
display.set_matplotlib_formats('svg')
```

```{.python .input  n=5}
#@save
from d8.object_detection import Dataset
```

## `from_voc`

```{.python .input  n=6}
#@save_cell
def make_ml(name):
    camel_case = name.replace('-',' ').title().replace(' ', '')
    url = f'https://arcraftimages.s3-accelerate.amazonaws.com/Datasets/{camel_case}/{camel_case}PascalVOC.zip'
    return Dataset.from_voc(url, 'images', 'annotations')

names = ['sheep', 'paper-prototype', 'raccoon', 'boggle-boards', 'plant-doc',
         'hard-hat-workers', 'pistol', 'cars-and-traffic-signs', 'tomato',
         'dice', 'potholes', 'ships', 'mask', 'chess', 'mobile-phones',
         'glasses', 'road-signs', 'fruits', 'bikes', 'headphones', 'fish',
         'drone', 'car-license-plates', 'pets', 'faces', 'helmets', 'clothing',
         'hands', 'soccer-ball'
        ]

for name in names:
    Dataset.add(name, make_ml, [name])
```

```{.python .input  n=7}
def show(name, layout=(2,4)):
    ds = Dataset.get(name)
    ds.show(layout)
    return ds.summary()

show('sheep')
```

```{.python .input  n=10}
#@save_cell
import d8 as ad
from d8.object_detection import dataset
import pandas as pd

@Dataset.add
def stanford_dogs():
    reader = ad.create_reader(ad.download('kaggle:jessicali9530/stanford-dogs-dataset'))
    images = reader.list_images()
    entries = []
    for img in images:
        xml_fp = 'annotations/Annotation/'+img.parent.name+'/'+img.stem
        for label in dataset.parse_voc_annotation(reader.open(xml_fp)):
            label.filepath = str(img)
            entries.append(label)
    return Dataset(pd.DataFrame(entries), reader)

```

```{.python .input  n=11}
show('stanford-dogs')
```

## `from_df_func`

```{.python .input  n=12}
#@save_cell
@Dataset.add
def wheat():
    def train_df_fn(reader):
        df = pd.read_csv(reader.open('train.csv'))
        bbox = df.bbox.str.split(',', expand=True)
        xmin = bbox[0].str.strip('[ ').astype(float) / df.width
        ymin = bbox[1].str.strip(' ').astype(float) / df.height
        return pd.DataFrame({
            'filepath':'train/'+df.image_id+'.jpg',
            'xmin':xmin,
            'ymin':ymin,
            'xmax':bbox[2].str.strip(' ').astype(float) / df.width + xmin,
            'ymax':bbox[3].str.strip(' ]').astype(float) / df.height + ymin,
            'classname':df.source})
    return Dataset.from_df_func('kaggle:global-wheat-detection', train_df_fn)
```

```{.python .input  n=13}
show('wheat')
```
