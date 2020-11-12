# Constructing Datasets
:label:`image_classification_constructing`

```eval_rst

.. currentmodule:: d8.image_classification

```


The :class:`Dataset` class provides multiple class methods to construct an image classification dataset.

```eval_rst

.. autosummary::

   Dataset.from_folders
   Dataset.from_label_func
   Dataset.from_df_func

```


In this tutorial, we show how all built-in datasets are constructed. You can follow these examples to construct your own datasets. If you would like to add your datasets into the `d8` built-in dataset collection, it'll be great! You can add your code in this file and send us a pull request.

```{.python .input  n=2}
#@hide
%matplotlib inline
%load_ext autoreload
%autoreload 2

from IPython import display
display.set_matplotlib_formats('svg')
```

Let's import `d8` and the `Dataset` class.

```{.python .input  n=3}
#@save
from typing import Sequence, Dict, Union, Callable, Any, Optional
import d8 as ad
from d8.image_classification import Dataset
```

If you don't know which functions to use. The easy way is to load the data to print its file structure. TODO(mli) add an example.

## `from_folders`

To use the `from_folders` method, we need to specify the name, URLs and roots for a dataset. For each dataset, we save the meta information in the following code block, which is a dictionary with:
- `name`: the string name
- `url`: a kaggle dataset (`kaggle:user_name/dataset_name`), a competition (`kaggle:competition_name`), a URL, or a list or URLs.
- `root`: a wildcard path or a list of paths.

```{.python .input}
#@save_cell
from_folders_meta: Sequence[Dict[str, Union[Sequence[str], str]]] = [
    {'name' : 'ibeans',
     'url'  : [f'https://storage.googleapis.com/ibeans/{part}.zip' for part in ('train', 'validation', 'test')],
     'root' : '*'},
    {'name' : 'boat',
     'url'  : 'kaggle:clorichel/boat-types-recognition',
     'root' : '.'},
    {'name' : 'intel',
     'url'  : 'kaggle:puneet6060/intel-image-classification',
     'root' : ('*/seg_train', '*/seg_test')},
    {'name' : 'fruits-360',
     'url'  : 'kaggle:moltean/fruits',
     'root' : ('*/Training', '*/Test')},
    {'name' : 'caltech-256',
     'url'  : 'kaggle:jessicali9530/caltech256',
     'root' : '256_ObjectCategories'},
    {'name' : 'cub-200',
     'url'  : 'kaggle:tarunkr/caltech-birds-2011-dataset',
     'root' : '*/images'},
    {'name' : 'cifar10',
     'url'  : 'kaggle:swaroopkml/cifar10-pngs-in-folders',
     'root' : ('*/train', '*/test')},
    {'name' : 'citrus-leaves',
     'url'  : 'kaggle:dtrilsbeek/citrus-leaves-prepared',
     'root' : ('*/train', '*/validation')},
    {'name' : 'cmaterdb',
     'url'  : 'kaggle:ipythonx/ekush-bangla-handwritten-data-numerals',
     'root' : '.'},
    {'name' : 'cassava',
     'url'  : 'kaggle:cassava-disease:train.zip',
     'root' : 'train'},
    {'name' : 'dtd',
     'url'  : 'kaggle:jmexpert/describable-textures-dataset-dtd',
     'root' : 'dtd/images'},
    {'name' : 'eurosat',
     'url'  : 'kaggle:apollo2506/eurosat-dataset',
     'root' : 'EuroSAT'},
    {'name' : 'food-101',
     'url'  : 'kaggle:kmader/food41',
     'root' : 'images'},
    {'name' : 'horses-or-humans',
     'url'  : 'kaggle:sanikamal/horses-or-humans-dataset',
     'root' : ('*/train', '*/validation')},
    {'name' : 'malaria',
     'url'  : 'kaggle:iarunava/cell-images-for-detecting-malaria',
     'root' : 'cell_images'},
    {'name' : 'flower-102',
     'url'  : 'kaggle:lenine/flower-102diffspecies-dataset',
     'root' : ('*/train', '*/valid')},
    {'name' : 'green-finder',
     'url'  : 'kaggle:tobiek/green-finder',
     'root' : '*'},
    {'name' : 'leaves',
     'url'  : 'kaggle:rohit9086/leaves',
     'root' : ('*/train', '*/test')},
    {'name' : 'plant-village',
     'url'  : 'kaggle:abdallahalidev/plantvillage-dataset',
     'root' : '*/segmented'},
    {'name' : 'rock-paper-scissors',
     'url'  : 'kaggle:drgfreeman/rockpaperscissors',
     'root' : '.'},
    {'name' : 'sun-397',
     'url'  : 'kaggle:lash45/sun397-50-50',
     'root' : ('*/train', '*/test')},
    {'name' : 'chessman',
     'url'  : 'kaggle:niteshfre/chessman-image-dataset',
     'root' : '*/Chess'},
    {'name' : 'casting-products',
     'url'  : 'kaggle:ravirajsinh45/real-life-industrial-dataset-of-casting-product',
     'root' : '*'},
    {'name' : 'monkey-10',
     'url'  : 'kaggle:slothkong/10-monkey-species',
     'root' : '*'},
    {'name' : 'dog-cat-panda',
     'url'  : 'kaggle:ashishsaxena2209/animal-image-datasetdog-cat-and-panda',
     'root' : 'animals'},
    {'name' : 'broad-leaved-dock',
     'url'  : 'kaggle:gavinarmstrong/open-sprayer-images',
     'root' : '*'},
    {'name' : 'food-or-not-food',
     'url'  : 'kaggle:trolukovich/food5k-image-dataset',
     'root' : '*'},
    {'name' : 'gemstones',
     'url'  : 'kaggle:lsind18/gemstones-images',
     'root' : '*'},
    {'name' : 'hurricane-damage',
     'url'  : 'kaggle:kmader/satellite-images-of-hurricane-damage',
     'root' : ('train_another', 'validation_another')},
    {'name' : 'animal-10',
     'url'  : 'kaggle:alessiocorrado99/animals10',
     'root' : 'raw-img'},
    {'name' : 'walk-or-run',
     'url'  : 'kaggle:huan9huan/walk-or-run',
     'root' : '*'},
    {'name' : 'gender',
     'url'  : 'kaggle:cashutosh/gender-classification-dataset',
     'root' : '*'},
    {'name' : 'brain-tumor',
     'url'  : 'kaggle:simeondee/brain-tumor-images-dataset',
     'root' : '*'},
    {'name' : 'facial-expression',
     'url'  : 'kaggle:astraszab/facial-expression-dataset-image-folders-fer2013',
     'root' : '*'},
    {'name' : 'rice-diseases',
     'url'  : 'kaggle:minhhuy2810/rice-diseases-image-dataset',
     'root' : '*'},
    {'name' : 'mushrooms',
     'url'  : 'kaggle:maysee/mushrooms-classification-common-genuss-images',
     'root' : '*'},
    {'name' : 'oregon-wildlife',
     'url'  : 'kaggle:virtualdvid/oregon-wildlife',
     'root' : '*'},
    {'name' : 'bird-225',
     'url'  : 'kaggle:gpiosenka/100-bird-species',
     'root' : '*'},
]
```

Then we add each dataset one-by-one.

```{.python .input}
#@save
for x in from_folders_meta:
    Dataset.add(x['name'], Dataset.from_folders, (x['url'], x['root']))
```

Check a random dataset if it looks right.

```{.python .input}
def show(name, layout=(2,8)):
    ds = Dataset.get(name)
    ds.show(layout)
    return ds.summary()

show('dtd')
```

## `from_label_func`

The `from_label_func` method provides a more flexible way to construct a dataset. Similar to above we need to specify the name and URLs, but instead of root folders, now we pass a function that maps a file_path, a :class:`pathlib.Path` object, to a string class name.

Even sometimes the data format fits into `from_folders`, we may still use this method as it allows us to change the class names.

```{.python .input}
#@save_cell
from_label_func_meta: Sequence[Dict[str, Union[Callable[[Any], Optional[str]], Sequence[str], str]]] = [
    {'name' : 'stanford-dogs',
     'url'  : 'kaggle:jessicali9530/stanford-dogs-dataset',
     'func' : lambda path: path.parent.name.split('-')[1].lower()},
    {'name' : 'butterfly',
     'url'  : 'kaggle:veeralakrishna/butterfly-dataset',
     'func' : lambda path: path.stem[:3] if 'images' in str(path) else None},
    {'name' : 'cub-200',
     'url'  : 'kaggle:tarunkr/caltech-birds-2011-dataset',
     'func' : lambda path: path.parent.name.split('.')[1].lower() if 'images' in str(path) else None},
    {'name' : 'dogs-vs-cats',
     'url'  : 'kaggle:dogs-vs-cats:train.zip',
     'func' : lambda path: path.name.split('.')[0]},
    {'name' : 'deep-weeds',
     'url'  : 'kaggle:coreylammie/deepweedsx',
     'func' : lambda path: path.with_suffix('').name.split('-')[-1]},
    {'name' : 'oxford-pets',
     'url'  : 'kaggle:alexisbcook/oxford-pets',
     'func' : lambda path: path.name.split('_')[0].lower() if 'images' in str(path) else None},
    {'name' : 'lego-brick',
     'url'  : 'kaggle:joosthazelzet/lego-brick-images',
     'func' : lambda path: path.name.split(' ')[0].lower() if str(path).startswith('dataset') else None},
    {'name' : 'satelite-plane',
     'url'  : 'kaggle:rhammell/planesnet',
     'func' : lambda path: path.name.split('__')[0]},
    {'name' : 'honey-bee',
     'url'  : 'kaggle:jenny18/honey-bee-annotated-images',
     'func' : lambda path: path.name.split('_')[0]},
    {'name' : 'coil-100',
     'url'  : 'kaggle:jessicali9530/coil100',
     'func' : lambda path: path.name.split('__')[0]},
    {'name' : 'flower-10',
     'url'  : 'kaggle:aksha05/flower-image-dataset',
     'func' : lambda path: path.name.split('_')[0].lower()}
]
```

Again, register all datasets and check one random dataset.

```{.python .input}
#@save
for y in from_label_func_meta:
    Dataset.add(y['name'], Dataset.from_label_func, (y['url'], y['func']))

show('flower-10')
```

## `from_df_func`

A `df_func` should return a Pandas DataFrame that contains at least two columns. One column is called `file_path`, containing the relative image file paths related to the root, the other column is called `class_name`, containing the image labels. Let's look at example.

```{.python .input  n=5}
ds = Dataset.get('boat')
ds.df.head(n=3)
```

TODO(mli) add an example here.

Last, you can check the detailed information of all added datasets in :ref:`image_classification_built_in`.
