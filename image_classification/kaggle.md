# Kaggle

```{.python .input  n=1}
#@hide
%matplotlib inline
%load_ext autoreload
%autoreload 2

from IPython import display
display.set_matplotlib_formats('svg')
```

```{.python .input  n=2}
#@save_cell
from autodatasets.image_classification import Dataset
```

```{.python .input  n=3}
#@save_cell
@Dataset.add
def boat():
    return Dataset.from_folders('kaggle:clorichel/boat-types-recognition', '.')
```

```{.python .input}
ds = Dataset.get('boat')
ds.summary()
```

```{.python .input}
ds.show()
```

```{.python .input  n=4}
#@save_cell
@Dataset.add
def intel_image_classification():
    return Dataset.from_folders('kaggle:puneet6060/intel-image-classification',
                                     'seg_train/seg_train', 'seg_test/seg_test')
```

```{.python .input  n=5}
ds = Dataset.get('intel_image_classification')
ds.summary()
```

```{.python .input  n=6}
ds.show()
```

```{.python .input  n=7}
#@save_cell
@Dataset.add
def fruits_360():
    return Dataset.from_folders('kaggle:moltean/fruits',
                                     'fruits-360/Training', 'fruits-360/Test')
```

```{.python .input  n=8}
ds = Dataset.get('fruits_360')
ds.summary()
```

```{.python .input}
ds.show((2,8))
```
