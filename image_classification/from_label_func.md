# from label func

```{.python .input  n=2}
#@hide
%matplotlib inline
%load_ext autoreload
%autoreload 2

from IPython import display
display.set_matplotlib_formats('svg')
```

```{.python .input  n=3}
#@save
from autodatasets.image_classification import Dataset
```

# Stanford dogs

```{.python .input}
#@save
Dataset.add('stanford-dogs', Dataset.from_label_func, 
            ('kaggle:jessicali9530/stanford-dogs-dataset', 
             lambda path: path.parent.name.split('-')[1].lower()))

ds = Dataset.get('stanford-dogs')
ds.show()
ds.summary()
```

# Butterfly

```{.python .input  n=7}
#@save
Dataset.add('butterfly', Dataset.from_label_func, 
            ('kaggle:veeralakrishna/butterfly-dataset', 
             lambda path: path.stem[:3] if 'images' in str(path) else None))

ds = Dataset.get('butterfly')
ds.show()
ds.summary()


```

## CUB-200

```{.python .input  n=9}
#@save
Dataset.add('cub-200', Dataset.from_label_func, 
            ('kaggle:tarunkr/caltech-birds-2011-dataset', 
             lambda path: path.parent.name.split('.')[1].lower() if 'images' in str(path) else None))

ds = Dataset.get('cub-200')
ds.show()
ds.summary()
```
