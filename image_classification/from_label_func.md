# from label func

```{.python .input  n=1}
#@hide
%matplotlib inline
%load_ext autoreload
%autoreload 2

from IPython import display
display.set_matplotlib_formats('svg')
```

```{.python .input  n=2}
#@save
from autodatasets.image_classification import Dataset

def show(name, layout=(2,8)):
    ds = Dataset.get(name)
    ds.show(layout)
    return ds.summary()
```

## Stanford dogs

```{.python .input  n=3}
#@save
name = 'stanford-dogs'
Dataset.add(name, Dataset.from_label_func, 
            ('kaggle:jessicali9530/stanford-dogs-dataset', 
             lambda path: path.parent.name.split('-')[1].lower()))

show(name)
```

## Butterfly

```{.python .input  n=4}
#@save
name = 'butterfly'
Dataset.add(name, Dataset.from_label_func, 
            ('kaggle:veeralakrishna/butterfly-dataset', 
             lambda path: path.stem[:3] if 'images' in str(path) else None))

show(name)
```

## CUB-200

```{.python .input  n=5}
#@save
name = 'cub-200'
Dataset.add(name, Dataset.from_label_func, 
            ('kaggle:tarunkr/caltech-birds-2011-dataset', 
             lambda path: path.parent.name.split('.')[1].lower() if 'images' in str(path) else None))

show(name)
```

## cat vs dog

```{.python .input}
#@save
name = 'dogs-vs-cats'
Dataset.add(name, Dataset.from_label_func, 
            ('kaggle:dogs-vs-cats:train.zip', lambda path: path.name.split('.')[0]))

show(name)
```

## deep weeds

```{.python .input}
#@save
name = 'deep-weeds'
Dataset.add(name, Dataset.from_label_func, 
            ('kaggle:coreylammie/deepweedsx', 
             lambda path: path.with_suffix('').name.split('-')[-1]))

show(name)

```

```{.python .input}
#@save
name = 'oxford-pets'
Dataset.add(name, Dataset.from_label_func, 
            ('kaggle:alexisbcook/oxford-pets', 
             lambda path: path.name.split('_')[0].lower() if 'images' in str(path) else None))

show(name)
```

```{.python .input}
#@save
name = 'lego-brick'
Dataset.add(name, Dataset.from_label_func, 
            ('kaggle:joosthazelzet/lego-brick-images', 
             lambda path: path.name.split(' ')[0].lower() if str(path).startswith('dataset') else None))

show(name)

```

```{.python .input}
#@save
name = 'satelite-plane'
Dataset.add(name, Dataset.from_label_func, 
            ('kaggle:rhammell/planesnet', 
             lambda path: path.name.split('__')[0]))

show(name)

```

```{.python .input}
#@save

name = 'honey-bee'
Dataset.add(name, Dataset.from_label_func, 
            ('kaggle:jenny18/honey-bee-annotated-images', 
             lambda path: path.name.split('_')[0]))

show(name)


```

```{.python .input}
#@save
name = 'coil-100'
Dataset.add(name, Dataset.from_label_func, 
            ('kaggle:jessicali9530/coil100',
             lambda path: path.name.split('__')[0]))

show(name)
```

```{.python .input}
#@save
name = 'flower-10'
Dataset.add(name, Dataset.from_label_func, 
            ('kaggle:aksha05/flower-image-dataset',
             lambda path: path.name.split('_')[0].lower()))

show(name)
```
