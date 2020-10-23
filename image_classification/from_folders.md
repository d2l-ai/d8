# From folders

```{.python .input  n=73}
#@hide
%matplotlib inline
%load_ext autoreload
%autoreload 2

from IPython import display
display.set_matplotlib_formats('svg')
```

## ibeans

```{.python .input  n=74}
#@save
from autodatasets.image_classification import Dataset
```

```{.python .input}
#@save
url = 'https://storage.googleapis.com/ibeans/'
urls = [url+'train.zip', url+'validation.zip', url+'test.zip']
Dataset.add('ibeans', Dataset.from_folders, [urls, 'train/train', 'validation/validation', 'test/test'])

ds = Dataset.get('ibeans')
ds.show((2,8))
ds.summary()
```

## boat

```{.python .input}
#@save
Dataset.add('boat', Dataset.from_folders, ('kaggle:clorichel/boat-types-recognition', '.'))

ds = Dataset.get('boat')
ds.show()
ds.summary()
```

## intel

```{.python .input}
#@save
Dataset.add('intel', Dataset.from_folders, (
    'kaggle:puneet6060/intel-image-classification',
    'seg_train/seg_train', 'seg_test/seg_test'))

ds = Dataset.get('intel')
ds.show()
ds.summary()
```

## Fruits 360

```{.python .input}
#@save
Dataset.add('fruits-360', Dataset.from_folders, 
            ('kaggle:moltean/fruits', 'fruits-360/Training', 'fruits-360/Test'))

ds = Dataset.get('fruits-360')
ds.show((2,8))
ds.summary()
```

## Caltech 256

```{.python .input}
#@save
Dataset.add('caltech-256', Dataset.from_folders, 
            ('kaggle:jessicali9530/caltech256', '256_ObjectCategories'))

ds = Dataset.get('caltech-256')
ds.show()
ds.summary()
```

## CUB-200

```{.python .input  n=75}
#@save
Dataset.add('cub-200', Dataset.from_folders, 
            ('kaggle:tarunkr/caltech-birds-2011-dataset', 'CUB_200_2011/images'))

ds = Dataset.get('cub-200')
ds.show()
ds.summary()
```

```{.python .input  n=78}
ds.reader.list_files()
```

```{.python .input  n=77}
#@save
Dataset.add('cassava', Dataset.from_folders, 
            ('kaggle:cassava-disease', 'train'))


ds = Dataset.get('cassava')
ds.show()
ds.summary()
```

# Stanford cars
