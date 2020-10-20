# Kaggle

```{.python .input}
%load_ext autoreload
%autoreload 2

from IPython import display
display.set_matplotlib_formats('svg')
```

```{.python .input}
#@save_cell
from autodatasets import image_classification as task
```

```{.python .input}
#@save_cell
@task.add_dataset
def boat():
    return task.dataset_from_folders('kaggle:clorichel/boat-types-recognition', '.')
```

```{.python .input}
#@save_cell
@task.add_dataset
def intel_image_classification():
    return task.dataset_from_folders('kaggle:puneet6060/intel-image-classification',
                                     'seg_train/seg_train', 'seg_test/seg_test')
```

```{.python .input}
ds = task.get_dataset('intel_image_classification')
ds.summary()
```

```{.python .input}
ds.show()
```
