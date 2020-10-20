# The `autodatasets` Package

```eval_rst
.. py:currentmodule:: autodatasets

```


This package provides a list of build-in datasets, and allow users to load a raw dataset with a few lines of python codes.

```{.python .input}
# Hide code
%load_ext autoreload
%autoreload 2
%matplotlib inline

from IPython import display
import pandas as pd

display.set_matplotlib_formats('svg')
pd.set_option('precision', 2)
```

## Image Classification

Here is an example to build a dataset based on a [Kaggle dataset](https://www.kaggle.com/clorichel/boat-types-recognition).

```{.python .input}
from autodatasets import image_classification as task

@task.add_dataset
def boat():
    return task.dataset_from_folders('kaggle:clorichel/boat-types-recognition', '.')
```

The above code block calls the :func:`image_classification.dataset_from_folders` function by passing the Kaggle dataset name, it starts with the prefix `kaggle:`, and then followed by the user name and the dataset name. You can also provide a URL to a zip/tar file such as `http://example.com/data.zip`, or a github repo such as `github:autodatasets/data`. The second argument specifies the root directory of the training set, which is the root `.` for this dataset. As this dataset didn't provide a split of training, validation and test, so we only specify the training set. This function assumes all images belong to a class is stored in the directory with the class as the name. You can check other functions in the :mod:`autodatasets.image_classification` module for other formats.

The decorator :func:`image_classification.add_dataset` will register the dataset into the package, which can be retrieved by using the function name. For example, we obtain the dataset and print the summary.

```{.python .input}
ds = task.get_dataset('boat')
ds.summary()
```

We can also show several examples of this dataset.

```{.python .input}
ds.show()
```

You can check more examples to construct image classification datasets at :ref:`sec_image_classification`.

A set of build-in datasets.

```{.python .input}
task.summary()
```

## Object Detection

```{.python .input}
from autodatasets import object_detection

object_detection.summary()
```

You can check :ref:`sec_object_detection` for detailed information and building your own datasets.

```toc
image_classification/index
object_detection/index
```


```toc
api
```

