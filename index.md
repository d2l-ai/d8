# The `autodatasets` Package

```eval_rst
.. py:currentmodule:: autodatasets

```


This package provides a list of build-in datasets, and allow users to load a raw dataset with a few lines of python codes.

```{.python .input}
#@hide
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
from autodatasets.image_classification import Dataset

Dataset.add('boat', Dataset.from_folders, ['kaggle:clorichel/boat-types-recognition', '.'])
```

The above code block calls :func:`image_classification.Dataset.from_folders` to pass the Kaggle dataset name, it starts with the prefix `kaggle:`, and then followed by the user name and the dataset name. You can also provide a URL to a zip/tar file such as `http://example.com/data.zip`, or a github repo such as `github:autodatasets/data`. The second argument specifies the root directory of the training set, which is the root `.` for this dataset. As this dataset didn't provide a split of training, validation and test, so we only specify the training set. This function assumes all images belong to a class is stored in the directory with the class as the name. You can check other functions in the :class:`autodatasets.image_classification.Dataset` class for other formats.

The decorator :func:`image_classification.Dataset.add` will register the dataset into the package, which can be retrieved by using the function name. For example, we obtain the dataset and print the summary.

```{.python .input}
ds = Dataset.get('boat')
ds.summary()
```

We can also show several examples of this dataset.

```{.python .input}
ds.show()
```

You can check more examples to construct image classification datasets at :ref:`sec_image_classification`.

A set of build-in datasets.

```{.python .input}
Dataset.summary_all(quick=True)
```

## Object Detection

```{.python .input}
from autodatasets.object_detection import Dataset

Dataset.add('paper-prototype', Dataset.from_voc,
            ['https://arcraftimages.s3-accelerate.amazonaws.com/Datasets/PaperPrototype/PaperPrototypePascalVOC.zip',
             'images', 'annotations'])
Dataset.get('paper-prototype').show()
```

```{.python .input}
Dataset.summary_all(quick=True)
```

You can check :ref:`sec_object_detection` for detailed information and building your own datasets.

```toc
:maxdepth: 2

image_classification/index
object_detection/index
benchmark/index
```


```toc
:maxdepth: 2

api
```

