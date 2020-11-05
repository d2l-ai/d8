# The `autodatasets` Package

`autodatasets` is a Python package to allow you to use your own datasets in various
machine learning frameworks by a few lines of Python codes. It also provides
hundreds of build-in datasets with a great of diversity to your machine learning algorithms.

## Installation

`autodatasets` is a light-weight python package. The easiest way to install it is through `pip`


```bash
pip install autodatasets
```


## Datasets

You can check our examples to construct datasets and build-in datasets by selecting
the problem type you are interested.

```eval_rst

.. container:: cards

    .. card::
        :title: Image Classification
        :link: image_classification/getting_started.html

        Datasets to recognize an object in an image.

    .. card::
        :title: Object Detection
        :link: object_detection/getting_started.html

        Datasets to detect multiple objects with their bounding boxes in an image.


```

## Table of Contents

```toc
:maxdepth: 2

image_classification/index
object_detection/index
api/index
```

## Old
This package provides a list of build-in datasets, and allow users to load a raw dataset with a few lines of python codes.


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


