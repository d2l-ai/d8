# Image Classification
:label:`image_classification_api`

The dataset class for image classification is provided in the `autodatasets.image_classification` module.

```eval_rst

.. currentmodule:: autodatasets.image_classification

.. autoclass:: Dataset

```

## Dataset Construction

We often construct a new dataset using one of the following class methods.
You could find examples from :ref:`sec_image_classification`.

```eval_rst

.. autosummary::

   Dataset.from_folders
   Dataset.from_label_func
   Dataset.from_df_func

```


## Dataset Manipulation

These functions manipulate a dataset.

```eval_rst

.. autosummary::

   Dataset.split

```

## Adding and Getting Datasets

`autodatasets` provides a large set of build-in dataset, the following functions list, add and get these datasts.

```eval_rst

.. autosummary::

   Dataset.list
   Dataset.get
   Dataset.add

```

## Visualization

These functions let you have a peak about a dataset.

```eval_rst

.. autosummary::

   Dataset.show
   Dataset.summary
   Dataset.summary_all

```

## Converting Formats

These functions export a `autodatasets` dataset into dataset formats for various libraries.

```eval_rst

.. autosummary::

   Dataset.to_mxnet

```


## xx

```eval_rst

.. autoclass:: autodatasets.image_classification.Dataset
   :members:
   :show-inheritance:
   :inherited-members:

```