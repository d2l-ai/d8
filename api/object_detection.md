# Object Detection
:label:`object_detection_api`

The dataset class for object detection is provided in the `autodatasets.object_detection` module.

```eval_rst

.. currentmodule:: autodatasets.object_detection

.. autoclass:: Dataset

```

## Adding and Getting Datasets

`autodatasets` provides a large set of build-in dataset, the following functions list, add and get these datasts.

```eval_rst

.. autosummary::

   Dataset.list
   Dataset.get
   Dataset.add

```

## Constructing a Dataset

We often construct a new dataset using one of the following class methods.
You could find examples from :ref:`sec_object_detection`.

```eval_rst

.. autosummary::

   Dataset.from_voc
   Dataset.from_df_func

```


## Manipulating a Dataset

These functions manipulate a dataset.

```eval_rst

.. autosummary::

   Dataset.split
   Dataset.merge

```


## Visualizing a Dataset

These functions let you have a peak about a dataset.

```eval_rst

.. autosummary::

   Dataset.show
   Dataset.summary
   Dataset.summary_all

```

## `Dataset`

```eval_rst

.. autoclass:: autodatasets.object_detection.Dataset
   :members:
   :show-inheritance:
   :inherited-members:

```
