# Getting Started

```{.python .input  n=1}
#@hide
%load_ext autoreload
%autoreload 2
%matplotlib inline

from IPython import display
import pandas as pd

display.set_matplotlib_formats('svg')
pd.set_option('precision', 2)
```

This tutorial explains how to get a built-in image classification datasets and how to construct a customized one.

## Getting a Built-in Dataset

Let's first import the `Dataset` class from the `object_detection`. Calling its class method `list` will return the list of build-in dataset names.



A object detection dataset should contain at least two components, the raw images stored on a folder or a zip/tar file, a DataFrame containing the labels. Let's print an example DataFrame:

You can see that it must contain 6 columns ...

Next, here is a list of examples how to prepare various detection datasets,mm

```{.python .input  n=2}
from d8.object_detection import Dataset

names = Dataset.list()
len(names), names[:5]
```

```{.python .input  n=4}
ds = Dataset.get('sheep')
ds.summary()
```

```{.python .input}
ds.show()
```

```{.python .input}
Dataset.summary_all(quick=True)
```
