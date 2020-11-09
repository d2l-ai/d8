# `horses-or-humans`

```{.python .input}
#@hide
# DO NOT EDIT THIS NOTEBOOK.

# This notebook is automatically generated from the `template` notebook in this
# folder by running `d8 gen_desc`
```


```{.python .input}
#@hide
%load_ext autoreload
%autoreload 2
%matplotlib inline

from IPython import display
import pandas as pd

display.set_matplotlib_formats('svg')
pd.set_option('precision', 2)

from d8.image_classification import Dataset
```

Summary about this dataset.

```{.python .input}
#@hide_code
name = "horses-or-humans"
ds = Dataset.get(name)
ds.summary()
```

Example images with their labels.

```{.python .input}
#@hide_code
ds.show()
```

The number of examples for each class.

```{.python .input}
#@hide_code
ds.df.groupby('classname')['classname'].count().sort_values().plot.barh(
    figsize=(6, 2.5*len(ds.classes)/10));
```
