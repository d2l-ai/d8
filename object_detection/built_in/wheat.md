# `wheat`

```{.python .input}
#@hide
# DO NOT EDIT THIS NOTEBOOK.

# This notebook is automatically generated from the `template` notebook in this
# folder by running `autodatasets gen_desc`
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

from autodatasets.object_detection import Dataset
```

Summary this dataset.

```{.python .input}
#@hide_code
name = "wheat"
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

The histogram of the center of bounding boxes. 

```{.python .input}
#@hide_code
pd.DataFrame({'x-center': (ds.df['xmin']+ds.df['xmax'])/2, 
              'y-center': (ds.df['ymin']+ds.df['ymax'])/2}).hist();
```
