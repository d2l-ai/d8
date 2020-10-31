# from df func

```{.python .input  n=1}
#@hide
%matplotlib inline
%load_ext autoreload
%autoreload 2

from IPython import display
display.set_matplotlib_formats('svg')
```

```{.python .input  n=2}
#@save
from autodatasets.image_classification import Dataset

def show(name, layout=(2,8)):
    ds = Dataset.get(name)
    ds.show(layout)
    return ds.summary()
```
