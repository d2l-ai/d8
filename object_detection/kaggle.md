# Detection Datasets from Kaggle

```{.python .input}
#@hide
%matplotlib inline
%load_ext autoreload
%autoreload 2

from IPython import display
display.set_matplotlib_formats('svg')
```

```{.python .input}
#@save_cell
import pandas as pd 
from autodatasets.object_detection import Dataset

@Dataset.add
def wheat():
    def train_df_fn(reader):
        df = pd.read_csv(reader.open('train.csv'))
        bbox = df.bbox.str.split(',', expand=True)
        xmin = bbox[0].str.strip('[ ').astype(float) / df.width
        ymin = bbox[1].str.strip(' ').astype(float) / df.height
        return pd.DataFrame({
            'filepath':'train/'+df.image_id+'.jpg',
            'xmin':xmin,
            'ymin':ymin,
            'xmax':bbox[2].str.strip(' ').astype(float) / df.width + xmin,
            'ymax':bbox[3].str.strip(' ]').astype(float) / df.height + ymin,
            'classname':df.source})
    return Dataset.from_df_func('kaggle:global-wheat-detection', train_df_fn)
```

```{.python .input}
ds = Dataset.get('wheat')
ds.show()
```
