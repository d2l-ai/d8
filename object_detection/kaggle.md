# Detection Datasets from Kaggle

```{.python .input}
%matplotlib inline
from IPython import display
display.set_matplotlib_formats('svg')

```

```{.python .input}
#@save_cell
from autodatasets import data_downloader, data_reader, object_detection
import pandas as pd 

class KaggleWheat(object_detection.Dataset):
    def __init__(self):
        name = 'global-wheat-detection'
        reader = data_reader.create_reader(
            data_downloader.download_kaggle(name, object_detection.DATAROOT/name))
        
        df = pd.read_csv(reader.open('train.csv'))
        bbox = df.bbox.str.split(',', expand=True)
        xmin = bbox[0].str.strip('[ ').astype(float) / df.width
        ymin = bbox[1].str.strip(' ').astype(float) / df.height
        train_df = pd.DataFrame({
            'filepath':'train/'+df.image_id+'.jpg',
            'xmin':xmin,
            'ymin':ymin,
            'xmax':bbox[2].str.strip(' ').astype(float) / df.width + xmin,
            'ymax':bbox[3].str.strip(' ]').astype(float) / df.height + ymin,
            'classname':df.source})
        super().__init__(name, reader, train_df)

object_detection.add_dataset('global-wheat-detection', KaggleWheat)        
```

```{.python .input}
ds = KaggleWheat()
ds.show()
```
