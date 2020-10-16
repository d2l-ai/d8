# Detection datasets from makeml.app

```{.python .input}
%matplotlib inline
from IPython import display
display.set_matplotlib_formats('svg')
```

```{.python .input}
#@save_cell
from autodatasets import data_downloader, data_reader, object_detection

class MakeMLDataset(object_detection.Dataset):
    def __init__(self, name:str, description:str, zip_path:str=None):
        self._download_root = 'https://arcraftimages.s3-accelerate.amazonaws.com/Datasets/'
        self.__doc__ = description
        if zip_path is None:
            camel_case = name.replace('-',' ').title().replace(' ', '')
            zip_path = f'{camel_case}/{camel_case}PascalVOC.zip'
        filepaths = data_downloader.download(self._download_root+zip_path, object_detection.DATAROOT/name)
        reader = data_reader.create_reader(filepaths)
        train_df = object_detection.parse_voc(reader, 'images', 'annotations')
        super().__init__(name, reader, train_df)
```

```{.python .input}
ds = MakeMLDataset('paperprototype', 
                   'Detect elements in handraw papers')
ds.show()
```

```{.python .input}
#@save_cell
datasets = (('sheep', 'Detect sheeps'),
            ('paperprototype', 'Detect elements in handraw papers'),
           )

for name, desc in datasets:
    object_detection.add_dataset(name, MakeMLDataset, (name, desc))
```
