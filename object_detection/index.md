# Object Detection Datasets

Summary of all provided datasets

```{.python .input}
from autodatasets import object_detection
import pandas as pd 

names = object_detection.list_datasets()

summaries = [object_detection.get_dataset(name).summary().loc['train'].to_dict() for name in names]
pd.DataFrame(summaries, index=names)
```

```toc
kaggle
makeml
```

