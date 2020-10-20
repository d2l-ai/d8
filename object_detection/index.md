# Object Detection Datasets
:label:`sec_object_detection`

A object detection dataset should contain at least two components, the raw images stored on a folder or a zip/tar file, a DataFrame containing the labels. Let's print an example DataFrame:

```{.python .input}
from autodatasets import object_detection

ds = object_detection.get_dataset(object_detection.list_datasets()[0])
ds.train_df.head()
```

You can see that it must contain 6 columns ...

Next, here is a list of examples how to prepare various detection datasets

```toc
kaggle
makeml
```

