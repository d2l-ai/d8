# This file is generated from image_classification/kaggle.md automatically through:
#    d2lbook build lib
# Don't edit it directly

#@save_cell
from autodatasets.image_classification import Dataset

#@save_cell
@Dataset.add
def boat():
    return Dataset.from_folders('kaggle:clorichel/boat-types-recognition', '.')

#@save_cell
@Dataset.add
def intel_image_classification():
    return Dataset.from_folders('kaggle:puneet6060/intel-image-classification',
                                     'seg_train/seg_train', 'seg_test/seg_test')

#@save_cell
@Dataset.add
def fruits_360():
    return Dataset.from_folders('kaggle:moltean/fruits',
                                     'fruits-360/Training', 'fruits-360/Test')

