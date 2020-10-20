# This file is generated from image_classification/kaggle.md automatically through:
#    d2lbook build lib
# Don't edit it directly

#@save_cell
from autodatasets import image_classification as task

#@save_cell
@task.add_dataset
def boat():
    return task.dataset_from_folders('kaggle:clorichel/boat-types-recognition', '.')

#@save_cell
@task.add_dataset
def intel_image_classification():
    return task.dataset_from_folders('kaggle:puneet6060/intel-image-classification',
                                     'seg_train/seg_train', 'seg_test/seg_test')

