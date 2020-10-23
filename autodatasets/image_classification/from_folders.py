# This file is generated from image_classification/from_folders.md automatically through:
#    d2lbook build lib
# Don't edit it directly

from autodatasets.image_classification import Dataset

url = 'https://storage.googleapis.com/ibeans/'
urls = [url+'train.zip', url+'validation.zip', url+'test.zip']
Dataset.add('ibeans', Dataset.from_folders, [urls, 'train/train', 'validation/validation', 'test/test'])

Dataset.add('boat', Dataset.from_folders, ('kaggle:clorichel/boat-types-recognition', '.'))

Dataset.add('intel', Dataset.from_folders, (
    'kaggle:puneet6060/intel-image-classification',
    'seg_train/seg_train', 'seg_test/seg_test'))

Dataset.add('fruits-360', Dataset.from_folders, 
            ('kaggle:moltean/fruits', 'fruits-360/Training', 'fruits-360/Test'))

