# This file is generated from image_classification/from_folders.md automatically through:
#    d2lbook build lib
# Don't edit it directly

from autodatasets.image_classification import Dataset

name = 'ibeans'
url = 'https://storage.googleapis.com/ibeans/'
urls = (url+'train.zip', url+'validation.zip', url+'test.zip')
Dataset.add(name, Dataset.from_folders, (urls, ('*/train', '*/validation', '*/test')))

name = 'boat'
Dataset.add(name, Dataset.from_folders, ('kaggle:clorichel/boat-types-recognition', '.'))

name = 'intel'
Dataset.add(name, Dataset.from_folders, (
    'kaggle:puneet6060/intel-image-classification',
    ('*/seg_train', '*/seg_test')))

name = 'fruits-360'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:moltean/fruits', ('*/Training', '*/Test')))

name = 'caltech-256'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:jessicali9530/caltech256', '256_ObjectCategories'))

name = 'cub-200'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:tarunkr/caltech-birds-2011-dataset', '*/images'))

name = 'cifar10'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:swaroopkml/cifar10-pngs-in-folders', ('*/train', '*/test')))

name = 'citrus-leaves'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:dtrilsbeek/citrus-leaves-prepared', ('*/train', '*/validation')))

name = 'cmaterdb'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:ipythonx/ekush-bangla-handwritten-data-numerals', '.'))

name = 'cassava'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:cassava-disease:train.zip', 'train'))

name = 'dtd'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:jmexpert/describable-textures-dataset-dtd', 'dtd/images'))

name = 'eurosat'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:apollo2506/eurosat-dataset', 'EuroSAT'))

name = 'food-101'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:kmader/food41', 'images'))

name = 'horses-or-humans'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:sanikamal/horses-or-humans-dataset', 
             ('*/train', '*/validation')))

name = 'malaria'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:iarunava/cell-images-for-detecting-malaria',
             'cell_images'))

name = 'flower-102'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:lenine/flower-102diffspecies-dataset',
             ('*/train', '*/valid')))

name = 'green-finder'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:tobiek/green-finder', '*'))

name = 'leaves'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:rohit9086/leaves', ('*/train', '*/test')))

name = 'plant-village'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:abdallahalidev/plantvillage-dataset', '*/segmented'))

name = 'rock-paper-scissors'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:drgfreeman/rockpaperscissors', '.'))

name = 'sun-397'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:lash45/sun397-50-50', ('*/train', '*/test')))

name = 'chessman'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:niteshfre/chessman-image-dataset', '*/Chess'))

name = 'casting-products'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:ravirajsinh45/real-life-industrial-dataset-of-casting-product', '*'))

name = 'monkey-10'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:slothkong/10-monkey-species', '*'))

#
# name = 'art-style-5'
# Dataset.add(name, Dataset.from_folders, 
#             ('kaggle:thedownhill/art-images-drawings-painting-sculpture-engraving', '*'))
#              'dataset/dataset_updated/training_set', 'dataset/dataset_updated/validation_set'))

name = 'dog-cat-panda'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:ashishsaxena2209/animal-image-datasetdog-cat-and-panda', 'animals'))

name = 'broad-leaved-dock'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:gavinarmstrong/open-sprayer-images', '*'))

name = 'food-or-not-food'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:trolukovich/food5k-image-dataset', '*'))

name = 'gemstones'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:lsind18/gemstones-images', '*'))

name = '12306'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:libowei/12306-captcha-image', '*'))

name = 'hurricane-damage'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:kmader/satellite-images-of-hurricane-damage', ('train_another', 'validation_another')))

name = 'animal-10'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:alessiocorrado99/animals10', 'raw-img'))

name = 'walk-or-run'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:huan9huan/walk-or-run', '*'))

name = 'gender'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:cashutosh/gender-classification-dataset', '*'))

#
# name = 'caricature'
# Dataset.add(name, Dataset.from_folders, 
#             ('kaggle:ranjeetapegu/caricature-image', '*'))

name = 'brain-tumor'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:simeondee/brain-tumor-images-dataset', '*'))

name = 'facial-expression'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:astraszab/facial-expression-dataset-image-folders-fer2013', '*'))

name = 'rice-diseases'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:minhhuy2810/rice-diseases-image-dataset', '*'))#'LabelledRice/Labelled'))

name = 'mushrooms'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:maysee/mushrooms-classification-common-genuss-images', '*'))

name = 'oregon-wildlife'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:virtualdvid/oregon-wildlife', '*'))

name = 'bird-225'
Dataset.add(name, Dataset.from_folders, 
            ('kaggle:gpiosenka/100-bird-species', '*'))

