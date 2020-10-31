# This file is generated from image_classification/from_label_func.md automatically through:
#    d2lbook build lib
# Don't edit it directly

from autodatasets.image_classification import Dataset

name = 'stanford-dogs'
Dataset.add(name, Dataset.from_label_func, 
            ('kaggle:jessicali9530/stanford-dogs-dataset', 
             lambda path: path.parent.name.split('-')[1].lower()))

name = 'butterfly'
Dataset.add(name, Dataset.from_label_func, 
            ('kaggle:veeralakrishna/butterfly-dataset', 
             lambda path: path.stem[:3] if 'images' in str(path) else None))

name = 'cub-200'
Dataset.add(name, Dataset.from_label_func, 
            ('kaggle:tarunkr/caltech-birds-2011-dataset', 
             lambda path: path.parent.name.split('.')[1].lower() if 'images' in str(path) else None))

name = 'dogs-vs-cats'
Dataset.add(name, Dataset.from_label_func, 
            ('kaggle:dogs-vs-cats:train.zip', lambda path: path.name.split('.')[0]))

name = 'deep-weeds'
Dataset.add(name, Dataset.from_label_func, 
            ('kaggle:coreylammie/deepweedsx', 
             lambda path: path.with_suffix('').name.split('-')[-1]))

name = 'oxford-pets'
Dataset.add(name, Dataset.from_label_func, 
            ('kaggle:alexisbcook/oxford-pets', 
             lambda path: path.name.split('_')[0].lower() if 'images' in str(path) else None))

name = 'lego-brick'
Dataset.add(name, Dataset.from_label_func, 
            ('kaggle:joosthazelzet/lego-brick-images', 
             lambda path: path.name.split(' ')[0].lower() if str(path).startswith('dataset') else None))

name = 'satelite-plane'
Dataset.add(name, Dataset.from_label_func, 
            ('kaggle:rhammell/planesnet', 
             lambda path: path.name.split('__')[0]))

name = 'coil-100'
Dataset.add(name, Dataset.from_label_func, 
            ('kaggle:jessicali9530/coil100',
             lambda path: path.name.split('__')[0]))

name = 'flower-10'
Dataset.add(name, Dataset.from_label_func, 
            ('kaggle:aksha05/flower-image-dataset',
             lambda path: path.name.split('_')[0].lower()))

