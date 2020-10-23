# This file is generated from object_detection/makeml.md automatically through:
#    d2lbook build lib
# Don't edit it directly

from autodatasets.object_detection import Dataset

def make_ml(name):
    camel_case = name.replace('-',' ').title().replace(' ', '')
    url = f'https://arcraftimages.s3-accelerate.amazonaws.com/Datasets/{camel_case}/{camel_case}PascalVOC.zip'
    return Dataset.from_voc(url, 'images', 'annotations')

names = ['sheep', 'paperprototype', 'raccoon', 'boggle-boards', 'plant-doc', 
         'hard-hat-workers', 'pistol', 'cars-and-traffic-signs', 'tomato', 
         'dice', 'potholes', 'ships', 'mask', 'chess', 'mobile-phones', 
         'glasses', 'road-signs', 'fruits', 'bikes', 'headphones', 'fish',
         'drone', 'car-license-plates', 'pets', 'faces', 'helmets', 'clothing',
         'hands', 'soccer-ball'
        ]

for name in names:
    Dataset.add(name, make_ml, [name])

