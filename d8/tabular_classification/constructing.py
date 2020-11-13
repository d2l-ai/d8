# This file is generated from tabular_classification/constructing.md automatically through:
#    d2lbook build lib
# Don't edit it directly

from d8.tabular_classification import Dataset

#@save_cell

_UCI = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'

from_csv_meta = [
    {'name'    : 'iris',
     'url'     : _UCI+'iris/iris.data',
     'label'   : -1,
     'columns' : ['sepal length', 'sepal width', 'petal length', 'petal width','class']},
    {'name'    : 'adult',
     'url'     : [_UCI+'adult/adult.data', _UCI+'adult/adult.test'],
     'label'   : -1,
     'columns' : ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary']},
    {'name'    : 'titanic',
     'url'     : 'https://www.kaggle.com/c/titanic/data?select=train.csv',
     'label'   : 1},
    {'name'    : 'credit-card-fraud',
     'url'     : 'https://www.kaggle.com/mlg-ulb/creditcardfraud?select=creditcard.csv',
     'label'   : -1},
    {'name'    : 'mushroom',
     'url'     : 'https://www.kaggle.com/uciml/mushroom-classification?select=mushrooms.csv',
     'label'   : 0},
    {'name'    : 'glass',
     'url'     : 'https://www.kaggle.com/uciml/glass?select=glass.csv',
     'label'   : -1},
    {'name'    : 'mobile-price',
     'url'     : 'https://www.kaggle.com/iabhishekofficial/mobile-price-classification?select=train.csv',
     'label'   : 'price_range'},
    {'name'    : 'fetal-health',
     'url'     : 'https://www.kaggle.com/andrewmvd/fetal-health-classification?select=fetal_health.csv',
     'label'   : -1},
    {'name'    : 'drug',
     'url'     : 'https://www.kaggle.com/prathamtripathi/drug-classification?select=drug200.csv',
     'label'   : -1},
    {'name'    : 'asteroids',
     'url'     : 'https://www.kaggle.com/shrutimehta/nasa-asteroids-classification?select=nasa.csv',
     'label'   : -1},
    {'name'    : 'taekwondo',
     'url'     : 'https://www.kaggle.com/ali2020armor/taekwondo-techniques-classification?select=Taekwondo_Technique_Classification_Stats.csv',
     'label'   : 0},
    {'name'    : 'cs-go',
     'url'     : 'https://www.kaggle.com/christianlillelund/csgo-round-winner-classification?select=csgo_round_snapshots.csv',
     'label'   : -1},
    {'name'    : 'wine',
     'url'     : 'https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009?select=winequality-red.csv',
     'label'   : -1},
    {'name'    : 'churn',
     'url'     : 'https://www.kaggle.com/shrutimechlearn/churn-modelling?select=Churn_Modelling.csv',
     'label'   : -1},
    {'name'    : 'usp_drug',
     'url'     : 'https://www.kaggle.com/danofer/usp-drug-classification?select=usp_drug_classification.csv',
     'label'   : 0},
    {'name'    : 'rain_au',
     'url'     : 'https://www.kaggle.com/jsphyg/weather-dataset-rattle-package?select=weatherAUS.csv',
     'label'   : -1},
    {'name'    : 'automobile-customer',
     'url'     : 'https://www.kaggle.com/kaushiksuresh147/customer-segmentation?select=Train.csv',
     'label'   : -1},
#     {'name'    : 'belgium_population',
#      'url'     : 'https://www.kaggle.com/sameerkulkarni91/belgium-population-classification?select=BELGIUM_POPULATION_STRUCTURE_2018.csv',
#      'label'   : -1},
    {'name'    : 'loan',
     'url'     : 'https://www.kaggle.com/burak3ergun/loan-data-set?select=loan_data_set.csv',
     'label'   : -1},
    {'name'    : 'crime',
     'url'     : 'https://www.kaggle.com/abidaaslam/crime?select=Crime1.csv',
     'label'   : 'Category'},
    {'name'    : 'toddler-autism',
     'url'     : 'https://www.kaggle.com/fabdelja/autism-screening-for-toddlers?select=Toddler+Autism+dataset+July+2018.csv',
     'label'   : -1},
    #  "Financial Distress" if it is greater than -0.50 the company should be considered as healthy (0). Otherwise, it would be regarded as financially distressed (1).
#     {'name'    : 'financial-distress',
#      'url'     : 'https://www.kaggle.com/shebrahimi/financial-distress#Financial Distress.csv',
#      'label'   : 'Financial Distress'},
    {'name'    : 'sf-crime',
     'url'     : 'https://www.kaggle.com/kaggle/san-francisco-crime-classification?select=train.csv',
     'label'   : 'Category'},
    {'name'    : 'dmoz-url',
     'url'     : 'https://www.kaggle.com/revanthrex/url-classification?select=URL+Classification.csv',
     'label'   : -1,
     'columns' : ['ID', 'URL', 'Category']},
#     {'name'    : '',
#      'url'     : '',
#      'label'   : ''},
#     {'name'    : '',
#      'url'     : '',
#      'label'   : ''},
]


for x in from_csv_meta:
    Dataset.add(x['name'], Dataset.from_csv, [x['url'], x['label'], x.get('columns', None), x.get('df_func', None)])



