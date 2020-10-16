from setuptools import setup, find_packages

requirements = [
    'kaggle',
    'pandas'
]

setup(
    name='autodatasets',
    version='0.0.1',
    python_requires='>=3.5',
    author='',
    author_email='',
    url='',
    description='',
    license='Apache 2.0',
    packages=find_packages(),
    zip_safe=True,
    install_requires=requirements,
    include_package_data=True,
    package_data={'autodatasets':['datasets.csv', 'meta.csv', ]},
    entry_points={
        'console_scripts': [
            'autodatasets = autodatasets.main:main',
            'ad = autodatasets.main:main'
        ]
    },
)
