from setuptools import setup, find_packages
import pathlib

requirements = [
    'kaggle',
    'pandas>=1.1.0',
    'tqdm',
]


_libinfo_py = pathlib.Path(__file__).parent/'d8/__init__.py'
with _libinfo_py.open('r') as f:
    for l in f.readlines():
        if '__version__' in l:
            __version__ = l.split('"')[1]

setup(
    name='d8',
    version=__version__,
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
    package_data={'d8':[]},
    entry_points={
        'console_scripts': [
            'd8 = d8.main:main',
            'ad = d8.main:main'
        ]
    },
)
