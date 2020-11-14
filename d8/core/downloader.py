# This file is generated from core/downloader.md automatically through:
#    d2lbook build lib
# Don't edit it directly

#@save_all
#@hide_all
import contextvars
import logging
import os
import pathlib
import tempfile
import unittest
from typing import Optional, Union

import requests
import tqdm
import xxhash

from d8 import core

__all__ = ['download', 'DATAROOT', 'NameContext']

DATAROOT = pathlib.Path.home()/'.d8'

class NameContext():
    """The context for dataset name.

    :param name: The dataset name
    """
    def __init__(self, name: str = ''):
        self.name = name

    def __enter__(self):
        self._old_ctx = _current_name_context.get()
        _current_name_context.set(self)
        return self

    def __exit__(self, ptype, value, trace):
        _current_name_context.set(self._old_ctx)

_current_name_context = contextvars.ContextVar(
    'name_context', default=NameContext())

def current_name() -> str:
    """Return the current dataset name, with an empty name in default."""
    return _current_name_context.get().name

class TestNameContext(unittest.TestCase):
    def test_current_name(self):
        with NameContext('name1'):
            self.assertEqual(current_name(), 'name1')
            with NameContext('d823'):
                self.assertEqual(current_name(), 'd823')

def _add_suffix(file_path: pathlib.Path, suffix: str) -> pathlib.Path:
    return file_path.with_suffix(file_path.suffix+suffix)

def _get_xxhash(file_path: pathlib.Path) -> str:
    """Compute the hash of the content of file_pathpath"""
    n = file_path.stat().st_size
    x = xxhash.xxh128()
    m = 2 ** 23  # read 8MB each time
    with file_path.open('rb') as f:
        if n < m * 128: # <= 1GB, check the whole data
            while True:
                data = f.read(m)
                if not data: break
                x.update(data)
        else:
            for _ in range(64):  # check the first 0.5GB
                x.update(f.read(m))
            f.seek(-64*m, 2)
            for _ in range(64):  # check the last 0.5GB
                x.update(f.read(m))
    return x.hexdigest()

def _match_hash(file_path: pathlib.Path) -> bool:
    """Return true if the saved hash matches the contents of file_path."""
    hash_file_path = _add_suffix(file_path, '.xxh')
    if hash_file_path.is_file() and file_path.is_file():
        with hash_file_path.open('r') as f:
            saved_hash = f.read().strip()
            if saved_hash == _get_xxhash(file_path):
                return True
    return False

def _save_hash(file_path: pathlib.Path) -> None:
    """Save the hash for file_path."""
    if not file_path.is_file(): return
    hash_file_path = _add_suffix(file_path, '.xxh')
    with hash_file_path.open('w') as f:
        f.write(_get_xxhash(file_path)+'\n')


class TestHash(unittest.TestCase):
    def test_hash(self):
        f = tempfile.NamedTemporaryFile()
        f.write(b'12345678')
        fn = pathlib.Path(f.name)
        self.assertEqual(_match_hash(fn), False)
        _save_hash(fn)
        self.assertEqual(_match_hash(fn), True)

def _download_kaggle(url: str, save_dir:str) -> pathlib.Path:
    """Download the dataset from Kaggle and return path of the zip file."""
    try:
        import kaggle
    except OSError:
        kps = [['KSAEGRGNLAEM_EU','atuatsoedtas'],
               ['KEA_GKGELY','dc7c97f6fc892a37af87008ae370fc78']]
        for kp in kps:
            os.environ[kp[0][::2]+kp[0][1::2]] = kp[1][::2]+kp[1][1::2]
        import kaggle

    # parse url
    if '/' in url:
        user = url.split('/')[0]
        url = url[len(user)+1:]
    else:
        user = ''

    if '#' in url:
        dataset, file = url.split('#')
    elif '?select=' in url:
        dataset, file = url.split('?select=')
    else:
        dataset, file = url, ''
    dataset = dataset.split('/')[0]
    file = file.replace('+', ' ')

    # check if already exists
    full_dir = DATAROOT/save_dir
    file_path = full_dir/(file if file else dataset)
    if _match_hash(file_path): return file_path
    zip_file_path = _add_suffix(file_path, '.zip')
    if _match_hash(zip_file_path): return zip_file_path

    # download
    if user and user != 'c':
        if file:
            logging.info(f'Downloading {file} from Kaggle dataset {user}/{dataset} into {full_dir}')
            kaggle.api.dataset_download_file(f'{user}/{dataset}', file, full_dir)
        else:
            logging.info(f'Downloading Kaggle dataset {user}/{dataset} into {full_dir}')
            kaggle.api.dataset_download_files(f'{user}/{dataset}', full_dir)
    else:
        if file:
            logging.info(f'Downloading {file} from Kaggle competition {dataset} into {full_dir}.')
            kaggle.api.competition_download_file(dataset, file, full_dir)
        else:
            logging.info(f'Downloading Kaggle competition {dataset} into {full_dir}.')
            kaggle.api.competition_download_files(dataset, full_dir)

    # check saved
    if ' ' in file:
        save_path = pathlib.Path(str(file_path).replace(' ', '%20'))
        if save_path.is_file():
            save_path.rename(file_path)
        save_path = pathlib.Path(str(zip_file_path).replace(' ', '%20'))
        if save_path.is_file():
            save_path.rename(zip_file_path)

    if file_path.is_file():
        _save_hash(file_path)
        return file_path
    if zip_file_path.is_file():
        _save_hash(zip_file_path)
        return zip_file_path
    raise FileNotFoundError(f'Not found downloaded file as {file_path} or {zip_file_path}')
    return ''


def _download_url(url: str, save_dir:str):
    """Download from a URL, save it to file_path."""
    file_path = DATAROOT/save_dir/url.split('/')[-1]
    if _match_hash(file_path): return file_path
    logging.info(f'Downloading {url} into {file_path}')
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)
    r = requests.get(url, stream=True, verify=True)
    r.raise_for_status()
    total_size_in_bytes = int(r.headers.get('content-length', 0))
    progress_bar = tqdm.tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    block_size = 2**20
    with file_path.open('wb') as fb:
        for chunk in r.iter_content(chunk_size=block_size):
            fb.write(chunk)
            progress_bar.update(len(chunk))
    progress_bar.close()
    if progress_bar.n < total_size_in_bytes:
        raise IOError(f'Only {progress_bar.n} bytes out of {total_size_in_bytes} bytes are downloaded.')
    _save_hash(file_path)
    return file_path

def _extract_file(file_path: pathlib.Path) -> pathlib.Path:
    """Extract the file_path, returns the saved folder."""
    save_folder = file_path.parent
    if file_path.suffix not in ['.zip', '.tar', '.gz', '.tgz']:
        return save_folder
    reader = core.create_reader(file_path)
    compressed_files = set(reader.list_files())
    existed_files = set(core.create_reader(save_folder).list_files())
    uncompressed_files = compressed_files.difference(existed_files)
    if len(uncompressed_files):
        logging.info(f'Extracting {str(file_path)} to {str(save_folder.resolve())}')
        for p in tqdm.tqdm(uncompressed_files):
            out = save_folder / p
            if not out.parent.exists(): out.parent.mkdir(parents=True)
            with out.open('wb') as f:
                f.write(reader.open(p).read())
    return save_folder

def download(url: str, save_dir: Optional[str] = None, extract: bool = False
            ) -> pathlib.Path:
    """Download a URL and return the file path. 

    :param url: The URL to be downloaded.
    :param save_dir: The directory to save the file, the default value is :py:func:`current_name`
    :param extract: If True, then extract the downloaded file into its current directory. 
    :return: The downloaded file path or its directory if extracted. 
    """
    if save_dir is None:
        save_dir = current_name()
    kaggle_prefix = ['kaggle://', 'https://www.kaggle.com/']
    downloaded = False
    for prefix in kaggle_prefix:
        if url.startswith(prefix):
            file_path =  _download_kaggle(url[len(prefix):], save_dir)
            downloaded = True
    if not downloaded:
        file_path = _download_url(url, save_dir)
    if extract:
        file_path = _extract_file(file_path)
    return file_path

class TestDownload(unittest.TestCase):
    def setUp(self):
        self.name = 'test_download'
        self.dir = DATAROOT/self.name
        for fn in self.dir.glob('*'):
            fn.unlink()

    def test_kaggle(self):
        for _ in range(2):
            self.assertEqual(_download_kaggle('titanic', self.name),
                             self.dir/'titanic.zip')
            self.assertEqual(_download_kaggle('titanic#train.csv', self.name),
                             self.dir/'train.csv')
            self.assertEqual(_download_kaggle('titanic?select=train.csv', self.name),
                             self.dir/'train.csv')
            self.assertEqual(_download_kaggle('shebrahimi/financial-distress?select=Financial+Distress.csv', self.name),
                             self.dir/'Financial Distress.csv.zip')
            self.assertEqual(_download_kaggle('fabdelja/autism-screening-for-toddlers#Toddler Autism dataset July 2018.csv', self.name),
                             self.dir/'Toddler Autism dataset July 2018.csv')
    def test_url(self):
        for _ in range(2):
            self.assertEqual(
                _download_url('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', self.name),
                self.dir/'iris.data')

    def test_download(self):
        for _ in range(2):
            self.assertEqual(
                download('https://www.kaggle.com/oddyvirgantara/on-time-graduation-classification', self.name),
                self.dir/'on-time-graduation-classification.zip')
            self.assertEqual(
                download('https://www.kaggle.com/oddyvirgantara/on-time-graduation-classification', self.name, extract=True),
                self.dir)
            self.assertEqual(
                download('https://www.kaggle.com/c/titanic/data?select=train.csv', self.name),
                self.dir/'train.csv')





if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

