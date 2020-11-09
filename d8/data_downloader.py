import logging
import pathlib
import hashlib
import requests
import tqdm
import os
import contextvars
from typing import Optional, Union
from . import data_reader

DATAROOT = pathlib.Path.home()/'.d8'

class NameContext():
    """The context for dataset name.

    :Example:

    >>> with NameContext('my_name'):
    >>>   download(url)  # data will save into DATAROOT/my_name

    :param name: The dataset name
    """
    def __init__(self, name: Optional[str] = ''):
        self.name = name

    def __enter__(self):
        self._old_ctx = _current_name_context.get()
        _current_name_context.set(self)
        return self

    def __exit__(self, ptype, value, trace):
        _current_name_context.set(self._old_ctx)

_current_name_context = contextvars.ContextVar('name_context', default=NameContext())

def current_name():
    """Return the current dataset name, with an empty name in default.

    :Example:

    >>> with NameContext('my_name'):
    >>>   print(current_name)
    my_name
    """

    return _current_name_context.get().name

def download(url: str,
             save_dir: Optional[Union[str, pathlib.Path]] = None,
             extract: bool = False
             ) -> pathlib.Path:
    """Download a URL.

    Download the URL into the ``DATAROOT/save_dir`` folder, where ``DATAROOT`` is the ``.d8`` folder on the home directory, and return the saved file path.

    >>> download('https://d8.github.io/docs/index.html')
    PosixPath('/home/ubuntu/.d8/index.html')

    After data is download, it will compute and save the data sha1sum. If next time we save data into the same local path, and neither previous downloaded data and saved .sha1 file is changed, then we will skip the download. In best practice, we can download URLs into different folders

    >>> download('https://d8.github.io/docs/index.html', 'root')
    PosixPath('/home/ubuntu/.d8/root/index.html')
    >>> download('https://d8.github.io/docs/object_detection/index.html', 'detection')
    PosixPath('/home/ubuntu/.d8/detection/index.html')

    This function also supports to download a Kaggle competition or dataset, the format is ``kaggle:name`` for a competition, and ``kaggle:user/name`` for a dataset.

    >>> download('kaggle:house-prices-advanced-regression-techniques')
    PosixPath('/home/ubuntu/.d8/house-prices-advanced-regression-techniques.zip')

    :param url: The URL to be downloaded.
    :param save_dir: The directory to save the file, the default value is :py:func:`current_name`
    :return: The downloaded file path
    """
    # if isinstance(url, list) or isinstance(url, tuple):
    #     return [download(u, save_dir) for u in url]
    if save_dir is None: save_dir = current_name()
    if url.startswith('kaggle:'):
        save_filepath =  _download_kaggle(url[7:], save_dir)
    else:
        save_filepath = DATAROOT/save_dir/url.split('/')[-1]
        _download_url(url, save_filepath)
    if extract:
        return extract_file(save_filepath)
    return save_filepath

def extract_file(filepath, save_folder=None):
    filepath = pathlib.Path(filepath)
    if save_folder is None:
        save_folder = filepath.parent
    else:
        save_folder = pathlib.Path(save_folder)
    if filepath.suffix not in ['.zip', '.tar', '.gz', '.tgz']:
        return filepath
    reader = data_reader.create_reader(filepath)
    compressed_files = set(reader.list_files())
    existed_files = set(data_reader.create_reader(save_folder).list_files())
    uncompressed_files = compressed_files.difference(existed_files)
    if len(uncompressed_files):
        logging.info(f'Extracting {str(filepath)} to {str(save_folder.resolve())}')
        for p in tqdm.tqdm(uncompressed_files):
            out = save_folder / p
            if not out.parent.exists(): out.parent.mkdir(parents=True)
            with out.open('wb') as f:
                f.write(reader.open(p).read())
    return save_folder

def _download_kaggle(name: str, save_dir):
    """Download the dataset from Kaggle and return path of the zip file."""
    try:
        import kaggle
    except OSError:
        kps = [['KSAEGRGNLAEM_EU','atuatsoedtas'],
               ['KEA_GKGELY','dc7c97f6fc892a37af87008ae370fc78']]
        for kp in kps:
            os.environ[kp[0][::2]+kp[0][1::2]] = kp[1][::2]+kp[1][1::2]
        import kaggle
    names = name.split('/')
    path = DATAROOT/save_dir
    if len(names) == 2:  # it's a dataset
        files = names[1].split(':')
        if len(files) == 2:
            filepath = path/files[1]
            if not filepath.exists():
                logging.info(f'Downloading {files[1]} form Kaggle dataset {names[0]}/{files[0]} into {str(path)}.')
            kaggle.api.dataset_download_file(name, files[1], path)
        else:
            filepath = path/(names[-1]+'.zip')
            if not filepath.exists():
                logging.info(f'Downloading Kaggle dataset {name} into {str(path)}.')
            kaggle.api.dataset_download_files(name, path)
        return filepath
    # it's a competition
    files = name.split(':')
    if len(files) == 2:
        filepath = path/files[1]
        if not filepath.exists():
            logging.info(f'Downloading {files[1]} from Kaggle competition {files[0]} into {str(path)}.')
        kaggle.api.competition_download_file(files[0], files[1], path)
    else:
        filepath = path/(name+'.zip')
        if not filepath.exists():
            logging.info(f'Downloading Kaggle competition {name} into {str(path)}.')
        kaggle.api.competition_download_files(name, path)
    return filepath


# TODO(mli) only check the first 1GB data for big datasets
def _get_sha1(filepath: pathlib.Path):
    if not filepath.exists():
        return None
    sha1 = hashlib.sha1()
    with filepath.open('rb') as f:
        while True:
            data = f.read(2**20)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()

def _download_url(url: str, save_filepath: pathlib.Path, overwrite=False):
    sha1_filepath = pathlib.Path(str(save_filepath)+'.sha1')
    if sha1_filepath.exists() and save_filepath.exists() and not overwrite:
        with sha1_filepath.open('r') as f:
            sha1_hash = f.read().strip()
            if sha1_hash == _get_sha1(save_filepath):
                logging.debug(f'Found valid cache at {save_filepath}. Skip downloading.')
                return
    logging.info(f'Downloading {url} into {save_filepath.parent}')
    if not save_filepath.parent.exists():
        save_filepath.parent.mkdir(parents=True)
    r = requests.get(url, stream=True, verify=True)
    r.raise_for_status()
    total_size_in_bytes = int(r.headers.get('content-length', 0))
    progress_bar = tqdm.tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    block_size = 2**20
    with save_filepath.open('wb') as f:
        for chunk in r.iter_content(chunk_size=block_size):
            f.write(chunk)
            progress_bar.update(len(chunk))
    progress_bar.close()
    if progress_bar.n != total_size_in_bytes:
        logging.error(f'Only {progress_bar.n} bytes out of {total_size_in_bytes} bytes are downloaded.')
    else:
        sha1_hash = _get_sha1(save_filepath)
        with sha1_filepath.open('w') as f:
            f.write(sha1_hash+'\n')
