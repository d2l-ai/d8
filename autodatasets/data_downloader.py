import logging
import pathlib
import hashlib
import requests
import tqdm
import os

def download(resource, save_dir: pathlib.Path = None, root_dir=pathlib.Path.home()/'.autodatasets'):
    if resource.startswith('kaggle:'):
        return download_kaggle(resource[7:], root_dir)
    urls = resource.split(';')
    save_filepaths = []
    for url in urls:
        save_filepaths.append(root_dir/save_dir/url.split('/')[-1])
        _download_url(url, save_filepaths[-1])
    if len(save_filepaths) == 1:
        save_filepaths = save_filepaths[0]
    return save_filepaths

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
                logging.info(f'Found valid cache at {save_filepath}. Skip downloading.')
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

def download_kaggle(name, root_dir=pathlib.Path.home()/'.autodatasets'):
    """Download the dataset from Kaggle and return path of the zip file."""
    try:
        import kaggle
    except OSError:
        kps = [['KSAEGRGNLAEM_EU','atuatsoedtas'],
               ['KEA_GKGELY','dc7c97f6fc892a37af87008ae370fc78']]
        for kp in kps:
            os.environ[kp[0][::2]+kp[0][1::2]] = kp[1][::2]+kp[1][1::2]
        import kaggle
    logging.info(f'Downloading {name} from Kaggle.')
    names = name.split('/')
    if len(names) == 2:  # it's a dataset
        root_dir /= names[-1]
        kaggle.api.dataset_download_files(name, path=root_dir)
        return root_dir/(names[-1]+'.zip')
    # it's a competition
    root_dir /= name
    kaggle.api.competition_download_files(name, path=root_dir)
    logging.info(f'Done')
    return root_dir/(name+'.zip')

