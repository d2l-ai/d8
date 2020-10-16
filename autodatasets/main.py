import argparse
import sys
import logging
import pandas as pd
import pathlib
import hashlib
import requests
import tqdm

from . import file_reader

logging.basicConfig(format='[autodatasets:%(filename)s:L%(lineno)d] %(levelname)-6s %(message)s')
logging.getLogger().setLevel(logging.INFO)
pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



def _is_null(series):
    return series.isnull().values.all()

class AutoDatasets():
    def __init__(self):
        self._meta_path = pathlib.Path(__file__).parent/'datasets.csv'
        self._meta = pd.read_csv(self._meta_path)
        self._verify_meta()
        self._write_meta()

    def _write_meta(self):
        self._meta.sort_values(by=['type','task','name']).to_csv(self._meta_path, index=False)

    def _verify_meta(self):
        duplicated_names = self._meta[self._meta.duplicated(subset=['name'])]['name'].to_list()
        if duplicated_names:
            raise KeyError(f'Duplicated names {duplicated_names} in {self._meta_path}')


    def download(self, name):
        row_idx = self._meta['name']==name
        entry = self._meta[row_idx]
        if len(entry) == 0:
            raise KeyError(f'Dataset name {name} not found in {self._meta_path}')
        urls = entry['resource'].to_list()[0].split(';')
        # has_sha1 = not _is_null(entry['sha1sum'])
        # if has_sha1:
        #     sha1s = entry['sha1sum'].to_list()[0].split(';')
        #     if len(sha1s) != len(urls):
        #         has_sha1 = False
        # if not has_sha1:
        #     sha1s = [None] * len(urls)
        save_filepaths = []
        for url in urls:
            filename = url.split('/')[-1]
            save_filepaths.append(pathlib.Path.home()/'.autodatasets'/name/filename)
            download(url, save_filepaths[-1])
        # if not has_sha1:
        #     sha1s, size, num_files = [], 0, 0
        #     for filepath in save_filepaths:
        #         sha1s.append(get_sha1(filepath))
        #         rd = file_reader.create_reader(filepath)
        #         size += rd.size // 2**20
        #         num_files += len(rd.list_files())
        #     self._meta.loc[row_idx, 'sha1sum'] = ';'.join(sha1s)
        #     self._meta.loc[row_idx, 'size(MB)'] = size
        #     self._meta.loc[row_idx, 'num_files'] = num_files
        #     self._write_meta()
        return save_filepaths

    def list(self, **kwargs):
        meta = self._meta
        for k, v in kwargs.items():
            if v:
                if v not in meta[k]:
                    raise KeyError(f'Not found {v} in {k}')
                meta = meta[meta[k]==v]
        return meta.drop(columns=['resource','sha1sum'])

def list_datasets(name=None, type=None, task=None):
    ad = AutoDatasets()
    filters = {'name':name, 'type':type, 'task':task}
    return ad.list(**filters)

def download_dataset(name):
    ad = AutoDatasets()
    return ad.download(name)

def main():

#     parser = argparse.ArgumentParser(description='''
# Auto Datasets: download ml datasets.

# Run autodatasets command -h to get the help message for each command.
# ''')
#     parser.add_argument('command', nargs=1, choices=['list', 'download', 'update'])
#     args = parser.parse_args(sys.argv[1:2])

    cmd = sys.argv[1]
    if cmd == 'list':
        print(list_datasets())
    elif cmd == 'download':
        download_dataset(sys.argv[2])

if __name__ == "__main__":
    main()
