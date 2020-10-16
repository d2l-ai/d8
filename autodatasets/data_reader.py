from typing import Union, List, Sequence
import zipfile
import tarfile
import pathlib
import abc
import mimetypes
import os
import PIL
import pandas as pd
import io

__all__ = ['create_reader', 'get_image_info']

class Reader(abc.ABC):
    def __init__(self, root: pathlib.Path):
        if not root.exists():
            raise NameError(f'{root} doesn\'t exists')
        self._root = root


    @abc.abstractclassmethod
    def open(self, path: Union[str, pathlib.Path]):
        """Open a path relative to root."""
        pass

    @abc.abstractclassmethod
    def _get_all(self, subdirectories=[]) -> List[pathlib.Path]:
        pass

    @abc.abstractproperty
    def size(self) -> int:
        pass

    def get_files(self, extensions=[], subdirectories=[]):
        # remove folders
        files = [f for f in self._get_all(subdirectories) if not (f.name.endswith('\\') or f.name.endswith('/'))]
        if extensions:
            files = [f for f in files if f.suffix in set(extensions)]
        return files

    def get_images(self, subdirectories=[]) -> List[pathlib.Path]:
        image_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))
        return self.get_files(image_extensions, subdirectories)

def get_image_info(reader: Reader, image_paths: Sequence[str]) -> pd.DataFrame:
    rows = []
    for img_path in image_paths:
        raw = reader.open(img_path).read()
        img = PIL.Image.open(io.BytesIO(raw))
        rows.append({'img_size(KB)':len(raw)/2**10,
                     'img_width(px)':img.size[0], 'img_height(px)':img.size[1]})
    return pd.DataFrame(rows)


def create_reader(root: Union[str, pathlib.Path]) -> Reader:
    root = pathlib.Path(root)
    if root.is_dir():
        return DirReader(root)
    if root.suffix == '.zip':
        return ZipReader(root)
    if root.suffix == '.tar':
        return TarReader(root)
    raise NameError(f'Not support {root}')


class ZipReader(Reader):
    def __init__(self, root: pathlib.Path):
        super().__init__(root)
        self._root_fp = zipfile.ZipFile(self._root, 'r')

    @property
    def size(self):
        return self._root.stat().st_size

    def open(self, path: Union[str, pathlib.Path]):
        return self._root_fp.open(str(path))

    def _get_all(self, subdirectories=[]):
        filenames = [file.filename for file in self._root_fp.infolist()
                     if not '__MACOSX' in file.filename]
        if subdirectories:
            filenames = [fn for fn in filenames
                         if any([fn.startswith(str(subdir)) for subdir in subdirectories])]
        return [pathlib.Path(fn) for fn in filenames]

class DirReader(Reader):
    pass

class TarReader(Reader):
    pass
