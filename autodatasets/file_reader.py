from typing import Union, List
import zipfile
import tarfile
import pathlib
import abc
import mimetypes

def create_reader(root: Union[str, pathlib.Path]):
    root = pathlib.Path(root)
    if root.is_dir():
        return FolderReader(root)
    if root.suffix == '.zip':
        return ZipReader(root)
    if root.suffix == '.tar':
        return TarReader(root)
    raise NameError(f'Not support {root}')

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
    def _list_all(self) -> List[pathlib.Path]:
        pass

    @abc.abstractproperty
    def size(self) -> int:
        pass

    def list_folders(self):
        return [f for f in self._list_all() if (f.name.endswith('\\') or f.name.endswith('/'))]

    def list_files(self, extensions=None):
        # remove folders
        files = [f for f in self._list_all() if not (f.name.endswith('\\') or f.name.endswith('/'))]
        if extensions:
            extensions = set(extensions)
            files = [f for f in files if f.suffix in extensions]
        return files

    def list_images(self) -> List[pathlib.Path]:
        image_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))
        return self.list_files(image_extensions)

class ZipReader(Reader):
    def __init__(self, root: pathlib.Path):
        super().__init__(root)
        self._root_fp = zipfile.ZipFile(self._root, 'r')

    @property
    def size(self):
        return self._root.stat().st_size

    def open(self, path: Union[str, pathlib.Path]):
        return self._root_fp.open(str(path))

    def _list_all(self):
        return [pathlib.Path(file.filename) for file in self._root_fp.infolist()
                if not '__MACOSX' in file.filename]

class FolderReader(Reader):
    pass

class TarReader(Reader):
    pass
