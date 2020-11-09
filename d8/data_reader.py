from typing import Union, List, Sequence, Optional
import zipfile
import tarfile
import pathlib
import abc
import mimetypes
import os
import PIL
import pandas as pd
import io
import glob

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Reader(abc.ABC):
    """The base class of the data reader.

    :param root: The root path.
    """
    def __init__(self, root: pathlib.Path):
        root = pathlib.Path(root)
        if not root.exists():
            raise NameError(f'{root} doesn\'t exists')
        self._root = root

    def __eq__(self, other: 'Reader'):
        return self._root == other._root

    @abc.abstractclassmethod
    def open(self, path: Union[str, pathlib.Path]):
        """Open file and return a stream.

        :param path: The relative filepath to the root
        :return: A file object depends on the reader type.
        """
        pass

    @abc.abstractclassmethod
    def _list_all(self) -> List[pathlib.Path]:
        pass

    def list_files(self, extensions: Sequence[str] =[], subfolders: Sequence[str] = []) -> List[pathlib.Path]:
        """List all files.

        :param extensions: If specified, then only keep files with extensions in this list.
        :return: The list of file paths.
        """
        # remove folders
        files = self._list_all()
        if extensions:
            files = [f for f in files if f.suffix.lower() in set(extensions)]
        if subfolders:
            files = [f for f in files if any([str(f).startswith(s) for s in subfolders])]
        return files

    def list_images(self, subfolders: Sequence[str] = []) -> List[pathlib.Path]:
        """List all image files.

        :return: The list of image file paths.
        """
        image_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))
        return self.list_files(image_extensions, subfolders)

    def read_image(self, filepath: Union[str, pathlib.Path],
                   max_width: Optional[int] = None,
                   max_height: Optional[int] = None):
        """Read an image.

        :param filepath: The image filepath.
        :param max_width: The maximal width in pixel for the returned image.
            Specifying it with a small value may accelerate the reading.
        :param max_height: The maximal height in pixel for the returned image.
            Specifying it with a small value may accelerate the reading.
        :return: The image as a numpy array.
        """
        img = PIL.Image.open(self.open(filepath))
        if img.mode != 'RGB': img = img.convert('RGB')
        ratio = 0
        if max_width: ratio = max(ratio, img.size[0] / max_width)
        if max_height: ratio = max(ratio, img.size[1] / max_height)
        if ratio > 0:
            # TODO(mli) handle gray images
            img.draft('RGB',(int(img.size[0]/ratio), int(img.size[1]/ratio)))
        return img

    def get_image_info(self, image_paths: Sequence[str]) -> pd.DataFrame:
        """Query image information such as size, width and height.

        :param reader: The data reader.
        :param image_paths: The image filepaths to query.
        :return: The results with each image in a row.
        """
        rows = []
        for img_path in image_paths:
            raw = self.open(img_path).read()
            img = PIL.Image.open(io.BytesIO(raw))
            rows.append({'filepath':img_path, 'size (KB)':len(raw)/2**10,
                        'width':img.size[0], 'height':img.size[1]})
        return pd.DataFrame(rows)


def create_reader(root: Union[str, pathlib.Path]) -> Reader:
    """The factory function to create a data reader.

    Based on the root path type, such as folder, zip file, tar file, proper data
    reader will be created.

    :param root: The root path, it must exists.
    :return: The created data reader
    """
    if isinstance(root, (tuple, list)):
        if len(root) == 1:
            return create_reader(root[0])
        return MultiReader(root)
    roots = list(glob.glob(str(root)))
    if len(roots) == 0:
        raise ValueError('Not found {root}')
    if len(roots) > 1:
        return create_reader(roots)
    root = pathlib.Path(roots[0])
    if root.is_dir():
        return FolderReader(root)
    if root.suffix == '.zip':
        return ZipReader(root)
    if root.suffix in ['.tar', '.tgz', '.gz']:
        return TarReader(root)
    raise ValueError(f'Not support {root}')

class FolderReader(Reader):
    def __init__(self, root: pathlib.Path):
        super().__init__(root)

    def open(self, path: Union[str, pathlib.Path]):
        return (self._root/path).open('rb')

    def _list_all(self):
        return [p.relative_to(self._root) for p in self._root.glob('**/*') if not p.is_dir()]

# Note that zip reader doesn't support multiprocess https://bugs.python.org/issue39363
class ZipReader(Reader):
    """A data reader to read from a zip file.

    :param root: The root path.
    """
    def __init__(self, root: pathlib.Path):
        super().__init__(root)
        self._root_fp = zipfile.ZipFile(self._root, 'r')

    def open(self, path: Union[str, pathlib.Path]):
        return self._root_fp.open(str(path))

    def _list_all(self):
        filenames = [file.filename for file in self._root_fp.infolist()
                     if not '__MACOSX' in file.filename and not file.is_dir()]
        return [pathlib.Path(fn) for fn in filenames]

class TarReader(Reader):
    """A data reader to read from a tar file.

    :param root: The root path.
    """
    def __init__(self, root: pathlib.Path):
        super().__init__(root)
        self._root_fp = tarfile.open(self._root, 'r')

    def open(self, path: Union[str, pathlib.Path]):
        return self._root_fp.open(str(path))

    def _list_all(self):
        return [pathlib.Path(fn) for fn in self._root_fp.getmembers()]

class MultiReader(Reader):
    """Reading multiple roots at the same time."""
    def __init__(self, roots: Union[Sequence[pathlib.Path], Sequence[str]]):
        self._readers = dict()
        for root in roots:
            root = pathlib.Path(root)
            self._readers[root.with_suffix('').name] = create_reader(root)

    def open(self, path: Union[str, pathlib.Path]):
        path = pathlib.Path(path)
        base = path.parents[len(path.parents)-2]
        if base.name not in self._readers:
            raise ValueError(f'The top-level path {base.name} should in {self._readers.keys()}')
        return self._readers[base.name].open(path.relative_to(base))

    def _list_all(self) -> List[pathlib.Path]:
        rets = []
        for name, reader in self._readers.items():
            for p in reader._list_all():
                rets.append(name/p)
        return rets
