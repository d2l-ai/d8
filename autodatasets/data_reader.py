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
import numpy as np

class Reader(abc.ABC):
    """The base class of the data reader.

    :param root: The root path.
    """
    def __init__(self, root: pathlib.Path):
        if not root.exists():
            raise NameError(f'{root} doesn\'t exists')
        self._root = root

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

    @abc.abstractproperty
    def size(self) -> int:
        """Returns the total size in bytes."""
        pass

    def list_files(self, extensions: Sequence =[]) -> List[pathlib.Path]:
        """List all files.

        :param extensions: If specified, then only keep files with extensions in this list.
        :return: The list of file paths.
        """
        # remove folders
        files = [f for f in self._list_all() if not (f.name.endswith('\\') or f.name.endswith('/'))]
        if extensions:
            files = [f for f in files if f.suffix in set(extensions)]
        return files

    def list_images(self) -> List[pathlib.Path]:
        """List all image files.

        :return: The list of image file paths.
        """
        image_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))
        return self.list_files(image_extensions)

    def read_image(self, filepath: Union[str, pathlib.Path],
                   max_width: Optional[int] = None,
                   max_height: Optional[int] = None) -> np.ndarray:
        """Read an image.

        :param filepath: The image filepath.
        :param max_width: The maximal width in pixel for the returned image.
            Specifying it with a small value may accelerate the reading.
        :param max_height: The maximal height in pixel for the returned image.
            Specifying it with a small value may accelerate the reading.
        :return: The image as a numpy array.
        """
        img = PIL.Image.open(self.open(filepath))
        ratio = 0
        if max_width: ratio = max(ratio, img.size[0] / max_width)
        if max_height: ratio = max(ratio, img.size[1] / max_height)
        if ratio > 0:
            # TODO(mli) handle gray images
            img.draft('RGB',(int(img.size[0]/ratio), int(img.size[1]/ratio)))
        return np.asarray(img)

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
    root = pathlib.Path(root)
    # if root.is_dir():
    #     return DirReader(root)
    if root.suffix == '.zip':
        return ZipReader(root)
    # if root.suffix == '.tar':
    #     return TarReader(root)
    raise NameError(f'Not support {root}')


class ZipReader(Reader):
    """A data reader to read from a zip file.

    :param root: The root path.
    """
    def __init__(self, root: pathlib.Path):
        super().__init__(root)
        self._root_fp = zipfile.ZipFile(self._root, 'r')

    @property
    def size(self):
        return self._root.stat().st_size

    def open(self, path: Union[str, pathlib.Path]):
        return self._root_fp.open(str(path))

    def _list_all(self):
        filenames = [file.filename for file in self._root_fp.infolist()
                     if not '__MACOSX' in file.filename]
        return [pathlib.Path(fn) for fn in filenames]

# class DirReader(Reader):
#     pass

# class TarReader(Reader):
#     pass
