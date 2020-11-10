# The `Dataset` API

```{.python .input  n=5}
%load_ext autoreload
%autoreload 2
```

```{.python .input}
from typing import
from d8 import base_dataset
import pandas as pd

class Dataset(base_dataset.ClassificationDataset):

    def show(self, layout=(2,8)) -> None:
        pass

    def __getitem__(self, idx):
        if idx < 0 or idx > self.__len__():
            raise IndexError(f'index {idx} out of range [0, {self.__len__()})')
        img_path = self.df['file_path'][idx]
        lbl_path = self.df['label_file_path'][idx]
        img = self.reader.read_image(file_path)
        return np.array(img), self.df['class_name'][idx]
```

```{.python .input  n=13}



base_dataset.BaseDataset.add('camseq01',
                         base_dataset.BaseDataset.from_df_func,
                             [['kaggle:carlolepelaars/camseq-semantic-segmentation',
                                 #'http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamSeq01/CamSeq01.zip',
'http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamSeq01/label_colors.txt'], lambda reader: pd.DataFrame()]
                            )



ds = base_dataset.BaseDataset.get('camseq01')

```

```{.json .output n=13}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "[d8:base_dataset.py:L34] WARNING No example is found as `df` is empty.\n[d8:base_dataset.py:L35] WARNING You may use `ds.reader.list_files()` to check all files.\n"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[PosixPath('/Users/mli/.d8/camseq01'), PosixPath('/Users/mli/.d8/camseq01')]\n"
 }
]
```

```{.python .input  n=19}
[p for p in ds.reader.list_images() if '_L' in str(p)]
```

```{.json .output n=19}
[
 {
  "data": {
   "text/plain": "[PosixPath('0016E5_08085_L.png'),\n PosixPath('0016E5_08045_L.png'),\n PosixPath('0016E5_08139_L.png'),\n PosixPath('0016E5_08061_L.png'),\n PosixPath('0016E5_08019_L.png'),\n PosixPath('0016E5_08141_L.png'),\n PosixPath('0016E5_08143_L.png'),\n PosixPath('0016E5_08063_L.png'),\n PosixPath('0016E5_08047_L.png'),\n PosixPath('0016E5_07969_L.png'),\n PosixPath('0016E5_08087_L.png'),\n PosixPath('0016E5_08043_L.png'),\n PosixPath('0016E5_08083_L.png'),\n PosixPath('0016E5_08147_L.png'),\n PosixPath('0016E5_07989_L.png'),\n PosixPath('0016E5_08067_L.png'),\n PosixPath('0016E5_08065_L.png'),\n PosixPath('0016E5_08145_L.png'),\n PosixPath('0016E5_08039_L.png'),\n PosixPath('0016E5_08081_L.png'),\n PosixPath('0016E5_08041_L.png'),\n PosixPath('0016E5_08119_L.png'),\n PosixPath('0016E5_07971_L.png'),\n PosixPath('0016E5_08027_L.png'),\n PosixPath('0016E5_08107_L.png'),\n PosixPath('0016E5_08123_L.png'),\n PosixPath('0016E5_07995_L.png'),\n PosixPath('0016E5_08003_L.png'),\n PosixPath('0016E5_08001_L.png'),\n PosixPath('0016E5_08159_L.png'),\n PosixPath('0016E5_08121_L.png'),\n PosixPath('0016E5_07997_L.png'),\n PosixPath('0016E5_08079_L.png'),\n PosixPath('0016E5_08105_L.png'),\n PosixPath('0016E5_08025_L.png'),\n PosixPath('0016E5_07973_L.png'),\n PosixPath('0016E5_08059_L.png'),\n PosixPath('0016E5_08101_L.png'),\n PosixPath('0016E5_08021_L.png'),\n PosixPath('0016E5_07977_L.png'),\n PosixPath('0016E5_08099_L.png'),\n PosixPath('0016E5_08005_L.png'),\n PosixPath('0016E5_07993_L.png'),\n PosixPath('0016E5_08125_L.png'),\n PosixPath('0016E5_07991_L.png'),\n PosixPath('0016E5_08127_L.png'),\n PosixPath('0016E5_08007_L.png'),\n PosixPath('0016E5_07975_L.png'),\n PosixPath('0016E5_08023_L.png'),\n PosixPath('0016E5_08103_L.png'),\n PosixPath('0016E5_08057_L.png'),\n PosixPath('0016E5_08097_L.png'),\n PosixPath('0016E5_07979_L.png'),\n PosixPath('0016E5_08153_L.png'),\n PosixPath('0016E5_08073_L.png'),\n PosixPath('0016E5_08071_L.png'),\n PosixPath('0016E5_08129_L.png'),\n PosixPath('0016E5_08151_L.png'),\n PosixPath('0016E5_08009_L.png'),\n PosixPath('0016E5_08095_L.png'),\n PosixPath('0016E5_08055_L.png'),\n PosixPath('0016E5_08091_L.png'),\n PosixPath('0016E5_08029_L.png'),\n PosixPath('0016E5_08109_L.png'),\n PosixPath('0016E5_08051_L.png'),\n PosixPath('0016E5_08075_L.png'),\n PosixPath('0016E5_08155_L.png'),\n PosixPath('0016E5_07959_L.png'),\n PosixPath('0016E5_08157_L.png'),\n PosixPath('0016E5_08077_L.png'),\n PosixPath('0016E5_07999_L.png'),\n PosixPath('0016E5_08053_L.png'),\n PosixPath('0016E5_08093_L.png'),\n PosixPath('0016E5_08115_L.png'),\n PosixPath('0016E5_08035_L.png'),\n PosixPath('0016E5_07963_L.png'),\n PosixPath('0016E5_08149_L.png'),\n PosixPath('0016E5_08011_L.png'),\n PosixPath('0016E5_08069_L.png'),\n PosixPath('0016E5_07987_L.png'),\n PosixPath('0016E5_08131_L.png'),\n PosixPath('0016E5_07985_L.png'),\n PosixPath('0016E5_08133_L.png'),\n PosixPath('0016E5_08013_L.png'),\n PosixPath('0016E5_07961_L.png'),\n PosixPath('0016E5_08037_L.png'),\n PosixPath('0016E5_08117_L.png'),\n PosixPath('0016E5_07965_L.png'),\n PosixPath('0016E5_08033_L.png'),\n PosixPath('0016E5_08113_L.png'),\n PosixPath('0016E5_08137_L.png'),\n PosixPath('0016E5_07981_L.png'),\n PosixPath('0016E5_08017_L.png'),\n PosixPath('0016E5_08015_L.png'),\n PosixPath('0016E5_08135_L.png'),\n PosixPath('0016E5_07983_L.png'),\n PosixPath('0016E5_08111_L.png'),\n PosixPath('0016E5_08049_L.png'),\n PosixPath('0016E5_08089_L.png'),\n PosixPath('0016E5_08031_L.png'),\n PosixPath('0016E5_07967_L.png')]"
  },
  "execution_count": 19,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=15}
ds.reader.list_files(['.txt'])
```

```{.json .output n=15}
[
 {
  "data": {
   "text/plain": "[PosixPath('label_colors.txt'), PosixPath('readme.txt')]"
  },
  "execution_count": 15,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=3}
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
```
