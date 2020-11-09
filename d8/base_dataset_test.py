import unittest
import pandas as pd

from .base_dataset import BaseDataset, ClassificationDataset

class TestBaseDataset(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'filepath':[1,2,3,4,5,6]})
        self.ds = BaseDataset(self.df)

    def test_split(self):
        a, b = self.ds.split(0.5)
        self.assertEqual(len(a), 3)
        self.assertEqual(len(b), 3)
        self.assertEqual(a.df['filepath'].tolist(), [6, 3, 2])

        c, d = self.ds.split(0.5)
        self.assertTrue(c.df.equals(a.df))
        self.assertTrue(b.df.equals(d.df))

        rets = self.ds.split([0.2, 0.3, 0.4])
        self.assertEqual(len(rets), 4)
        self.assertEqual(len(rets[0]), 1)
        self.assertEqual(len(rets[1]), 2)
        self.assertEqual(len(rets[2]), 2)
        self.assertEqual(len(rets[3]), 1)

    def test_merge(self):
        rets = self.ds.split([0.3, 0.4], shuffle=False)
        ds = rets[0].merge(*rets[1:])
        self.assertTrue(ds.df['filepath'].equals(self.ds.df['filepath']))


    def test_add(self):
        @BaseDataset.add
        def test():
            return BaseDataset(self.df)

        BaseDataset.add('test2', BaseDataset, [self.df])

    def test_get(self):
        self.assertTrue(BaseDataset.get('test').df.equals(self.df))
        self.assertTrue(BaseDataset.get('test2').df.equals(self.df))

    def test_list(self):
        self.assertEqual(BaseDataset.list(), ['test', 'test2'])


    def test_from_df_func(self):
        ds = BaseDataset.from_df_func(None, lambda reader: self.df)
        self.assertTrue(ds.df.equals(self.df))

class TestClassificationDataset(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'classname':[1,2,3,1,2,3]})
        self.ds = ClassificationDataset(self.df)

    def test_split(self):
        a, b = self.ds.split(0.8)
        self.assertEqual(b.df['classname'].tolist(), [1,2])
        self.assertEqual(a.classes, [1,2,3])
        self.assertEqual(b.classes, [1,2,3])

if __name__ == '__main__':
    unittest.main()
