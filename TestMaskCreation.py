import unittest

from PennFudan_Dataset import MaskCreator


class MyTestCase(unittest.TestCase):

    def test_simple_mask_creation(self):
        mc = MaskCreator()
        array = mc.create_mask(1,2)
        self.assertIsNotNone(array)
        self.assertTrue(array.ndim == 2)

if __name__ == '__main__':
    unittest.main()
