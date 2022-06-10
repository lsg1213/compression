import torch
import torch.nn as nn
import unittest
from utils import *


class UtilsTest(unittest.TestCase):
    def test_grid_sample(self):
        tensor = torch.tensor([[[[1., 0, 2, 3]]]])
        grid = torch.tensor([[[[-1.0000,  0.0000],
                               [-0.7500,  0.0000],
                               [-0.5000,  0.0000],
                               [-0.2500,  0.0000],
                               [ 0.0000,  0.0000],
                               [ 0.2500,  0.0000],
                               [ 0.5000,  0.0000],
                               [ 0.7500,  0.0000],
                               [ 1.0000,  0.0000]]]]) 
        target = torch.tensor([[[[1.5, 1, 0.5, 0, 1, 2, 2.5, 3, 3.5]]]])
        self.assertEqual(grid_sample(tensor, grid).numpy().tolist(),
                         target.numpy().tolist())


if __name__ == '__main__':
    unittest.main()

