import unittest
import torch

class TestDiagonalCUDA(unittest.TestCase):
    device = torch.device("cuda")
    def test_vector_to_matrix(self):
        a = torch.ones(100, device=self.device)
        self.assertEqual(a.diag().device.type, "cuda")

    def test_vector_to_matrix(self):
        a = torch.ones((100, 100), device=self.device)
        self.assertEqual(a.diag().device.type, "cuda")
