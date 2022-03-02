import unittest

from heron.utils import Complex
from torch import tensor
import torch

import numpy as np
import numpy.testing as npt
np.random.seed(90)

class testComplex(unittest.TestCase):
    """Perform tests on the complex number class."""

    def setUp(self):
        self.ones = Complex(tensor([[1., 1.]]))
        self.only_real = Complex(tensor([[1., 0.]]))
        self.only_imaginary = Complex(tensor([[0., 1.]]))

        self.cases = [self.ones,
                      self.only_real,
                      self.only_imaginary]

    def test_real(self):
        """Returns correct real part"""
        answers = [1, 1, 0]
        for i, case in enumerate(self.cases):
            with self.subTest(i=i):
                npt.assert_equal(case.real.numpy(), answers[i])

    def test_imag(self):
        """Returns correct imaginary part"""
        answers = [1, 0, 1]
        for i, case in enumerate(self.cases):
            with self.subTest(i=i):
                npt.assert_equal(case.imag.numpy(), answers[i])
        
    def test_modulus_squared(self):
        """Ensure the squared modulus is correctly calculated."""
        answers = [2, 1, 1]
        for i, case in enumerate(self.cases):
            with self.subTest(i=i):
                npt.assert_almost_equal(case.r2, answers[i])

    def test_modulus(self):
        answers = [np.sqrt(2), 1, 1]
        for i, case in enumerate(self.cases):
            with self.subTest(i=i):
                npt.assert_almost_equal(case.modulus, answers[i])

    def test_reciprocal(self):
        """Ensure that the inversion of some numbers."""
        answers = [tensor([[0.5, -0.5]]),
                   tensor([[1.0, 0.0]]),
                   tensor([[0.0, -1.0]])]
        
        for i, case in enumerate(self.cases):
            with self.subTest(i=i):
                npt.assert_almost_equal(case.reciprocal.tensor.numpy(), answers[i].numpy())

    def test_inversion_product(self):
        """Ensure that the product of an inverted number and itself is 1."""
        for i, case in enumerate(self.cases):
            with self.subTest(i=i):
                result = case * case.reciprocal
                npt.assert_almost_equal(result.tensor.numpy(), self.only_real.tensor.numpy())

    def test_product(self):
        """Test product of complex numbers."""
        for i, case in enumerate(self.cases):
            with self.subTest(i=i):
                npt.assert_almost_equal((case*self.cases[1]).tensor.numpy(), case.tensor.numpy())
