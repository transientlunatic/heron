"""
Tests on the calculation of inner products in CUDA and torch.
"""

import unittest
from heron.likelihood import InnerProduct, device
import torch

class Test_InnerProduct(unittest.TestCase):
    TEST_DATA = torch.randn(100, device=device) * 1e-23
    TEST_TEMPLATE = torch.randn(100, device=device) * 1e-23

    TEST_PSD = torch.randn(100, device=device, dtype=torch.complex128) * 1e-46
    
    def test_simple_case_dimensionality(self):
        """Test the inner product of two vectors with no noise returns a scalar value."""
        ip = InnerProduct(duration=1)
        self.assertTrue(ip(self.TEST_DATA, self.TEST_TEMPLATE).ndim==0)

    def test_simple_case_value(self):
        """Test that the inner product of two vectors gives a plausible result."""
        data = torch.ones(50)*2
        ip = InnerProduct(duration=1)
        self.assertTrue(ip(data, data)==800)
        # Should equal 4 * normal dot product

    def test_simple_case_duration(self):
        """Test that the inner product accounts for the duration."""
        data = torch.ones(50)*2
        ip = InnerProduct(duration=4)
        self.assertTrue(ip(data, data)==200)

    def test_psd_case_dimensionality(self):
        """Test that the inner product of two vectors with data noise returns a scalar."""
        ip = InnerProduct(duration=1,
                          metric=self.TEST_PSD
                          )
        self.assertTrue(ip(self.TEST_DATA, self.TEST_TEMPLATE).ndim==0)
        

    def test_psd_case_value(self):
        """Test that the inner product of two vectors with data noise returns a scalar."""
        torch.manual_seed(0)
        data = torch.ones(100, device=device,
                          dtype=torch.complex128)*2
        psd = (1+torch.randn(100, device=device)*1e-5) / 50
        ip = InnerProduct(duration=1,
                          metric=psd
                          )
        self.assertEqual(float(ip(data, data).cpu()), 800.0008055386942)

    def test_covariance_case_dimensionality(self):
        """Test that the inner product weighted by a matrix gives a scalar."""

        metric = (1+torch.randn((100,100), device=device)*1e-5) / 50
        
        ip = InnerProduct(duration=1,
                          metric=metric
                          )
        self.assertTrue(ip(self.TEST_DATA, self.TEST_TEMPLATE).ndim==0)

    def test_psd_case_value(self):
        """Test that the inner product weighted by a matrix gives a sensible answer."""
        torch.manual_seed(0)
        data = torch.ones(100, device=device,
                          dtype=torch.complex128)*2
        metric = (1+torch.randn((100,100), device=device)*1e-5) / 50
        ip = InnerProduct(duration=1,
                          metric=metric
                          )
        self.assertTrue(abs(float(ip(data, data).cpu()) - 800.0008055386942) < 0.01)
