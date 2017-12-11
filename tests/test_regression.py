import numpy as np
from heron import regression, data, priors
import copy
from george import kernels
import george
"""This test module aims to test the Gaussian process regressors
included with this package, which take underlying numerical relativity
data, and perform inferential regression on them in order to produce a
callable surrogate."""

# TESTS for SINGLE-TASK GAUSSIAN PROCESSES

import unittest

np.random.seed(90)

class TestSingleGP(unittest.TestCase):
    """
    Test a single-task Gaussian process instance.
    """

    def setUp(self):
        self.targets = np.linspace(0, 20, 200)
        self.labels = np.sin(self.targets) + np.random.rand(200)

        self.targets_2d = np.array([np.linspace(0, 20, 200), np.linspace(0,1, 200)])
        self.labels_2d_input_dull = np.sin(self.targets_2d[0,:])# + np.random.rand(200)
        
        self.dull_labels = np.sin(self.targets)
        pass

    def test_1D_noiseless(self):
        """Perform a test on the simplest case, where there is no noise, and
        where the targets are 1-dimensional."""

        data_object = data.Data(targets=self.targets, target_names=["x"],
                                labels=self.dull_labels, label_names = ["y"],
                                test_size = 0,
                                label_sigma = 0,
        )

        #sep = np.abs(data_object.get_starting() + 0.0001)
        sep = np.array([1])
        kernel = kernels.Matern52Kernel(sep**2, ndim = len(sep))
        GP = regression.SingleTaskGP(data_object, kernel = kernel, solver=george.BasicSolver)

        # Test that the underlying Gaussian process returns something
        # close to the input data for the locations of the training
        # targets
        np.testing.assert_allclose(GP.prediction(self.targets)[0],
                                   self.dull_labels,
                                   rtol=1e-4, atol=1e-5
        )

        # Check that the Gaussian process returns something moderately
        # sensible for any given input
        x = np.linspace(0,20,1000)
        np.testing.assert_allclose(GP.prediction(x)[0],
                                   np.sin(x),
                                   rtol=1e-4, atol=1e-4
        )

    def test_2D_noiseless(self):
        """Perform a test on a multidimensional input case with no noise."""

        data_object = data.Data(targets=self.targets_2d.T, target_names=["x1", "x2"],
                                labels=self.labels_2d_input_dull, label_names = ["y"],
                                test_size = 0,
                                label_sigma = 0,
        )
        sep = np.array([1,1])
        kernel = kernels.Matern52Kernel(sep**2, ndim = len(sep))
        GP = regression.SingleTaskGP(data_object, kernel = kernel, solver=george.BasicSolver)

        # Test that the underlying Gaussian process returns something
        # close to the input data for the locations of the training
        # targets
        np.testing.assert_allclose(GP.prediction(self.targets_2d.T)[0],
                                   self.labels_2d_input_dull,
                                   rtol=1e-4, atol=1e-5
        )
