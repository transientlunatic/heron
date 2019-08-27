"""
Tests for the base heron model classes.

"""

import unittest

from heron import models

import numpy as np
import numpy.testing as npt
np.random.seed(90)

class TestModel(unittest.TestCase):
    """
    Test the base model class which the functional models are constructed from.
    """

    def setUp(self):
        self.model = models.Model()


    def test_process_inputs(self):
        """
        Test that the base model doesn't perform any processing on the inputs.
        """
        TIMES = np.linspace(0, 10, 100)
        P = {"mass ratio": 0.9}

        times_p, p_p = self.model._process_inputs(TIMES.copy(), P.copy())

        npt.assert_array_equal(TIMES, times_p)
        self.assertEqual(P, p_p)

    
    def test_generate_eval_matrix(self):
        """
        Check that the default model class correctly assembles an
        evaluation vector.
        """
        times = np.linspace(0, 1., 11)

        
        p = {"second quantity": 0.9}

        POINTS = np.array([[0.0, 0.9], [0.1, 0.9], [0.2, 0.9], [0.3, 0.9],
                           [0.4, 0.9], [0.5, 0.9], [0.6, 0.9], [0.7, 0.9],
                           [0.8, 0.9], [0.9, 0.9], [1.0, 0.9]])

        points_p = self.model._generate_eval_matrix(p=p.copy(),
                                                    times=times.copy())

        npt.assert_array_almost_equal(POINTS, points_p)
    
