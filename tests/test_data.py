#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_data
----------------------------------

Tests for the `data` module.
"""

import unittest

from heron import data
import numpy as np

np.random.seed(90)

class TestData(unittest.TestCase):

    def setUp(self):
        # Create some random data to use to test the
        # data object.
        self.test_targets = 100 * np.random.rand(100,5)
        self.test_labels = 50 * np.random.rand(100, 2)

        #self.test_targets_flat = np.copy(self.test_targets)
        self.test_targets_flat = 100* np.ones(100)
        self.test_labels_flat = 1 * np.ones( 100 )

        self.test_targets_1d = 100 * np.random.rand(100)
        self.test_labels_1d = 50 * np.random.rand(100)
        
        pass

    def test_normalisation(self):
        # Check that the object correctly normalises the 
        # input data.
        DESIRED_MAX = 1
        DESIRED_MIN = 0

        test_object = data.Data(self.test_targets, self.test_labels)

        self.assertEqual(DESIRED_MAX, test_object.targets.max())
        self.assertEqual(DESIRED_MAX, test_object.labels.max())
        self.assertEqual(DESIRED_MIN, test_object.targets.min())
        self.assertEqual(DESIRED_MIN, test_object.labels.min())

    def test_normalisation_1d(self):
        # Check that the object correctly normalises the 
        # input data if the input data is one-dimensional
        DESIRED_MAX = 1
        DESIRED_MIN = 0
        
        test_object = data.Data(self.test_targets_1d, self.test_labels_1d)

        self.assertAlmostEqual(DESIRED_MAX, test_object.targets.max())
        self.assertAlmostEqual(DESIRED_MAX, test_object.labels.max())
        self.assertAlmostEqual(DESIRED_MIN, test_object.targets.min())
        self.assertAlmostEqual(DESIRED_MIN, test_object.labels.min())
        
    def test_normalisation_norange(self):
        # Check that the object correctly normalises the input data,
        # when that data has no dynamic range.  This means that all of
        # the values in the dimension should be 0
        DESIRED_MAX = 0
        DESIRED_MIN = 0

        
        test_object = data.Data(self.test_targets_flat, self.test_labels_flat, test_size = 0)

        self.assertEqual(DESIRED_MAX, test_object.targets.max())
        self.assertEqual(DESIRED_MAX, test_object.labels.max())
        self.assertEqual(DESIRED_MIN, test_object.targets.min())
        self.assertEqual(DESIRED_MIN, test_object.labels.min())

    def test_denormalisation(self):
        # Check that data is correctly denormalised.
        test_object = data.Data(self.test_targets, self.test_labels)
        targets = test_object.denormalise(test_object.targets, "target")
        labels = test_object.denormalise(test_object.labels, "label")
        self.assertAlmostEqual(targets.max(), self.test_targets.max())
        self.assertAlmostEqual(labels.max(), self.test_labels.max())
        self.assertAlmostEqual(targets.min(), self.test_targets.min())
        self.assertAlmostEqual(labels.min(), self.test_labels.min())

    def test_data_addition(self):
        # Check that new data is handled correctly when it's added to
        # the object
        test_object =  data.Data(self.test_targets, self.test_labels, test_size=0)

        new_targets = np.random.rand(20,5)
        new_labels  = np.random.rand(20,2)

        test_object.add_data(new_targets, new_labels)

        DESIRED_MAX = 1
        DESIRED_MIN = 0

        # check the normalisation is correctly handled
        #self.assertEqual(DESIRED_MAX, test_object.targets.max())
        #self.assertEqual(DESIRED_MAX, test_object.labels.max())
        #self.assertEqual(DESIRED_MIN, test_object.targets.min())
        #self.assertEqual(DESIRED_MIN, test_object.labels.min())

        LEN = len(new_targets) + len(self.test_targets)

        self.assertEqual(LEN, len(test_object.targets))


        
    def tearDown(self):
        pass



if __name__ == '__main__':
    import sys
    sys.exit(unittest.main())
