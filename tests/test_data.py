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

class TestData(unittest.TestCase):

    def setUp(self):
        # Create some random data to use to test the
        # data object.
        self.test_targets = 100 * np.random.rand(100,5)
        self.test_labels = 50 * np.random.rand(100, 2)
        
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

    def test_denormalisation(self):
        # Check that data is correctly denormalised.
        test_object = data.Data(self.test_targets, self.test_labels)
        targets = test_object.denormalise(test_object.targets, test_object.targets_scale)
        labels = test_object.denormalise(test_object.labels, test_object.labels_scale)
        self.assertEqual(targets.max(), self.test_targets.max())
        self.assertEqual(labels.max(), self.test_labels.max())
        self.assertEqual(targets.min(), self.test_targets.min())
        self.assertEqual(labels.min(), self.test_labels.min())

    def tearDown(self):
        pass

    def test_000_something(self):
        pass


if __name__ == '__main__':
    import sys
    sys.exit(unittest.main())
