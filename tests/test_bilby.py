"""
Tests for the bilby interfaces
"""

import unittest
import os
import heron.bilby

class TestBilbyData(unittest.TestCase):

    def setUp(self):
        self.current_directory = os.path.dirname(os.path.realpath(__file__))
        self.data_directory = os.path.join(self.current_directory, "testing-data")

    def test_read_pickle(self):
        data = heron.bilby.read_pickle(
            os.path.join(self.data_directory, "bilby-pipe-data.pickle")
        )
        self.assertTrue(set(data['strain'].keys()) == {"H1", "L1"})
