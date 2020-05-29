"""
Tests for the science testing code.
"""

import unittest

from heron import testing
from elk.catalogue import NRCatalogue

from heron.models.georgebased import HeronHodlr

class TestTests(unittest.TestCase):
    """
    Test the science testing code.
    """

    def setUp(self):
        self.model = HeronHodlr()
        self.samples_catalogue = NRCatalogue("GeorgiaTech")
        self.samples_catalogue.waveforms = self.samples_catalogue.waveforms[:2]

    def test_nrcat_match(self):
        """Test the NR catalogue matcher."""
        matches = testing.nrcat_match(self.model, self.samples_catalogue)

        self.assertEqual(len(matches.values()), 2)
