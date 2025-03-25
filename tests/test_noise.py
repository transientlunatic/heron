"""
Tests for noise and PSD-related functions.
"""

import unittest

import heron.models.testing

class TestFlatPSD(unittest.TestCase):

    def setUp(self):
        self.psd = heron.models.testing.FlatPSD()

    def test_plot_fd_psd(self):

        f = self.psd.frequency_domain().plot()
        f.savefig("test_psd.png")
