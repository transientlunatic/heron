import unittest

import astropy.units as u

from heron.models import lalsimulation
from heron.training import makedata

class Test_Manifold_Generation(unittest.TestCase):

    def setUp(self):
        self.manifold = makedata.make_manifold(parameter="m1", lower=10, upper=100, step=2, unit=u.solMass)

    def test_waveform_parameters(self):
        pass

    def test_manifold_plot(self):
        f = self.manifold.plot()
        f.savefig("manifold.png")
        
