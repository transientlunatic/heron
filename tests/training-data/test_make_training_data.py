import unittest

import astropy.units as u

from heron.models import lalsimulation
from heron.training import makedata

class Test_Manifold_Generation(unittest.TestCase):

    def setUp(self):
        self.manifold = makedata.make_manifold(varied={"mass ratio": dict(lower=0.1, upper=1, step=0.05)},
                                               fixed={"total mass": 60*u.solMass,
                                                      "delta T": 1/(1024*u.Hertz)})

    def test_waveform_parameters(self):
        pass

    def test_manifold_plot(self):
        f = self.manifold.plot()
        f.axes[0].set_xlim([-0.4, 0.02])
        f.savefig("manifold.png")
        
    def test_manifold_array(self):
        data = self.manifold.array()
        print(data.shape)

class Test_Optimal_Manifold_Generation(unittest.TestCase):

    def setUp(self):
        self.manifold = makedata.make_optimal_manifold(varied={"mass ratio": dict(lower=0.1, upper=1, step=0.05)},
                                               fixed={"total mass": 60*u.solMass,
                                                      "delta T": 1/(1024*u.Hertz)})

    def test_waveform_parameters(self):
        pass

    def test_manifold_plot(self):
        f = self.manifold.plot()
        f.axes[0].set_xlim([-0.4, 0.02])
        f.savefig("manifold.png")
        
    def test_manifold_array(self):
        data = self.manifold.array()
        print(data.shape)
