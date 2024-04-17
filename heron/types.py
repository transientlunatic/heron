from itertools import cycle

# GWPy to help with timeseries
from gwpy.timeseries import TimeSeries

import numpy as array_library
import matplotlib.pyplot as plt

class PSD:
    def __init__(self, data, frequencies):
        self.data = data
        self.frequencies = frequencies
        self.df = frequencies[1] - frequencies[0]

class Waveform(TimeSeries):
    def __init__(self, covariance, *args, **kwargs):
        self.covariance = covariance
        super(Waveform).__init__(*args, **kwargs)
    pass
        
class WaveformDict:
    def __init__(self, parameters=None, **kwargs):
        self.waveforms = kwargs
        self._parameters = parameters

    def __getitem__(self, item):
        return self.waveforms[item]

    @property
    def parameters(self):
        return self._parameters
    
    @property
    def hrss(self):
        if "plus" in self.waveforms and "cross" in self.waveforms:
            return array_library.sqrt(self.waveforms["plus"]**2 + self.waveforms["cross"]**2)
        else:
            raise NotImplementedError

class WaveformManifold:
    """
    Store a manifold of different waveform points.
    """
    def __init__(self):
        self.locations = []
        self.data = []

    def add_waveform(self, waveforms: WaveformDict):
        self.locations.append(waveforms.parameters)
        self.data.append(waveforms)

    def array(self, component="plus", parameter="m1"):
        all_data = []
        for wn in range(len(self.locations)):
            data = array_library.array(list(zip(cycle([self.locations[wn][parameter]]),
                                                self.data[wn][component].times.value,
                                                self.data[wn][component].value)))
            all_data.append(data)
        return array_library.vstack(all_data)
        
    def plot(self, component="plus", parameter="m1"):
        f, ax = plt.subplots(1, 1)
        for wn in range(len(self.locations)):
            data = array_library.array(list(zip(cycle([self.locations[wn][parameter]]), self.data[wn][component].times.value)))
            plt.scatter(data[:,1], data[:,0], c=self.data[wn][component], marker='.')
        return f
