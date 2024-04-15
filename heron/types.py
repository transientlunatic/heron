# GWPy to help with timeseries
from gwpy.timeseries import TimeSeries

import numpy as array_library


class PSD:
    def __init__(self, data, frequencies):
        self.data = data
        self.frequencies = frequencies
        self.df = frequencies[1] - frequencies[0]

class Waveform(TimeSeries):
    pass
        
class WaveformDict:
    def __init__(self, **kwargs):
        self.waveforms = kwargs

    def __getitem__(self, item):
        return self.waveforms[item]

    @property
    def hrss(self):
        if "plus" in self.waveforms and "cross" in self.waveforms:
            return array_library.sqrt(self.waveforms["plus"]**2 + self.waveforms["cross"]**2)
        else:
            raise NotImplementedError
