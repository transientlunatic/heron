

class Model(object):
    """
    This is the factory class for statistical models used for waveform generation.

    A model class must expose the following methods:
    - `distribution` : produce a distribution of waveforms at a given point in the parameter space
    - `mean` : produce a mean waveform at a given point in the parameter space
    - `train` : provide an interface for training the model
    """

    pass
