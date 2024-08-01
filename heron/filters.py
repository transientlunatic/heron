import numpy as np


class Filter:
    """The factory class for all filters within heron."""

    def array(self, array):
        return np.array(array)

    def solve(self, matrix_a, matrix_b):
        return np.linalg.solve(matrix_a, matrix_b)

    def sqrt(self, a):
        return np.sqrt(a)

    def abs(self, a):
        return np.abs(a)


class InnerProduct(Filter):

    def __init__(self, psd=None):
        self.psd = psd

    def __call__(self, waveform_a, waveform_b):
        """
        Calculate the inner product between two waveforms either with or without noise.
        """
        dt = waveform_a.dt
        assert self.abs(waveform_a.dt - waveform_b.dt).value < 1e-4
        N = len(waveform_a.times)

        if self.psd:
            C = self.psd.covariance_matrix(times=waveform_a.times)
            h_h = (
                (
                    self.array(waveform_a.data).T
                    @ self.solve(C, np.array(waveform_b.data))
                )
                * (dt * dt / N / 4)
                / 4
            )
        else:
            h_h = (
                (self.array(waveform_a.data).T @ self.array(waveform_b.data))
                * (dt * dt / N / 4)
                / 4
            )

        return self.sqrt(self.abs(h_h))


class Overlap(Filter):

    def __init__(self, psd=None):
        self.psd = psd
        self.ip = InnerProduct(psd)

    def __call__(self, waveform_a, waveform_b):
        """
        Calculate the overlap of two waveforms, either with or without noise.
        """
        nominator = self.ip(waveform_a, waveform_b)
        den_a = self.ip(waveform_a, waveform_a)
        den_b = self.ip(waveform_b, waveform_b)

        return nominator / self.sqrt(den_a * den_b)


class Match(Filter):

    def __init__(self, psd=None):
        self.psd = psd
        self.ip = InnerProduct(psd)

    def __call__(self, waveform_a, waveform_b, psd=None):
        """
        Calculate the match of two waveforms, either with or without noise.
        """

        waveform_a_f = waveform_a.frequency_domain()
        waveform_b_f = waveform_b.frequency_domain()
