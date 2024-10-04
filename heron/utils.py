"""
Various utilities and non-GPR based stuff.
"""

import torch
import yaml

import numpy as np

# from pesummary.io import read
# from pesummary.core.file.formats.base_read import SingleAnalysisRead


def load_yaml(filename):
    with open(filename, "r") as f:
        data = yaml.safe_load(f)

    return data


def diag_cuda(a):
    """Make a vector into a diagonal matrix."""
    b = torch.diag(a)
    return b
    # a = torch.view_as_real(a)
    # if a.dim() == 2:
    #     b = torch.stack([torch.diag(a[:, 0]), torch.diag(a[:, 1])], dim=-1)
    # elif a.dim() == 3:
    #     b = torch.stack([torch.diag(a[:, :, 0]), torch.diag(a[:, :, 1])], dim=-1)
    # return torch.view_as_complex(b)


class Complex:
    """
    Complex numbers in torch and CUDA.
    """

    def __init__(self, tensor):
        """
        Construct a complex number, or an tensor of
        complex numbers, from a tensor with the first
        column representing the real parts, and the second
        the imagonary parts.

        Parameters
        ----------
        tensor : ``torch.tensor``
           The input tensor.
           The first column should represent the real part of the number(s),
           the second column the imaginary parts.
        """
        self.tensor = tensor

    @property
    def conjugate(self):
        """Return the complex conjugate of this number."""
        new_tensor = self.tensor.clone().T
        new_tensor[1] = new_tensor[1] * -1
        return Complex(new_tensor.T)

    @property
    def real(self):
        """Return the real part of the number."""
        return self.tensor.clone().T[0]  # [:, 0]

    @property
    def imag(self):
        """Return the imaginary part of the number."""
        return self.tensor.clone().T[1]

    def __getitem__(self, slice_expression):
        return Complex(self.tensor[slice_expression])

    def __add__(self, other):
        if isinstance(other, Complex):
            return Complex(self.tensor + other.tensor)
        elif isinstance(other, type(self.tensor)):
            # Assume other is a real tensor
            t = self.tensor.clone()
            t[:, 0] += other
            return Complex(t)

    def __sub__(self, other):
        return self.tensor - other.tensor

    def __div__(self, other):
        return self.division(other)

    def __truediv__(self, other):
        return self.division(other)

    def __mul__(self, other):
        if isinstance(other, Complex):
            return self.product(other)
        elif isinstance(other, type(self.tensor)):
            return Complex(self.tensor * other)
        elif isinstance(other, float):
            return Complex(other * self.tensor)
        elif isinstance(other, complex):
            t = self.tensor.clone()
            t[:, 0] = self.tensor[:, 0] * other.real - self.tensor[:, 1] * other.imag
            t[:, 1] = self.tensor[:, 1] * other.real + self.tensor[:, 0] * other.imag
            return Complex(t)
        else:
            raise NotImplementedError(f"Cannot __mul__({type(self)},{type(other)})")

    def __repr__(self):
        return self.tensor.__repr__()

    def product(self, b):
        """
        Calculate the product of this complex number with another.

        Parameters
        ----------
        b : ``Complex``
           The other complex number.

        Returns
        -------
        product : ``Complex``
           The product of this number and **b**.
        """

        return Complex(
            torch.stack(
                [
                    (self.real * b.real) - (self.imag * b.imag),
                    (self.real * b.imag) + (self.imag * b.real),
                ]
            ).T
        )

    def clone(self):
        return Complex(self.tensor.clone())

    @property
    def r2(self):
        """
        Calculate the polar radius squared of the number
        """
        temp = self.tensor.clone().T
        return temp[0] ** 2 + temp[1] ** 2

    @property
    def modulus(self):
        """Return the modulus of this number."""
        return torch.sqrt(self.r2)

    @property
    def reciprocal(self):
        """
        Calculate the reciprocal of this number.
        """
        return Complex(
            torch.stack(
                [torch.div(self.real, self.r2), -torch.div(self.imag, self.r2)]
            ).T
        )

    def division(self, b):
        """
        Calculate the division of this number with another
        complex number.
        """
        return self * b.reciprocal


def noise_psd(N, frequencies, psd=lambda f: 1):
    """
    Generate noise with a given PSD
    """
    reals = np.random.randn(len(frequencies))
    imags = np.random.randn(len(frequencies))

    T = 1 / (frequencies[1] - frequencies[0])
    if callable(psd):
        psd = np.array([psd(float(f)) for f in frequencies])
    S = np.sqrt(N * N / 4 / (T) * psd)

    noise_r = S * (reals)
    noise_i = S * (imags)

    noise_f = noise_r + 1j * noise_i

    return np.fft.irfft(noise_f, n=(N))


# def make_metafile(datafile, outfile="metafile.dat"):
#     class CustomReadClass(SingleAnalysisRead):
#         """Class to read in our custom file

#         Parameters
#         ----------
#         path_to_results_file: str
#             path to the result file you wish to read in
#         """

#         def __init__(self, path_to_results_file, **kwargs):
#             super(CustomReadClass, self).__init__(path_to_results_file, **kwargs)
#             self.load(self.custom_load_function)

#         def custom_load_function(self, path, **kwargs):
#             """Function to load data from a custom hdf5 file"""
#             import h5py

#             with h5py.File(path, "r") as f:
#                 parameters = list(f["posterior_samples"].dtype.names)
#                 pars = [parameter.replace(" ", "_") for parameter in parameters]
#                 samples = np.array(
#                     [f["posterior_samples"][param] for param in parameters]
#                 ).T
#             # Return a dictionary of data
#             return {"parameters": pars, "samples": samples}

#     data = read(datafile, package="gw", cls=CustomReadClass)
#     data.to_dat(filename=outfile)
