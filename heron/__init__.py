"""
HERON: A Gaussian Process framework for Python
----------------------------------------------
"""


__date__ = "2017-06-07"
__maintainer__ = "Daniel Williams <daniel.williams@ligo.org>"


from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
