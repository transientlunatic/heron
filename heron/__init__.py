"""
Heron: Probabilistic waveform emulation with uncertainty.
"""

__date__ = "2017-06-07"
__maintainer__ = "Daniel Williams <daniel.williams@ligo.org>"

import logging
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "dev"

logger = logging.getLogger("heron")
