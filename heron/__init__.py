"""
HERON: A Gaussian Process framework for Python
----------------------------------------------
"""

__date__ = "2017-06-07"
__maintainer__ = "Daniel Williams <daniel.williams@ligo.org>"

import logging
from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    __version__ = "dev"
    pass

logger = logging.getLogger("heron")

logger_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "update": 9,
}
# try:
#     LOGGER_LEVEL = logger_levels[config.get("logging", "logging level")]
# except configparser.NoOptionError:
LOGGER_LEVEL = logging.DEBUG

# try:
#     PRINT_LEVEL = logger_levels[config.get("logging", "print level")]
# except configparser.NoOptionError:
PRINT_LEVEL = logging.WARNING

ch = logging.StreamHandler()
print_formatter = logging.Formatter("[%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
ch.setFormatter(print_formatter)
ch.setLevel(PRINT_LEVEL)

logfile = f"{__name__}.log"
fh = logging.FileHandler(logfile)
formatter = logging.Formatter(
    "%(asctime)s [%(name)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
)
fh.setFormatter(formatter)
fh.setLevel(LOGGER_LEVEL)

logger.addHandler(ch)
logger.addHandler(fh)

logger.info(f"Running {__name__} {__version__}")
