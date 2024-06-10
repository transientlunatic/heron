import logging

import torch
from lal import antenna
from astropy import units as u

logger = logging.getLogger("heron.models")

class WaveformModel:

    def _convert(self, args):
        if "total_mass" in args and "mass_ratio" in args:
            args = self._convert_mass_ratio_total_mass(args)
        elif "total_mass" in args or "mass_ratio" in args:
            print("Need both total mass and the mass ratio")
        if "luminosity_distance" in args:
            args = self._convert_luminosity_distance(args)

        return args

    def _convert_luminosity_distance(self, args):
        args["distance"] = args.pop("luminosity_distance")
        return args

    def _convert_mass_ratio_total_mass(self, args):
        logger.info("Converting total mass and mass ratio to components")
        m1 = args["m1"] = (args["total_mass"] / (1 + args["mass_ratio"]))
        if isinstance(args["m1"], u.Quantity):
            args["m1"] = args["m1"].to(u.kilogram)
        else:
            args["m1"] = (args["m1"] * u.solMass).to(u.kilogram)
        args["m2"] = (args["total_mass"] / (1 + 1 / args["mass_ratio"]))
        if isinstance(args["m2"], u.Quantity):
            args["m2"] = args["m2"].to(u.kilogram)
        else:
            args["m2"] = (args["m2"] * u.solMass).to(u.kilogram)
        self.logger.info(f"Converted {args}")
        args.pop("total_mass")
        args.pop("mass_ratio")

        return args


class WaveformApproximant(WaveformModel):
    """
    This class handles a waveform approximant model.
    """

    pass


class WaveformSurrogate(WaveformModel):
    """
    This class handles a waveform surrogate model.
    """

    pass


class PSDModel:
    pass


class PSDApproximant(PSDModel):
    pass
