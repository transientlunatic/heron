import torch
from lal import antenna
from astropy import units as u


class WaveformModel:

    def _convert(self, args):
        if "mass_ratio" in args and "chirp_mass" in args:
            args["total_mass"] = (
                args["chirp_mass"]
                * (1 + args["mass_ratio"]) ** 1.2
                / args["mass_ratio"] ** 0.6
            )
        if "total_mass" in args and "mass_ratio" in args:
            args = self._convert_mass_ratio_total_mass(args)
        if "luminosity_distance" in args:
            args = self._convert_luminosity_distance(args)
        if "geocent_time" in args:
            args["gpstime"] = args.pop("geocent_time")

        return args

    def _convert_luminosity_distance(self, args):
        args["distance"] = args.pop("luminosity_distance")
        return args

    def _convert_mass_ratio_total_mass(self, args):

        args["m1"] = (args["total_mass"] / (1 + args["mass_ratio"])).to_value(
            u.kilogram
        )
        args["m2"] = (args["total_mass"] / (1 + 1 / args["mass_ratio"])).to_value(
            u.kilogram
        )
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
