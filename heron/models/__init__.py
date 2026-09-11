import numpy as np
from astropy import units as u

# Solar mass in kg — avoids lal dependency for a physical constant
MSUN_SI = 1.98892e30


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
        args["m1"] = (args["total_mass"] / (1 + args["mass_ratio"]))
        args["m2"] = (args["total_mass"] / (1 + (1 / args["mass_ratio"])))
        if isinstance(args["m1"], u.Quantity):
            args["m1"] = float(args["m1"].to_value(u.kilogram))
            args["m2"] = float(args["m2"].to_value(u.kilogram))
        elif args["m1"] < 1000:
            # Heuristic: values < 1000 are likely solar masses
            args["m1"] *= MSUN_SI
            args["m2"] *= MSUN_SI

        args.pop("total_mass")
        args.pop("mass_ratio")
        return args


class WaveformApproximant(WaveformModel):
    """Base class for analytical/numerical waveform approximants."""
    pass


class WaveformSurrogate(WaveformModel):
    """Base class for waveform surrogate models."""
    pass
