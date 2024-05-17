import torch
from lal import antenna
from astropy import units as u

class WaveformModel:

    def _convert(self, args):
        
        if "total_mass" in args and "mass_ratio" in args:
            args = self._convert_mass_ratio_total_mass(args)
        if "luminosity_distance" in args:
            args = self._convert_luminosity_distance(args)

        return args

    def _convert_luminosity_distance(self, args):
            args["distance"] = args.pop("luminosity_distance")
            return args
            
    def _convert_mass_ratio_total_mass(self, args):
        
            args['m1'] = (args["total_mass"] / (1 + args["mass_ratio"])).to_value(u.kilogram)
            args['m2'] = (args["total_mass"] / (1 + 1 / args["mass_ratio"])).to_value(u.kilogram)
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
