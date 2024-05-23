"""This module contains logic to allow priors to be defined in an efficient manner.

Initially we'll use just the bilby prior handling machinery, but in
the middle-term it would be good to make this more flexible so that
other approaches can also be incorporated more easily.

"""

"""
   gpstime: Uniform(name="gpstime", minimum=3999.9, maximum=4000.1, boundary=None)

   total_mass: Uniform(name='total_mass', minimum=40, maximum=80)
   chirp_mass: Uniform(name='chirp_mass', minimum=1, maximum=1000)
   mass_ratio: bilby.gw.prior.UniformInComponentsMassRatio(name='mass_ratio', minimum=0.4, maximum=1.0)
   luminosity_distance: PowerLaw(name='luminosity_distance', alpha=2, maximum=4000, minimum=10,  unit='Mpc')
   theta_jn: Sine(name='theta_jn')
   azimuth: Uniform(name='azimuth', minimum=0, maximum=2 * np.pi, boundary='periodic')
   zenith: Sine(name='zenith')
   psi: Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic')
   phase: Uniform(name='phase', minimum=0, maximum=np.pi, boundary='periodic')
"""

import bilby

KNOWN_PRIORS = {
    "UniformSourceFrame": bilby.gw.prior.UniformSourceFrame,
    "UniformInComponentsMassRatio": bilby.gw.prior.UniformInComponentsMassRatio,
    "Uniform": bilby.prior.Uniform,
    "PowerLaw": bilby.prior.PowerLaw,
    "Sine": bilby.prior.Sine,
    "Cosine": bilby.prior.Cosine,
}


class PriorDict(bilby.core.prior.PriorDict):

    def from_dictionary(self, dictionary):

        for key, value in dictionary.items():
            if value["function"] in KNOWN_PRIORS:
                kind = value.pop("function")
                dictionary[key] = KNOWN_PRIORS[kind](**value)

        super().from_dictionary(dictionary)

    @property
    def names(self):
        return list(self.keys())
