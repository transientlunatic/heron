"""This module contains logic to allow priors to be defined in an efficient manner.

Initially we'll use just the bilby prior handling machinery, but in
the middle-term it would be good to make this more flexible so that
other approaches can also be incorporated more easily.

"""

import bilby

KNOWN_PRIORS = {
    "UniformInComponentsChirpMass": bilby.gw.prior.UniformInComponentsChirpMass,
    "Uniform": bilby.prior.Uniform,
    "PowerLaw": bilby.prior.PowerLaw,
    "Sine": bilby.prior.Sine,
    "Cosine": bilby.prior.Cosine,
    "UniformSourceFrame": bilby.gw.prior.UniformSourceFrame,
    "UniformInComponentsMassRatio": bilby.gw.prior.UniformInComponentsMassRatio,
}


class PriorDict(bilby.core.prior.PriorDict):

    def from_dictionary(self, dictionary):

        for key, value in dictionary.items():
            if value["function"] in KNOWN_PRIORS:
                kind = value.pop("function")
                print(kind, value)
                dictionary[key] = KNOWN_PRIORS[kind](**value)

        super().from_dictionary(dictionary)

    @property
    def names(self):
        return list(self.keys())
