"""Parameter-estimation (PE) layer for Heron.

A lightweight, modular inference stack built on the surrogate waveform models.
This subpackage depends only on the waveform side of Heron (``heron.models``,
``heron.types``) and the existing top-level PE primitives (``heron.likelihood``,
``heron.noise``, ``heron.detector``, ``heron.sampling``) — never the reverse —
so it can be extracted into a standalone package later.

Public API::

    from heron.inference import (
        Detector, KNOWN_DETECTORS, estimate_psd_welch,
        NetworkLikelihood, project_polarisations,
        Uniform, Sine, Cosine, PowerLaw, PriorDict,
        DynestySampler, NessaiSampler,
        Injection, credible_levels_from_grid, pp_plot,
    )
"""
from heron.sampling import DynestySampler, SamplerResult

from heron.inference.detectors import Detector, KNOWN_DETECTORS, estimate_psd_welch
from heron.inference.projection import project_polarisations, project_variances
from heron.inference.network import NetworkLikelihood
from heron.inference.prior import Uniform, Sine, Cosine, PowerLaw, Prior, PriorDict
from heron.inference.sampler import NessaiSampler
from heron.inference.injection import Injection, InjectionResult
from heron.inference.coverage import (
    credible_level_1d,
    credible_levels_from_grid,
    ks_uniform_pvalue,
    pp_plot,
)

__all__ = [
    "Detector",
    "KNOWN_DETECTORS",
    "estimate_psd_welch",
    "NetworkLikelihood",
    "project_polarisations",
    "project_variances",
    "Prior",
    "Uniform",
    "Sine",
    "Cosine",
    "PowerLaw",
    "PriorDict",
    "DynestySampler",
    "NessaiSampler",
    "SamplerResult",
    "Injection",
    "InjectionResult",
    "credible_level_1d",
    "credible_levels_from_grid",
    "ks_uniform_pvalue",
    "pp_plot",
]
