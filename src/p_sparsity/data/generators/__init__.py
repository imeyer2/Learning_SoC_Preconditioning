"""
Problem generators.
"""

from .anisotropic import AnisotropicDiffusionGenerator
from .elasticity import ElasticityGenerator
from .helmholtz import HelmholtzGenerator

__all__ = [
    "AnisotropicDiffusionGenerator",
    "ElasticityGenerator",
    "HelmholtzGenerator",
]
