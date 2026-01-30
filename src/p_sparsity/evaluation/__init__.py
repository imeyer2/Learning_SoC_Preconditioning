"""Evaluation module for analyzing AMG solver performance."""

from .pcg_analysis import run_pcg_analysis, PCGResult, compare_pcg_performance
from .vcycle_analysis import run_vcycle_analysis, VCycleResult, compare_vcycle_performance
from .eigenvalue_analysis import run_eigenvalue_analysis, EigenvalueResult, compare_spectral_properties

__all__ = [
    'run_pcg_analysis',
    'PCGResult',
    'compare_pcg_performance',
    'run_vcycle_analysis',
    'VCycleResult',
    'compare_vcycle_performance',
    'run_eigenvalue_analysis',
    'EigenvalueResult',
    'compare_spectral_properties',
]
