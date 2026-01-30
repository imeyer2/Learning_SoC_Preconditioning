"""Visualization module for AMG analysis and results."""

from .sparsity import plot_sparsity_pattern, plot_C_comparison
from .convergence import (
    plot_convergence_curves,
    plot_reduction_factors,
    plot_vcycle_comparison,
    plot_eigenvalue_spectrum,
    plot_condition_number_comparison,
)
from .training_curves import plot_training_progress, plot_reward_distribution
from .comparison import plot_performance_comparison, create_comparison_table

__all__ = [
    'plot_sparsity_pattern',
    'plot_C_comparison',
    'plot_convergence_curves',
    'plot_reduction_factors',
    'plot_vcycle_comparison',
    'plot_eigenvalue_spectrum',
    'plot_condition_number_comparison',
    'plot_training_progress',
    'plot_reward_distribution',
    'plot_performance_comparison',
    'create_comparison_table',
]
