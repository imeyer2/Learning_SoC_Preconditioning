"""Visualization module for AMG analysis and results."""

from .sparsity import (
    plot_sparsity_pattern,
    plot_C_comparison,
    plot_sparse_density_heatmap,
    plot_C_overlay_heatmap,
    plot_P_comparison,
    compute_C_overlap_stats,
    plot_iteration_bar,
    plot_weight_vs_logit,
)
from .convergence import (
    plot_convergence_curves,
    plot_reduction_factors,
    plot_vcycle_comparison,
    plot_eigenvalue_spectrum,
    plot_condition_number_comparison,
    plot_eigenvalue_spectra_comparison,
    plot_pcg_convergence_comparison,
)
from .training_curves import plot_training_progress, plot_reward_distribution
from .comparison import plot_performance_comparison, create_comparison_table, plot_suite_summary

__all__ = [
    # Sparsity
    'plot_sparsity_pattern',
    'plot_C_comparison',
    'plot_sparse_density_heatmap',
    'plot_C_overlay_heatmap',
    'plot_P_comparison',
    'compute_C_overlap_stats',
    'plot_iteration_bar',
    'plot_weight_vs_logit',
    # Convergence
    'plot_convergence_curves',
    'plot_reduction_factors',
    'plot_vcycle_comparison',
    'plot_eigenvalue_spectrum',
    'plot_condition_number_comparison',
    'plot_eigenvalue_spectra_comparison',
    'plot_pcg_convergence_comparison',
    # Training
    'plot_training_progress',
    'plot_reward_distribution',
    # Comparison
    'plot_performance_comparison',
    'create_comparison_table',
    'plot_suite_summary',
]
