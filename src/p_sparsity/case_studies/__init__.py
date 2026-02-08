"""
Case studies module for controlled experiments.

Provides infrastructure for running reproducible case study experiments
with proper train/test parameter separation.
"""

from .config import (
    CaseStudyConfig,
    VariationConfig,
    TrainConfig,
    TestConfig,
    ProblemConfig,
    ParameterRange,
    ProblemType,
    create_anisotropic_case_study,
)

from .param_registry import ParameterRegistry, ParameterSet

from .generators import (
    UnifiedProblemGenerator,
    ProblemInstance,
    AnisotropicGenerator,
    ElasticityGenerator,
    HelmholtzGenerator,
    PoissonGenerator,
    create_generator,
)

from .metrics import (
    MetricsCollector,
    ProblemMetrics,
    SolverMetrics,
    PCGMetrics,
    VCycleMetrics,
    SpectralMetrics,
    ScalingMetrics,
    save_metrics,
    load_metrics,
)

from .runner import CaseStudyRunner, VariationResults, run_case_study

from .visualization import (
    plot_scaling_curves,
    plot_energy_decay,
    plot_iteration_histogram,
    plot_residual_curves,
    generate_comparison_table,
    generate_all_plots,
    plot_from_results_json,
)

from .eda import run_full_eda, EDAConfig

__all__ = [
    # Config
    'CaseStudyConfig',
    'VariationConfig', 
    'TrainConfig',
    'TestConfig',
    'ProblemConfig',
    'ParameterRange',
    'ProblemType',
    'create_anisotropic_case_study',
    
    # Registry
    'ParameterRegistry',
    'ParameterSet',
    
    # Generators
    'UnifiedProblemGenerator',
    'ProblemInstance',
    'AnisotropicGenerator',
    'ElasticityGenerator',
    'HelmholtzGenerator',
    'PoissonGenerator',
    'create_generator',
    
    # Metrics
    'MetricsCollector',
    'ProblemMetrics',
    'SolverMetrics',
    'PCGMetrics',
    'VCycleMetrics',
    'SpectralMetrics',
    'ScalingMetrics',
    'save_metrics',
    'load_metrics',
    
    # Runner
    'CaseStudyRunner',
    'VariationResults',
    'run_case_study',
    
    # Visualization
    'plot_scaling_curves',
    'plot_energy_decay',
    'plot_iteration_histogram',
    'plot_residual_curves',
    'generate_comparison_table',
    'generate_all_plots',
    'plot_from_results_json',
    
    # EDA
    'run_full_eda',
    'EDAConfig',
]
