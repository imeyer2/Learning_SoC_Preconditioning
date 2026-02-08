"""
Configuration schema for case studies.

Defines dataclasses for case study configurations that can be loaded from YAML
and used to orchestrate training/testing experiments with parameter tracking.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import yaml
from pathlib import Path


class ProblemType(Enum):
    """Supported problem types for case studies."""
    ANISOTROPIC = "anisotropic"
    ELASTICITY = "elasticity"
    HELMHOLTZ = "helmholtz"
    POISSON = "poisson"


@dataclass
class ParameterRange:
    """
    Defines a range for sampling problem parameters.
    
    Supports:
    - Continuous ranges: min/max with optional log scale
    - Discrete sets: explicit list of values
    - Fixed values: single value
    """
    name: str
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    values: Optional[List[float]] = None  # For discrete choices
    log_scale: bool = False  # Sample in log space
    fixed: Optional[float] = None  # Fixed value (not varied)
    
    def __post_init__(self):
        # Validate configuration
        has_range = self.min_val is not None and self.max_val is not None
        has_values = self.values is not None
        has_fixed = self.fixed is not None
        
        if sum([has_range, has_values, has_fixed]) != 1:
            raise ValueError(
                f"Parameter '{self.name}' must have exactly one of: "
                "(min_val, max_val), values, or fixed"
            )
    
    @classmethod
    def from_dict(cls, name: str, d: Dict) -> 'ParameterRange':
        """Create from dictionary (YAML parsing)."""
        if isinstance(d, (int, float)):
            # Shorthand for fixed value
            return cls(name=name, fixed=float(d))
        elif isinstance(d, list):
            # Shorthand for discrete values
            return cls(name=name, values=d)
        else:
            # Parse values, converting strings to float if needed (e.g., "1.0e3")
            min_val = d.get('min')
            max_val = d.get('max')
            if min_val is not None:
                min_val = float(min_val)
            if max_val is not None:
                max_val = float(max_val)
            
            return cls(
                name=name,
                min_val=min_val,
                max_val=max_val,
                values=d.get('values'),
                log_scale=d.get('log_scale', False),
                fixed=float(d['fixed']) if d.get('fixed') is not None else None,
            )


@dataclass
class ProblemConfig:
    """
    Configuration for a specific problem type.
    
    Contains parameter ranges that define the space of problems to generate.
    """
    problem_type: ProblemType
    param_ranges: Dict[str, ParameterRange] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ProblemConfig':
        """Create from dictionary."""
        problem_type = ProblemType(d['type'])
        param_ranges = {}
        for name, spec in d.get('parameters', {}).items():
            param_ranges[name] = ParameterRange.from_dict(name, spec)
        return cls(problem_type=problem_type, param_ranges=param_ranges)


@dataclass
class TrainConfig:
    """Training configuration for a case study."""
    grid_size: int  # Primary grid size (or first in list)
    num_samples: int
    seed: int = 42
    
    # Multi-grid training support
    grid_sizes: Optional[List[int]] = None  # For training on multiple sizes
    samples_per_size: Optional[int] = None  # Samples per grid size
    
    # Training hyperparameters
    epochs: int = 50
    batch_size: int = 1
    learning_rate: float = 1e-3
    
    # Exploration/exploitation tuning
    entropy_coef: Optional[float] = None  # Override default entropy coefficient
    temperature: Optional[Dict[str, Any]] = None  # Override temperature settings
    
    # AMG hierarchy settings
    max_levels: Optional[int] = None  # Max levels in AMG hierarchy (None = auto)
    max_coarse: int = 10  # Max coarse grid size
    
    # Model config reference (or inline)
    model_config: Optional[str] = None  # Path to model config YAML
    
    @classmethod
    def from_dict(cls, d: Dict, training_overrides: Optional[Dict] = None) -> 'TrainConfig':
        # Handle both single grid_size and multi-grid grid_sizes
        if 'grid_sizes' in d:
            grid_sizes = d['grid_sizes']
            grid_size = grid_sizes[0]  # Use first as primary
            samples_per_size = d.get('samples_per_size', 10)
            num_samples = d.get('num_samples', len(grid_sizes) * samples_per_size)
        else:
            grid_size = d['grid_size']
            grid_sizes = None
            samples_per_size = None
            num_samples = d['num_samples']
        
        # Merge training overrides (from top-level training: section)
        t = training_overrides or {}
        
        return cls(
            grid_size=grid_size,
            num_samples=num_samples,
            seed=d.get('seed', 42),
            grid_sizes=grid_sizes,
            samples_per_size=samples_per_size,
            epochs=t.get('epochs', d.get('epochs', 50)),
            batch_size=t.get('batch_size', d.get('batch_size', 1)),
            learning_rate=t.get('learning_rate', d.get('learning_rate', 1e-3)),
            entropy_coef=t.get('entropy_coef'),
            temperature=t.get('temperature'),
            max_levels=t.get('max_levels'),
            max_coarse=t.get('max_coarse', 10),
            model_config=d.get('model_config'),
        )


@dataclass
class TestConfig:
    """Configuration for a single test variation."""
    name: str
    grid_sizes: List[int]
    num_samples_per_size: int
    seed: int = 123
    
    # For variation B (scaling study)
    is_scaling_study: bool = False
    
    # Metrics to collect
    collect_wall_time: bool = True
    collect_energy_history: bool = True
    collect_iterations: bool = True
    collect_spectral: bool = False  # Expensive for large matrices
    
    @classmethod
    def from_dict(cls, name: str, d: Dict) -> 'TestConfig':
        grid_sizes = d.get('grid_sizes', [d.get('grid_size', 32)])
        if isinstance(grid_sizes, int):
            grid_sizes = [grid_sizes]
            
        return cls(
            name=name,
            grid_sizes=grid_sizes,
            num_samples_per_size=d.get('num_samples', 10),
            seed=d.get('seed', 123),
            is_scaling_study=d.get('scaling_study', False),
            collect_wall_time=d.get('collect_wall_time', True),
            collect_energy_history=d.get('collect_energy_history', True),
            collect_iterations=d.get('collect_iterations', True),
            collect_spectral=d.get('collect_spectral', False),
        )


@dataclass 
class VariationConfig:
    """
    Configuration for a case study variation (A, B, C, etc.)
    
    Each variation has its own train and test configurations.
    
    Special flags:
    - use_random_init: If True, skip training and use random model weights.
                       Useful for ablation studies comparing trained vs random.
    - compare_with_trained: Path to trained checkpoint to include in comparison.
                           When set with use_random_init=True, evaluation will
                           compare: trained, random, baseline, and tuned solvers.
    """
    name: str
    description: str
    train: TrainConfig
    test: TestConfig
    use_random_init: bool = False  # If True, skip training and evaluate random model
    compare_with_trained: Optional[str] = None  # Path to trained checkpoint for comparison
    
    @classmethod
    def from_dict(cls, name: str, d: Dict, training_overrides: Optional[Dict] = None) -> 'VariationConfig':
        return cls(
            name=name,
            description=d.get('description', ''),
            train=TrainConfig.from_dict(d['train'], training_overrides),
            test=TestConfig.from_dict(name, d['test']),
            use_random_init=d.get('use_random_init', False),
            compare_with_trained=d.get('compare_with_trained', None),
        )


@dataclass
class CaseStudyConfig:
    """
    Complete configuration for a case study.
    
    A case study consists of:
    - A problem type with parameter ranges
    - Multiple variations (A, B, C, ...) each with train/test configs
    - Output/results configuration
    """
    name: str
    description: str
    problem: ProblemConfig
    variations: Dict[str, VariationConfig]
    output_dir: str = "case_studies/results"
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'CaseStudyConfig':
        # Get top-level training overrides
        training_overrides = d.get('training', {})
        
        variations = {}
        for var_name, var_config in d.get('variations', {}).items():
            variations[var_name] = VariationConfig.from_dict(var_name, var_config, training_overrides)
        
        return cls(
            name=d['name'],
            description=d.get('description', ''),
            problem=ProblemConfig.from_dict(d['problem']),
            variations=variations,
            output_dir=d.get('output_dir', 'case_studies/results'),
        )
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'CaseStudyConfig':
        """Load case study configuration from YAML file."""
        with open(path, 'r') as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        # Convert to dict for serialization
        d = {
            'name': self.name,
            'description': self.description,
            'problem': {
                'type': self.problem.problem_type.value,
                'parameters': {
                    name: self._param_range_to_dict(pr)
                    for name, pr in self.problem.param_ranges.items()
                }
            },
            'variations': {
                name: self._variation_to_dict(var)
                for name, var in self.variations.items()
            },
            'output_dir': self.output_dir,
        }
        with open(path, 'w') as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)
    
    def _param_range_to_dict(self, pr: ParameterRange) -> Dict:
        if pr.fixed is not None:
            return pr.fixed
        elif pr.values is not None:
            return pr.values
        else:
            d = {'min': pr.min_val, 'max': pr.max_val}
            if pr.log_scale:
                d['log_scale'] = True
            return d
    
    def _variation_to_dict(self, var: VariationConfig) -> Dict:
        return {
            'description': var.description,
            'train': {
                'grid_size': var.train.grid_size,
                'num_samples': var.train.num_samples,
                'seed': var.train.seed,
                'epochs': var.train.epochs,
                'batch_size': var.train.batch_size,
                'learning_rate': var.train.learning_rate,
            },
            'test': {
                'grid_sizes': var.test.grid_sizes,
                'num_samples': var.test.num_samples_per_size,
                'seed': var.test.seed,
                'scaling_study': var.test.is_scaling_study,
            }
        }


# Convenience function for quick config creation
def create_anisotropic_case_study(
    name: str = "case_study_1",
    train_grid: int = 32,
    train_samples: int = 30,
    test_grids_a: List[int] = None,
    test_samples_a: int = 50,
    scaling_grids: List[int] = None,
) -> CaseStudyConfig:
    """
    Create a standard anisotropic case study configuration.
    
    This is a convenience function for the common case study structure.
    """
    if test_grids_a is None:
        test_grids_a = [train_grid]
    if scaling_grids is None:
        scaling_grids = [64, 128, 256, 512, 1024]
    
    return CaseStudyConfig(
        name=name,
        description="Anisotropic diffusion case study",
        problem=ProblemConfig(
            problem_type=ProblemType.ANISOTROPIC,
            param_ranges={
                'theta': ParameterRange(name='theta', min_val=0.0, max_val=6.28),
                'epsilon': ParameterRange(name='epsilon', min_val=0.001, max_val=0.1, log_scale=True),
            }
        ),
        variations={
            'A': VariationConfig(
                name='A',
                description='Same grid size train/test with disjoint parameters',
                train=TrainConfig(grid_size=train_grid, num_samples=train_samples),
                test=TestConfig(
                    name='A',
                    grid_sizes=test_grids_a,
                    num_samples_per_size=test_samples_a,
                ),
            ),
            'B': VariationConfig(
                name='B',
                description='Scaling study: train small, test across sizes',
                train=TrainConfig(grid_size=train_grid, num_samples=train_samples),
                test=TestConfig(
                    name='B',
                    grid_sizes=scaling_grids,
                    num_samples_per_size=10,
                    is_scaling_study=True,
                    collect_spectral=False,  # Too expensive for large
                ),
            ),
            'C': VariationConfig(
                name='C',
                description='Inverse: train large, test small',
                train=TrainConfig(grid_size=128, num_samples=train_samples),
                test=TestConfig(
                    name='C',
                    grid_sizes=[32],
                    num_samples_per_size=test_samples_a,
                ),
            ),
        },
    )


def load_case_study_config(path: Union[str, Path]) -> CaseStudyConfig:
    """
    Load a case study configuration from a YAML file.
    
    This is a convenience function that wraps CaseStudyConfig.from_yaml()
    with additional parsing for the flexible YAML schema used in config files.
    
    Args:
        path: Path to the YAML configuration file
        
    Returns:
        Parsed CaseStudyConfig instance
        
    Example YAML structure:
        name: "study_name"
        description: "Description of the study"
        problem_type: "anisotropic"
        parameter_ranges:
          theta: {min: 0, max: 6.28}
          epsilon: {min: 0.001, max: 0.5, scale: "log"}
        variations:
          A:
            train: {grid_sizes: [64], samples_per_size: 50}
            test: {grid_sizes: [64], samples_per_size: 50}
    """
    path = Path(path)
    
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    
    # Handle the flexible config format from our YAML files
    name = d.get('name', path.stem)
    description = d.get('description', '')
    
    # Parse problem type
    problem_type_str = d.get('problem_type', 'anisotropic')
    problem_type = ProblemType(problem_type_str)
    
    # Parse parameter ranges
    param_ranges = {}
    for param_name, param_spec in d.get('parameter_ranges', {}).items():
        if isinstance(param_spec, dict):
            min_val = param_spec.get('min')
            max_val = param_spec.get('max')
            log_scale = param_spec.get('scale') == 'log'
            values = param_spec.get('values')
            fixed = param_spec.get('fixed')
            
            param_ranges[param_name] = ParameterRange(
                name=param_name,
                min_val=min_val,
                max_val=max_val,
                values=values,
                log_scale=log_scale,
                fixed=fixed,
            )
        elif isinstance(param_spec, (int, float)):
            param_ranges[param_name] = ParameterRange(name=param_name, fixed=float(param_spec))
        elif isinstance(param_spec, list):
            param_ranges[param_name] = ParameterRange(name=param_name, values=param_spec)
    
    problem_config = ProblemConfig(problem_type=problem_type, param_ranges=param_ranges)
    
    # Parse variations
    variations = {}
    for var_name, var_spec in d.get('variations', {}).items():
        # Train config
        train_spec = var_spec.get('train', {})
        train_grid_sizes = train_spec.get('grid_sizes', [64])
        train_grid_size = train_grid_sizes[0] if isinstance(train_grid_sizes, list) else train_grid_sizes
        
        # Get training hyperparameters from top-level training section
        training_section = d.get('training', {})
        
        train_config = TrainConfig(
            grid_size=train_grid_size,
            num_samples=train_spec.get('samples_per_size', 50) * len(train_grid_sizes),
            seed=train_spec.get('seed', 42),
            epochs=training_section.get('epochs', 50),
            batch_size=training_section.get('batch_size', 8),
            learning_rate=training_section.get('learning_rate', 1e-3),
            entropy_coef=training_section.get('entropy_coef'),
            temperature=training_section.get('temperature'),
        )
        
        # Test config
        test_spec = var_spec.get('test', {})
        test_grid_sizes = test_spec.get('grid_sizes', [64])
        is_scaling = len(test_grid_sizes) > 2 or test_spec.get('scaling_study', False)
        
        test_config = TestConfig(
            name=var_name,
            grid_sizes=test_grid_sizes,
            num_samples_per_size=test_spec.get('samples_per_size', 10),
            seed=test_spec.get('seed', 123),
            is_scaling_study=is_scaling,
        )
        
        variations[var_name] = VariationConfig(
            name=var_name,
            description=var_spec.get('description', ''),
            train=train_config,
            test=test_config,
        )
    
    output_dir = d.get('output', {}).get('base_dir', 'case_studies/results')
    
    return CaseStudyConfig(
        name=name,
        description=description,
        problem=problem_config,
        variations=variations,
        output_dir=output_dir,
    )
