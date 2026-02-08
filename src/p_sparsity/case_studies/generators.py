"""
Unified problem generator interface for case studies.

Provides a consistent API across all problem types (anisotropic, elasticity, helmholtz)
for use in case study experiments.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Type
import numpy as np
import scipy.sparse as sp

from .config import ProblemType, ParameterRange


@dataclass
class ProblemInstance:
    """
    A single problem instance with its matrix and metadata.
    
    This is the standard output format from all problem generators.
    """
    A: sp.csr_matrix
    grid_size: int
    params: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    coords: Optional[np.ndarray] = None
    
    @property
    def n(self) -> int:
        """Number of degrees of freedom."""
        return self.A.shape[0]
    
    @property
    def nnz(self) -> int:
        """Number of nonzeros."""
        return self.A.nnz
    
    def summary(self) -> str:
        """Get a summary string."""
        return (
            f"ProblemInstance(n={self.n}, nnz={self.nnz}, "
            f"grid={self.grid_size}x{self.grid_size}, "
            f"params={self.params})"
        )


class UnifiedProblemGenerator(ABC):
    """
    Abstract base class for unified problem generators.
    
    All generators follow the same interface:
    - generate(grid_size, params) -> ProblemInstance
    - get_default_param_ranges() -> Dict[str, ParameterRange]
    - sample_params(rng, ranges) -> Dict[str, float]
    """
    
    problem_type: ProblemType
    
    @abstractmethod
    def generate(self, grid_size: int, params: Dict[str, float]) -> ProblemInstance:
        """
        Generate a problem instance.
        
        Args:
            grid_size: Grid dimension (n x n grid -> n^2 DOFs for 2D)
            params: Parameter dictionary
            
        Returns:
            ProblemInstance with matrix, metadata, etc.
        """
        pass
    
    @abstractmethod
    def get_default_param_ranges(self) -> Dict[str, ParameterRange]:
        """
        Get default parameter ranges for this problem type.
        
        Returns:
            Dictionary of parameter name -> ParameterRange
        """
        pass
    
    def sample_params(
        self, 
        rng: np.random.Generator,
        ranges: Optional[Dict[str, ParameterRange]] = None
    ) -> Dict[str, float]:
        """
        Sample a random parameter set.
        
        Args:
            rng: Random number generator
            ranges: Parameter ranges (uses defaults if None)
            
        Returns:
            Dictionary of parameter name -> sampled value
        """
        if ranges is None:
            ranges = self.get_default_param_ranges()
        
        params = {}
        for name, pr in ranges.items():
            if pr.fixed is not None:
                params[name] = pr.fixed
            elif pr.values is not None:
                params[name] = float(rng.choice(pr.values))
            else:
                if pr.log_scale:
                    log_val = rng.uniform(np.log(pr.min_val), np.log(pr.max_val))
                    params[name] = float(np.exp(log_val))
                else:
                    params[name] = float(rng.uniform(pr.min_val, pr.max_val))
        
        return params
    
    def generate_batch(
        self,
        grid_size: int,
        num_samples: int,
        rng: np.random.Generator,
        ranges: Optional[Dict[str, ParameterRange]] = None
    ) -> List[Tuple[ProblemInstance, Dict[str, float]]]:
        """
        Generate multiple problem instances.
        
        Args:
            grid_size: Grid dimension
            num_samples: Number of samples to generate
            rng: Random number generator
            ranges: Parameter ranges
            
        Returns:
            List of (ProblemInstance, params) tuples
        """
        results = []
        for _ in range(num_samples):
            params = self.sample_params(rng, ranges)
            instance = self.generate(grid_size, params)
            results.append((instance, params))
        return results


class AnisotropicGenerator(UnifiedProblemGenerator):
    """
    Anisotropic diffusion problem generator.
    
    Generates: -div(K * grad(u)) = f
    where K is an anisotropic tensor with epsilon (anisotropy ratio)
    and theta (rotation angle).
    """
    
    problem_type = ProblemType.ANISOTROPIC
    
    def __init__(self, stencil_type: str = "FD"):
        self.stencil_type = stencil_type
    
    def generate(self, grid_size: int, params: Dict[str, float]) -> ProblemInstance:
        import pyamg
        
        epsilon = params.get('epsilon', 1.0)
        theta = params.get('theta', 0.0)
        
        stencil = pyamg.gallery.diffusion_stencil_2d(
            type=self.stencil_type,
            epsilon=epsilon,
            theta=theta
        )
        A = pyamg.gallery.stencil_grid(stencil, (grid_size, grid_size), format="csr")
        A = A.astype(np.float64)
        
        # Generate coordinates
        n = grid_size * grid_size
        coords = np.zeros((n, 2), dtype=np.float64)
        coords[:, 0] = np.tile(np.linspace(0, 1, grid_size), grid_size)
        coords[:, 1] = np.repeat(np.linspace(0, 1, grid_size), grid_size)
        
        metadata = {
            'problem_type': 'anisotropic',
            'stencil_type': self.stencil_type,
            'epsilon': epsilon,
            'theta': theta,
            'pde': '-div(K * grad(u)) = f',
        }
        
        return ProblemInstance(
            A=A,
            grid_size=grid_size,
            params=params.copy(),
            metadata=metadata,
            coords=coords,
        )
    
    def get_default_param_ranges(self) -> Dict[str, ParameterRange]:
        return {
            'theta': ParameterRange(name='theta', min_val=0.0, max_val=2*np.pi),
            'epsilon': ParameterRange(name='epsilon', min_val=0.001, max_val=1.0, log_scale=True),
        }


class ElasticityGenerator(UnifiedProblemGenerator):
    """
    Linear elasticity problem generator.
    
    Generates 2D linear elasticity problems with Lame parameters
    lambda and mu (or equivalently E and nu).
    """
    
    problem_type = ProblemType.ELASTICITY
    
    def generate(self, grid_size: int, params: Dict[str, float]) -> ProblemInstance:
        import pyamg
        
        # Get Lame parameters from E, nu
        E = params.get('E', 1e5)  # Young's modulus
        nu = params.get('nu', 0.3)  # Poisson's ratio
        
        # Convert to Lame parameters
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        
        # Generate elasticity stencil
        # PyAMG doesn't have built-in elasticity, so we use a simple approximation
        # For proper elasticity, would need FEM assembly
        try:
            A = pyamg.gallery.linear_elasticity((grid_size, grid_size))[0]
        except:
            # Fallback: use stencil-based approximation
            # This creates a block system approximating elasticity
            from pyamg.gallery import stencil_grid
            
            # 2D elasticity stencil (simplified)
            stencil = np.array([
                [0, -mu, 0],
                [-mu, 4*mu + 2*(lam + mu), -mu],
                [0, -mu, 0]
            ]) / (grid_size ** 2)
            
            A_scalar = stencil_grid(stencil, (grid_size, grid_size), format='csr')
            # Create 2x2 block system for 2D elasticity (ux, uy DOFs)
            n_scalar = grid_size * grid_size
            A = sp.bmat([
                [A_scalar, sp.csr_matrix((n_scalar, n_scalar))],
                [sp.csr_matrix((n_scalar, n_scalar)), A_scalar]
            ], format='csr')
        
        A = A.astype(np.float64)
        
        # Coordinates (need to handle vector-valued DOFs)
        n_nodes = grid_size * grid_size
        coords_nodes = np.zeros((n_nodes, 2), dtype=np.float64)
        coords_nodes[:, 0] = np.tile(np.linspace(0, 1, grid_size), grid_size)
        coords_nodes[:, 1] = np.repeat(np.linspace(0, 1, grid_size), grid_size)
        
        # For 2-component displacement, duplicate coordinates
        if A.shape[0] == 2 * n_nodes:
            coords = np.vstack([coords_nodes, coords_nodes])
        else:
            coords = coords_nodes
        
        metadata = {
            'problem_type': 'elasticity',
            'E': E,
            'nu': nu,
            'lam': lam,
            'mu': mu,
            'pde': 'linear elasticity',
        }
        
        return ProblemInstance(
            A=A,
            grid_size=grid_size,
            params=params.copy(),
            metadata=metadata,
            coords=coords,
        )
    
    def get_default_param_ranges(self) -> Dict[str, ParameterRange]:
        return {
            'E': ParameterRange(name='E', min_val=1e4, max_val=1e6, log_scale=True),
            'nu': ParameterRange(name='nu', min_val=0.1, max_val=0.45),  # < 0.5 for stability
        }


class HelmholtzGenerator(UnifiedProblemGenerator):
    """
    Helmholtz equation problem generator.
    
    Generates: -Δu - k²u = f
    where k is the wavenumber.
    """
    
    problem_type = ProblemType.HELMHOLTZ
    
    def generate(self, grid_size: int, params: Dict[str, float]) -> ProblemInstance:
        import pyamg
        from pyamg.gallery import stencil_grid
        
        k = params.get('k', 10.0)  # Wavenumber
        
        h = 1.0 / (grid_size - 1)
        
        # Laplacian stencil
        laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]) / h**2
        
        # Identity for -k² term
        identity_stencil = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        
        # Helmholtz stencil: -Δ - k²I
        helmholtz_stencil = laplacian - k**2 * identity_stencil
        
        A = stencil_grid(helmholtz_stencil, (grid_size, grid_size), format='csr')
        A = A.astype(np.float64)
        
        # Note: For high k, the matrix becomes indefinite
        # This is challenging for AMG!
        
        n = grid_size * grid_size
        coords = np.zeros((n, 2), dtype=np.float64)
        coords[:, 0] = np.tile(np.linspace(0, 1, grid_size), grid_size)
        coords[:, 1] = np.repeat(np.linspace(0, 1, grid_size), grid_size)
        
        # Compute points per wavelength (quality measure)
        wavelength = 2 * np.pi / k if k > 0 else np.inf
        ppw = wavelength / h if k > 0 else np.inf
        
        metadata = {
            'problem_type': 'helmholtz',
            'k': k,
            'wavelength': wavelength,
            'points_per_wavelength': ppw,
            'h': h,
            'pde': '-Δu - k²u = f',
            'is_indefinite': k > 0,
        }
        
        return ProblemInstance(
            A=A,
            grid_size=grid_size,
            params=params.copy(),
            metadata=metadata,
            coords=coords,
        )
    
    def get_default_param_ranges(self) -> Dict[str, ParameterRange]:
        return {
            'k': ParameterRange(name='k', min_val=1.0, max_val=50.0),
        }


class PoissonGenerator(UnifiedProblemGenerator):
    """
    Simple Poisson equation generator (isotropic).
    
    Generates: -Δu = f
    """
    
    problem_type = ProblemType.POISSON
    
    def generate(self, grid_size: int, params: Dict[str, float]) -> ProblemInstance:
        import pyamg
        from pyamg.gallery import stencil_grid
        
        stencil = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
        A = stencil_grid(stencil, (grid_size, grid_size), format='csr')
        A = A.astype(np.float64)
        
        n = grid_size * grid_size
        coords = np.zeros((n, 2), dtype=np.float64)
        coords[:, 0] = np.tile(np.linspace(0, 1, grid_size), grid_size)
        coords[:, 1] = np.repeat(np.linspace(0, 1, grid_size), grid_size)
        
        metadata = {
            'problem_type': 'poisson',
            'pde': '-Δu = f',
        }
        
        return ProblemInstance(
            A=A,
            grid_size=grid_size,
            params=params.copy(),
            metadata=metadata,
            coords=coords,
        )
    
    def get_default_param_ranges(self) -> Dict[str, ParameterRange]:
        # Poisson has no parameters
        return {}


# Factory for creating generators
GENERATOR_REGISTRY: Dict[ProblemType, Type[UnifiedProblemGenerator]] = {
    ProblemType.ANISOTROPIC: AnisotropicGenerator,
    ProblemType.ELASTICITY: ElasticityGenerator,
    ProblemType.HELMHOLTZ: HelmholtzGenerator,
    ProblemType.POISSON: PoissonGenerator,
}


def create_generator(problem_type: ProblemType, **kwargs) -> UnifiedProblemGenerator:
    """
    Create a problem generator for the given type.
    
    Args:
        problem_type: Type of problem
        **kwargs: Additional arguments for the generator
        
    Returns:
        UnifiedProblemGenerator instance
    """
    if problem_type not in GENERATOR_REGISTRY:
        raise ValueError(f"Unknown problem type: {problem_type}")
    
    generator_cls = GENERATOR_REGISTRY[problem_type]
    return generator_cls(**kwargs)


def get_generator_for_config(problem_config: 'ProblemConfig') -> UnifiedProblemGenerator:
    """
    Create a generator from a ProblemConfig.
    
    Args:
        problem_config: Problem configuration from case study
        
    Returns:
        Configured generator instance
    """
    return create_generator(problem_config.problem_type)
