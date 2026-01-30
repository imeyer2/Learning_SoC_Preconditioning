"""
Helmholtz problem generator.

Placeholder for Helmholtz (shifted Laplacian) problems.
"""

from typing import Dict, Any
import numpy as np
import scipy.sparse as sp

from ..base import ProblemGenerator
from ..registry import register_generator


@register_generator("helmholtz")
class HelmholtzGenerator(ProblemGenerator):
    """
    Generator for Helmholtz (shifted Laplacian) problems.
    
    Generates matrices from:
        -Δu + k²u = f
    
    where k is the wave number.
    
    TODO: Implement using PyAMG's Helmholtz gallery or custom implementation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.wave_number = config.get("wave_number", 10.0)
    
    def generate(
        self,
        grid_size: int,
        wave_number: float = None,
        **kwargs
    ) -> sp.csr_matrix:
        """
        Generate Helmholtz matrix.
        
        Args:
            grid_size: Grid size
            wave_number: Wave number k (defaults to config value)
            
        Returns:
            A: Sparse matrix in CSR format
        """
        if wave_number is None:
            wave_number = self.wave_number
        
        # TODO: Implement proper Helmholtz with wave number
        # For now, return Poisson + shift as placeholder
        from pyamg import gallery
        A = gallery.poisson((grid_size, grid_size), format='csr')
        
        # Add shift: A_helm = A + k²*I
        n = A.shape[0]
        shift = sp.eye(n, format='csr') * (wave_number ** 2)
        A_helm = A + shift
        
        print(f"WARNING: HelmholtzGenerator is a placeholder. Using Poisson + k²I with k={wave_number}.")
        return A_helm
    
    def get_coordinates(self, A: sp.csr_matrix) -> np.ndarray:
        """Get 2D grid coordinates."""
        n = A.shape[0]
        grid_n = int(np.sqrt(n))
        if grid_n * grid_n != n:
            raise ValueError(f"Matrix size {n} is not a perfect square")
        
        coords = np.zeros((n, 2), dtype=np.float64)
        coords[:, 0] = np.tile(np.linspace(0, 1, grid_n), grid_n)
        coords[:, 1] = np.repeat(np.linspace(0, 1, grid_n), grid_n)
        return coords
    
    def get_metadata(self, A: sp.csr_matrix, **params) -> Dict[str, Any]:
        """Get problem metadata."""
        metadata = super().get_metadata(A)
        metadata.update({
            "problem_type": "helmholtz",
            "pde_type": "elliptic_indefinite",
            "wave_number": params.get("wave_number", self.wave_number),
        })
        return metadata
