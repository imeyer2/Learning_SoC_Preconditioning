"""
Anisotropic diffusion problem generator.
"""

import math
from typing import Dict, Any
import numpy as np
import scipy.sparse as sp
import pyamg

from ..base import ProblemGenerator
from ..registry import register_generator


@register_generator("anisotropic")
class AnisotropicDiffusionGenerator(ProblemGenerator):
    """
    Generator for 2D anisotropic diffusion problems.
    
    Generates matrices from:
        -div(K * grad(u)) = f
    
    where K is an anisotropic diffusion tensor with parameter epsilon
    and rotation angle theta.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.stencil_type = config.get("stencil_type", "FD")
    
    def generate(
        self,
        grid_size: int,
        epsilon: float = 1.0,
        theta: float = 0.0,
        **kwargs
    ) -> sp.csr_matrix:
        """
        Generate anisotropic diffusion matrix.
        
        Args:
            grid_size: Grid size (n x n)
            epsilon: Anisotropy parameter (0 < epsilon <= 1)
            theta: Rotation angle in radians
            
        Returns:
            A: Sparse matrix in CSR format
        """
        stencil = pyamg.gallery.diffusion_stencil_2d(
            type=self.stencil_type,
            epsilon=epsilon,
            theta=theta
        )
        A = pyamg.gallery.stencil_grid(stencil, (grid_size, grid_size), format="csr")
        return A
    
    def get_coordinates(self, A: sp.csr_matrix) -> np.ndarray:
        """
        Get 2D grid coordinates.
        
        Args:
            A: Sparse matrix
            
        Returns:
            coords: (N, 2) coordinates
        """
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
            "problem_type": "anisotropic_diffusion",
            "pde_type": "elliptic",
            "stencil_type": self.stencil_type,
            "epsilon": params.get("epsilon"),
            "theta": params.get("theta"),
        })
        return metadata


def generate_training_modes(config: Dict[str, Any]) -> list:
    """
    Generate parameter modes for training set variation.
    
    Args:
        config: Configuration with 'modes' list
        
    Returns:
        params_list: List of parameter dicts
    """
    modes = config.get("modes", [
        {"epsilon": 1.0, "theta": 0.0},
        {"epsilon": 0.001, "theta": 0.0},
        {"epsilon": 0.001, "theta_range": [0.0, math.pi / 2]},
    ])
    
    params_list = []
    for mode in modes:
        if "theta_range" in mode:
            # Parameterized mode - will sample theta from range
            params_list.append({
                "epsilon": mode["epsilon"],
                "theta_range": mode["theta_range"],
            })
        else:
            # Fixed mode
            params_list.append({
                "epsilon": mode["epsilon"],
                "theta": mode["theta"],
            })
    
    return params_list


def sample_anisotropic_params(modes: list, index: int, rng: np.random.Generator) -> Dict:
    """
    Sample parameters for a specific training sample.
    
    Args:
        modes: List of parameter modes
        index: Sample index
        rng: Random number generator
        
    Returns:
        params: Parameter dictionary with epsilon and theta
    """
    mode_idx = index % len(modes)
    mode = modes[mode_idx]
    
    params = {"epsilon": mode["epsilon"]}
    
    if "theta_range" in mode:
        # Sample theta uniformly from range
        theta_min, theta_max = mode["theta_range"]
        params["theta"] = float(rng.uniform(theta_min, theta_max))
    else:
        params["theta"] = mode["theta"]
    
    return params
