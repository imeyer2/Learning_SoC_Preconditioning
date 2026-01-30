"""
Base classes for problem generators.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np
import scipy.sparse as sp
import torch


@dataclass
class TrainSample:
    """Container for a single training sample."""
    
    A: sp.csr_matrix              # Sparse matrix
    grid_n: int                   # Grid size (for 2D: grid_n x grid_n)
    coords: np.ndarray            # Node coordinates (N, d)
    edge_index: torch.Tensor      # (2, E) edge connectivity
    edge_weight: torch.Tensor     # (E,) edge weights
    x: torch.Tensor               # (N, F) node features
    row_groups: List[torch.Tensor]  # Per-row edge indices for sampling
    metadata: Optional[Dict[str, Any]] = None  # Problem-specific metadata


class ProblemGenerator(ABC):
    """
    Abstract base class for problem generators.
    
    All problem generators must implement:
    - generate(): Create the sparse matrix A
    - get_coordinates(): Return node coordinates
    - get_metadata(): Return problem metadata
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize generator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    @abstractmethod
    def generate(self, **kwargs) -> sp.csr_matrix:
        """
        Generate a sparse matrix problem.
        
        Returns:
            A: Sparse matrix in CSR format
        """
        pass
    
    @abstractmethod
    def get_coordinates(self, A: sp.csr_matrix) -> np.ndarray:
        """
        Get node coordinates for the problem.
        
        Args:
            A: Sparse matrix
            
        Returns:
            coords: (N, d) array of coordinates
        """
        pass
    
    def get_metadata(self, A: sp.csr_matrix) -> Dict[str, Any]:
        """
        Get metadata about the problem.
        
        Args:
            A: Sparse matrix
            
        Returns:
            metadata: Dictionary with problem information
        """
        return {
            "problem_type": self.__class__.__name__,
            "matrix_shape": A.shape,
            "nnz": A.nnz,
        }
    
    def get_grid_size(self, A: sp.csr_matrix) -> int:
        """
        Get grid size (for structured grids).
        
        Args:
            A: Sparse matrix
            
        Returns:
            grid_n: Grid size
        """
        # Default: assume square 2D grid
        n = A.shape[0]
        grid_n = int(np.sqrt(n))
        if grid_n * grid_n != n:
            raise ValueError(f"Matrix size {n} is not a perfect square")
        return grid_n
