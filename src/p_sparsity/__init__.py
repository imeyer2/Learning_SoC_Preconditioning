"""
P-Sparsity: Learning AMG Preconditioner Sparsity Patterns with RL

A modular framework for training GNN-based policies to optimize
algebraic multigrid preconditioner patterns across various PDE types.
"""

__version__ = "0.1.0"

from . import data
from . import models
from . import utils

__all__ = ["data", "models", "utils"]
