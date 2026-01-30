"""
Data module for P-Sparsity.

Provides problem generators, smooth error generation, and dataset utilities.
"""

from .base import ProblemGenerator, TrainSample
from .registry import register_generator, get_generator, list_generators
from .smooth_errors import (
    relaxed_smooth_vectors,
    generate_smoothed_errors,
    node_features_for_policy,
)
from .dataset import make_dataset
from . import generators

__all__ = [
    "ProblemGenerator",
    "TrainSample",
    "register_generator",
    "get_generator",
    "list_generators",
    "relaxed_smooth_vectors",
    "generate_smoothed_errors",
    "node_features_for_policy",
    "make_dataset",
]
