"""
Dataset builder for training and validation.
"""

from typing import List, Dict, Any
import numpy as np
from torch_geometric.utils import from_scipy_sparse_matrix
import torch

from .base import TrainSample
from .registry import get_generator
from .smooth_errors import node_features_for_policy
from .generators.anisotropic import generate_training_modes, sample_anisotropic_params


def build_row_groups(edge_index: torch.Tensor, num_nodes: int) -> List[torch.Tensor]:
    """
    Group edges by source node for per-row sampling.
    
    Args:
        edge_index: (2, E) edge connectivity
        num_nodes: Number of nodes
        
    Returns:
        row_groups: List of edge index tensors, one per node
    """
    row = edge_index[0].cpu().numpy()
    buckets: List[List[int]] = [[] for _ in range(num_nodes)]
    
    for e, r in enumerate(row):
        buckets[int(r)].append(e)
    
    return [torch.tensor(b, dtype=torch.long) for b in buckets]


def make_train_sample(
    generator,
    grid_size: int,
    params: Dict[str, Any],
    feature_config: Dict[str, Any],
) -> TrainSample:
    """
    Create a single training sample.
    
    Args:
        generator: Problem generator instance
        grid_size: Grid size
        params: Problem-specific parameters
        feature_config: Node feature configuration
        
    Returns:
        sample: TrainSample instance
    """
    # Generate matrix
    A = generator.generate(grid_size=grid_size, **params)
    
    # Get coordinates
    coords = generator.get_coordinates(A)
    
    # Build graph
    ei, ew = from_scipy_sparse_matrix(A)
    ew = ew.float()
    
    # Normalize edge weights
    if feature_config.get("normalize_weights", True):
        max_w = ew.abs().max()
        if max_w > 0:
            ew = ew / max_w
    
    # Build node features
    x = node_features_for_policy(A, coords, feature_config)
    
    # Build row groups for sampling
    row_groups = build_row_groups(ei, num_nodes=A.shape[0])
    
    # Get metadata
    metadata = generator.get_metadata(A, **params)
    
    return TrainSample(
        A=A,
        grid_n=grid_size,
        coords=coords,
        edge_index=ei,
        edge_weight=ew,
        x=x,
        row_groups=row_groups,
        metadata=metadata,
    )


def make_dataset(
    problem_type: str,
    num_samples: int,
    grid_size: int,
    config: Dict[str, Any],
    seed: int = 0,
) -> List[TrainSample]:
    """
    Create a dataset of training samples.
    
    Args:
        problem_type: Type of problem (anisotropic, elasticity, etc.)
        num_samples: Number of samples to generate
        grid_size: Grid size for problems
        config: Full data configuration
        seed: Random seed
        
    Returns:
        dataset: List of TrainSample instances
    """
    rng = np.random.default_rng(seed)
    
    # Get generator
    problem_config = config.get(problem_type, {})
    generator = get_generator(problem_type, problem_config)
    
    # Get feature config
    feature_config = {
        "use_relaxed_vectors": config.get("node_features", {}).get("use_relaxed_vectors", True),
        "use_coordinates": config.get("node_features", {}).get("use_coordinates", True),
        "use_degree": config.get("node_features", {}).get("use_degree", False),
        "use_random": config.get("node_features", {}).get("use_random", False),
        "normalize": config.get("node_features", {}).get("normalize", True),
        "normalize_weights": config.get("graph", {}).get("normalize_weights", True),
        **config.get("smooth_errors", {}),
    }
    
    # Generate parameter modes (problem-specific)
    if problem_type == "anisotropic":
        modes = generate_training_modes(problem_config)
    else:
        # Default: single mode with no variation
        modes = [{}]
    
    # Generate samples
    dataset = []
    for i in range(num_samples):
        # Sample parameters
        if problem_type == "anisotropic":
            params = sample_anisotropic_params(modes, i, rng)
        else:
            params = modes[i % len(modes)]
        
        # Create sample
        sample = make_train_sample(generator, grid_size, params, feature_config)
        dataset.append(sample)
    
    return dataset
