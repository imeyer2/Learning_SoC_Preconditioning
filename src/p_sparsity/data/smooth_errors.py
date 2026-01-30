"""
Smooth error generation for policy input features.
"""

from typing import Optional
import numpy as np
import scipy.sparse as sp
import torch
import pyamg


def relaxed_smooth_vectors(
    A: sp.csr_matrix,
    num_vecs: int,
    iters: int,
    scheme: str = "jacobi",
    omega: float = 2.0 / 3.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate smooth error samples by relaxing Ax = 0 from random initial errors.
    
    Args:
        A: Sparse matrix
        num_vecs: Number of vectors to generate
        iters: Number of relaxation iterations
        scheme: Relaxation scheme (jacobi, gauss_seidel, richardson)
        omega: Relaxation parameter
        seed: Random seed
        
    Returns:
        X: (N, num_vecs) array of smooth vectors
    """
    n = A.shape[0]
    
    if seed is not None:
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, num_vecs))
    else:
        X = np.random.randn(n, num_vecs)
    
    B = np.zeros((n, num_vecs))
    
    for k in range(num_vecs):
        x = np.ascontiguousarray(X[:, k])
        b = np.ascontiguousarray(B[:, k])
        
        if scheme == "jacobi":
            pyamg.relaxation.relaxation.jacobi(A, x, b, iterations=iters, omega=omega)
        elif scheme == "gauss_seidel":
            pyamg.relaxation.relaxation.gauss_seidel(A, x, b, iterations=iters, sweep='forward')
        elif scheme == "richardson":
            # Richardson: x_new = x_old + omega * (b - A*x_old)
            for _ in range(iters):
                r = b - A @ x
                x += omega * r
        else:
            raise ValueError(f"Unknown relaxation scheme: {scheme}")
        
        X[:, k] = x
    
    # Normalize
    X = X / (np.abs(X).max() + 1e-6)
    return X


def generate_smoothed_errors(
    A: sp.csr_matrix,
    config: dict,
) -> np.ndarray:
    """
    Generate smoothed errors based on configuration.
    
    Args:
        A: Sparse matrix
        config: Configuration dict with keys:
            - num_vecs: Number of vectors
            - relax_iters: Relaxation iterations
            - relaxation_scheme: Scheme name
            - omega: Relaxation parameter
            - seed: Random seed
            
    Returns:
        X: (N, num_vecs) smooth error vectors
    """
    return relaxed_smooth_vectors(
        A=A,
        num_vecs=config.get("num_vecs", 4),
        iters=config.get("relax_iters", 5),
        scheme=config.get("relaxation_scheme", "jacobi"),
        omega=config.get("omega", 2.0 / 3.0),
        seed=config.get("seed"),
    )


def node_features_for_policy(
    A: sp.csr_matrix,
    coords: np.ndarray,
    config: dict,
) -> torch.Tensor:
    """
    Build node features for policy input.
    
    Args:
        A: Sparse matrix
        coords: (N, d) node coordinates
        config: Configuration dict
        
    Returns:
        x: (N, F) node feature tensor
    """
    features = []
    
    # Relaxed vectors
    if config.get("use_relaxed_vectors", True):
        X_relaxed = generate_smoothed_errors(A, config)
        features.append(X_relaxed)
    
    # Coordinates
    if config.get("use_coordinates", True):
        features.append(coords)
    
    # Degree
    if config.get("use_degree", False):
        degrees = np.array(A.sum(axis=1)).flatten()
        # Normalize
        degrees = degrees / (degrees.max() + 1e-6)
        features.append(degrees[:, None])
    
    # Random features (useful baseline)
    if config.get("use_random", False):
        num_random = config.get("num_random", 2)
        random_feat = np.random.randn(A.shape[0], num_random)
        features.append(random_feat)
    
    # Concatenate
    x = np.hstack(features) if len(features) > 1 else features[0]
    
    # Normalize
    if config.get("normalize", True):
        # Per-feature normalization
        x_mean = x.mean(axis=0, keepdims=True)
        x_std = x.std(axis=0, keepdims=True) + 1e-6
        x = (x - x_mean) / x_std
    
    return torch.tensor(x, dtype=torch.float32)
