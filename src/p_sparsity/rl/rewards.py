"""
Reward functions for RL training.

Pluggable reward functions that evaluate AMG solver quality.
"""

from typing import Dict, Callable
import numpy as np
import scipy.sparse as sp
import pyamg

# Global registry
_REWARD_REGISTRY: Dict[str, Callable] = {}


def register_reward(name: str):
    """Decorator to register a reward function."""
    def decorator(func: Callable):
        _REWARD_REGISTRY[name] = func
        return func
    return decorator


def get_reward_function(name: str) -> Callable:
    """Get reward function by name."""
    if name not in _REWARD_REGISTRY:
        raise KeyError(f"Reward function '{name}' not found. Available: {list(_REWARD_REGISTRY.keys())}")
    return _REWARD_REGISTRY[name]


def energy_norm_sq(A: sp.csr_matrix, e: np.ndarray) -> float:
    """Compute A-energy norm squared: e^T A e"""
    Ae = A @ e
    return float(e @ Ae)


def one_vcycle_error_reduce_ratio(
    ml: pyamg.multilevel.MultilevelSolver,
    A: sp.csr_matrix,
    e0: np.ndarray,
    cycle: str = "V"
) -> float:
    """
    Apply one V-cycle and return energy reduction ratio.
    
    Args:
        ml: PyAMG solver
        A: System matrix
        e0: Initial error vector
        cycle: Cycle type (V or W)
        
    Returns:
        ratio: ||e1||_A^2 / ||e0||_A^2
    """
    b0 = np.zeros_like(e0)
    
    try:
        # One cycle: maxiter=1, accel=None prevents Krylov acceleration
        e1 = ml.solve(b0, x0=e0, tol=0.0, maxiter=1, accel=None, cycle=cycle)
    except TypeError:
        # Some PyAMG versions use different signature
        e1 = ml.solve(b0, x0=e0, tol=1e-30, maxiter=1, accel=None, cycle=cycle)
    
    num = energy_norm_sq(A, e1)
    den = energy_norm_sq(A, e0) + 1e-30
    return num / den


@register_reward("vcycle_energy_reduction")
def vcycle_energy_reduction_reward(
    A: sp.csr_matrix,
    ml: pyamg.multilevel.MultilevelSolver,
    config: dict,
) -> float:
    """
    Reward based on V-cycle energy reduction.
    
    Reward = -mean(log(ratio)) - complexity_penalty
    where ratio = ||e_after||_A^2 / ||e_before||_A^2
    
    Args:
        A: System matrix
        ml: PyAMG solver
        config: Reward configuration dict
        
    Returns:
        reward: Higher is better
    """
    from ..data import relaxed_smooth_vectors
    
    vcycle_config = config.get("vcycle", {})
    num_test_vecs = vcycle_config.get("num_test_vecs", 6)
    relax_iters = vcycle_config.get("relax_iters", 25)
    omega = vcycle_config.get("omega", 2.0 / 3.0)
    cycle_type = vcycle_config.get("cycle_type", "V")
    
    # Generate smooth error vectors
    E = relaxed_smooth_vectors(
        A,
        num_vecs=num_test_vecs,
        iters=relax_iters,
        scheme="jacobi",
        omega=omega,
    )
    
    # Compute reduction ratios
    ratios = []
    for k in range(num_test_vecs):
        e0 = E[:, k].copy()
        ratio = one_vcycle_error_reduce_ratio(ml, A, e0, cycle=cycle_type)
        ratios.append(max(ratio, 1e-12))
    
    # Base reward: negative mean log ratio (minimize convergence factor)
    mean_log = float(np.mean(np.log(np.array(ratios))))
    base_reward = -mean_log
    
    # Operator complexity penalty
    complexity_config = config.get("complexity", {})
    target = complexity_config.get("target", 1.35)
    penalty_weight = complexity_config.get("penalty_weight", 1.0)
    
    A0_nnz = ml.levels[0].A.nnz
    total_A_nnz = sum(lvl.A.nnz for lvl in ml.levels)
    op_complex = float(total_A_nnz / max(A0_nnz, 1))
    
    penalty = penalty_weight * max(0.0, op_complex - target)
    
    return base_reward - penalty
