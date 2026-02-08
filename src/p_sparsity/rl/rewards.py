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


@register_reward("pcg_residual_reduction")
def pcg_residual_reduction_reward(
    A: sp.csr_matrix,
    ml: pyamg.multilevel.MultilevelSolver,
    config: dict,
) -> float:
    """
    Reward based on PCG residual reduction after fixed iterations.
    
    This directly optimizes for PCG performance rather than V-cycle energy reduction.
    
    Reward = ||r_0|| / ||r_k|| (residual reduction factor)
    where k is a small fixed number of PCG iterations.
    
    Args:
        A: System matrix  
        ml: PyAMG solver (used as preconditioner)
        config: Reward configuration dict with 'pcg' sub-dict:
            - num_iters: Number of PCG iterations (default: 5)
            - num_test_vecs: Number of random RHS to average over (default: 3)
            
    Returns:
        reward: Higher is better (larger residual reduction = better preconditioner)
    """
    import scipy.sparse.linalg as spla
    
    pcg_config = config.get("pcg", {})
    num_iters = pcg_config.get("num_iters", 5)
    num_test_vecs = pcg_config.get("num_test_vecs", 3)
    
    # Get preconditioner
    M = ml.aspreconditioner(cycle='V')
    
    reduction_factors = []
    
    for _ in range(num_test_vecs):
        # Random RHS
        b = np.random.randn(A.shape[0])
        r0_norm = np.linalg.norm(b)  # Initial residual (x0=0)
        
        # Track residual during PCG
        residuals = [r0_norm]
        def callback(xk):
            # Compute residual: r = b - A @ xk
            r = b - A @ xk
            residuals.append(np.linalg.norm(r))
        
        # Run PCG for exactly num_iters iterations
        # Use very loose tolerance so we hit maxiter
        try:
            x, info = spla.cg(A, b, M=M, maxiter=num_iters, rtol=1e-30, callback=callback)
        except Exception:
            # If CG fails, return small reward
            reduction_factors.append(1.0)
            continue
        
        # Compute convergence factor (per-iteration residual reduction)
        # rho = (r_k / r_0)^(1/k) => log(rho) = (1/k) * log(r_k/r_0)
        # We want reward = -log(rho) = (1/k) * log(r_0/r_k)
        final_residual = residuals[-1] if len(residuals) > 1 else r0_norm
        k = len(residuals) - 1  # Actual number of iterations
        if k > 0 and final_residual > 0:
            # Log convergence factor (higher = faster convergence = better)
            # This is log(r0/rk)/k, bounded and stable
            log_reduction = np.log(r0_norm / final_residual) / k
            reduction_factors.append(log_reduction)
        else:
            reduction_factors.append(0.0)
    
    # Average log-reduction as reward (typical range: 0.5 - 3.0)
    mean_reduction = float(np.mean(reduction_factors))
    
    # Operator complexity penalty (optional)
    complexity_config = config.get("complexity", {})
    target = complexity_config.get("target", 1.35)
    penalty_weight = complexity_config.get("penalty_weight", 0.0)  # Default: no penalty
    
    if penalty_weight > 0:
        A0_nnz = ml.levels[0].A.nnz
        total_A_nnz = sum(lvl.A.nnz for lvl in ml.levels)
        op_complex = float(total_A_nnz / max(A0_nnz, 1))
        penalty = penalty_weight * max(0.0, op_complex - target)
        mean_reduction -= penalty
    
    return mean_reduction
