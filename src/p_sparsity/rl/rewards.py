"""
Reward functions for RL training.

Pluggable reward functions that evaluate AMG solver quality.
"""

from typing import Dict, Callable, Tuple
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


def _compute_reward_worker(args: Tuple) -> Tuple[float, dict]:
    """
    Worker function for parallel reward computation.
    
    Takes pickled inputs and returns reward and hierarchy info. Used with multiprocessing.
    
    Args:
        args: Tuple of (A_data, C_data, B, reward_name, reward_config, pyamg_config)
              where A_data and C_data are tuples of (data, indices, indptr, shape)
              
    Returns:
        Tuple of (reward, hierarchy_info dict)
    """
    A_data, C_data, B, reward_name, reward_config, pyamg_config = args
    
    # Reconstruct sparse matrices
    A = sp.csr_matrix((A_data[0], A_data[1], A_data[2]), shape=A_data[3])
    C = sp.csr_matrix((C_data[0], C_data[1], C_data[2]), shape=C_data[3])
    
    try:
        # Build PyAMG solver
        from ..pyamg_interface import build_pyamg_solver
        ml = build_pyamg_solver(
            A, C, B,
            coarse_solver=pyamg_config.get("coarse_solver", "splu"),
            max_coarse=pyamg_config.get("max_coarse", 50),
        )
        
        # Collect hierarchy info
        dofs = [lvl.A.shape[0] for lvl in ml.levels]
        nnzs = [lvl.A.nnz for lvl in ml.levels]
        hierarchy_info = {
            'num_levels': len(ml.levels),
            'dofs': dofs,
            'coarsest_dofs': dofs[-1],
            'operator_complexity': sum(nnzs) / nnzs[0] if nnzs[0] > 0 else 0,
        }
        
        # Compute reward
        reward_fn = get_reward_function(reward_name)
        reward = reward_fn(A, ml, reward_config)
        return (reward, hierarchy_info)
    except Exception as e:
        # Return penalty if solver fails
        return (-5.0, {'error': str(e), 'num_levels': 0, 'dofs': [], 'coarsest_dofs': 0})


def compute_rewards_parallel(
    tasks: list,
    reward_name: str,
    reward_config: dict,
    pyamg_config: dict,
    n_workers: int = 4,
    verbose: bool = False,
) -> Tuple[list, list]:
    """
    Compute rewards for multiple (A, C, B) tuples in parallel.
    
    Args:
        tasks: List of (A, C, B) tuples
        reward_name: Name of reward function
        reward_config: Reward configuration dict
        pyamg_config: PyAMG configuration dict
        n_workers: Number of parallel workers
        verbose: If True, print hierarchy info
        
    Returns:
        Tuple of (rewards list, hierarchy_infos list)
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    # Prepare picklable arguments
    worker_args = []
    for A, C, B in tasks:
        # Convert sparse matrices to picklable format
        A_data = (A.data.copy(), A.indices.copy(), A.indptr.copy(), A.shape)
        C_data = (C.data.copy(), C.indices.copy(), C.indptr.copy(), C.shape)
        worker_args.append((A_data, C_data, B, reward_name, reward_config, pyamg_config))
    
    # Run in parallel
    rewards = [None] * len(tasks)
    hierarchy_infos = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(_compute_reward_worker, args): i 
            for i, args in enumerate(worker_args)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                reward, info = future.result()
                rewards[idx] = reward
                hierarchy_infos[idx] = info
            except Exception as e:
                rewards[idx] = -5.0
                hierarchy_infos[idx] = {'error': str(e)}
    
    # Print summary if verbose
    if verbose and hierarchy_infos:
        valid_infos = [h for h in hierarchy_infos if h and 'dofs' in h and h['dofs']]
        if valid_infos:
            avg_levels = np.mean([h['num_levels'] for h in valid_infos])
            avg_coarsest = np.mean([h['coarsest_dofs'] for h in valid_infos])
            max_coarsest = max(h['coarsest_dofs'] for h in valid_infos)
            print(f"  Batch hierarchy stats: avg_levels={avg_levels:.1f}, "
                  f"avg_coarsest={avg_coarsest:.0f}, max_coarsest={max_coarsest}")
            if max_coarsest > 500:
                print(f"  ⚠️  Large coarse grids detected (max={max_coarsest}) - may cause slowdown")
                
    return rewards, hierarchy_infos


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
