"""PCG convergence analysis for AMG preconditioners."""

import numpy as np
import scipy.sparse.linalg as spla
import time
from dataclasses import dataclass
from typing import List, Optional

from tqdm import tqdm


class PCGTimeout(Exception):
    """Raised when PCG exceeds time limit or stagnates."""
    pass


@dataclass
class PCGResult:
    """Results from PCG convergence test."""
    residuals: List[float]
    iterations: int
    converged: bool
    final_residual: float
    reduction_rate: float
    
    def __repr__(self):
        status = "converged" if self.converged else "failed"
        return (f"PCGResult(iterations={self.iterations}, {status}, "
                f"final_residual={self.final_residual:.2e}, "
                f"avg_reduction={self.reduction_rate:.4f})")


def run_pcg_analysis(
    A,
    ml_solver,
    b: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = True,
    show_progress: bool = True,
    timeout: float = 60.0,
    stagnation_window: int = 10
) -> PCGResult:
    """
    Run PCG with AMG preconditioner and analyze convergence.
    
    Uses custom PCG implementation for better interrupt handling.
    
    Args:
        A: System matrix (scipy.sparse)
        ml_solver: PyAMG multilevel solver
        b: Right-hand side vector (if None, use random)
        x0: Initial guess (if None, use zeros)
        max_iter: Maximum PCG iterations
        tol: Convergence tolerance
        verbose: Print iteration details
        show_progress: Show progress indicator for long runs
        timeout: Maximum time in seconds (default 60s)
        stagnation_window: Abort if no progress in this many iterations
        
    Returns:
        PCGResult with convergence history
    """
    n = A.shape[0]
    
    # Setup problem
    if b is None:
        b = np.random.randn(n)
    if x0 is None:
        x0 = np.zeros(n)
    
    # Get preconditioner
    M = ml_solver.aspreconditioner()
    
    # Custom PCG implementation (interruptible)
    x = x0.copy()
    r = b - A @ x
    
    residuals = []
    initial_res_norm = np.linalg.norm(r)
    residuals.append(initial_res_norm)
    
    if initial_res_norm < tol:
        return PCGResult(
            residuals=residuals,
            iterations=0,
            converged=True,
            final_residual=initial_res_norm,
            reduction_rate=0.0
        )
    
    # Apply preconditioner
    z = M @ r
    p = z.copy()
    rz = r @ z
    
    start_time = time.time()
    converged = False
    abort_reason = ""
    
    for iteration in range(max_iter):
        # Check timeout at start of each iteration
        elapsed = time.time() - start_time
        if elapsed > timeout:
            abort_reason = f"timeout ({elapsed:.1f}s > {timeout}s)"
            if show_progress:
                print(f"    ⚠ PCG aborted: {abort_reason}", flush=True)
            break
        
        # Matrix-vector product
        Ap = A @ p
        alpha = rz / (p @ Ap)
        
        x = x + alpha * p
        r = r - alpha * Ap
        
        res_norm = np.linalg.norm(r)
        residuals.append(res_norm)
        
        # Progress output
        if show_progress and (iteration + 1) % 10 == 0:
            print(f"    PCG iter {iteration+1}: residual = {res_norm:.2e} ({elapsed:.1f}s)", flush=True)
        
        # Check convergence
        if res_norm < tol * initial_res_norm:
            converged = True
            if show_progress:
                print(f"    ✓ PCG converged in {iteration+1} iterations ({elapsed:.1f}s)", flush=True)
            break
        
        # Check for stagnation
        if len(residuals) >= stagnation_window:
            recent = residuals[-stagnation_window:]
            if recent[-1] >= recent[0] * 0.99:
                abort_reason = f"stagnation (no progress in {stagnation_window} iters)"
                if show_progress:
                    print(f"    ⚠ PCG aborted: {abort_reason}", flush=True)
                break
        
        # Check for divergence
        if res_norm > initial_res_norm * 100:
            abort_reason = "divergence (residual grew 100x)"
            if show_progress:
                print(f"    ⚠ PCG aborted: {abort_reason}", flush=True)
            break
        
        # Preconditioner application (this is often the slow part)
        z = M @ r
        rz_new = r @ z
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new
    
    # Compute convergence metrics
    final_res = residuals[-1] if residuals else np.inf
    
    # Average reduction rate
    if len(residuals) > 1:
        reduction_factors = [residuals[i] / residuals[i-1] 
                           for i in range(1, len(residuals)) 
                           if residuals[i-1] > 0]
        avg_reduction = np.mean(reduction_factors) if reduction_factors else 1.0
    else:
        avg_reduction = 1.0
    
    return PCGResult(
        residuals=residuals,
        iterations=len(residuals) - 1,  # Don't count initial residual
        converged=converged,
        final_residual=final_res,
        reduction_rate=avg_reduction
    )


def compare_pcg_performance(
    A,
    ml_learned,
    ml_baseline,
    num_trials: int = 5,
    ml_tuned=None,
    **pcg_kwargs
) -> dict:
    """
    Compare PCG performance between learned and baseline AMG.
    
    Args:
        A: System matrix
        ml_learned: Learned AMG solver
        ml_baseline: Baseline AMG solver
        num_trials: Number of random RHS to test
        ml_tuned: Optional tuned AMG solver (e.g., theta=0.25)
        **pcg_kwargs: Additional arguments for run_pcg_analysis
        
    Returns:
        dict with comparison statistics including residual histories
    """
    learned_results = []
    baseline_results = []
    tuned_results = []
    
    # Print PCG settings
    max_iter = pcg_kwargs.get('max_iter', 100)
    tol = pcg_kwargs.get('tol', 1e-8)
    timeout = pcg_kwargs.get('timeout', 30.0)  # Default 30s per solver call
    n = A.shape[0]
    print(f"  PCG settings: n={n}, tol={tol:.0e}, max_iter={max_iter}, timeout={timeout}s, trials={num_trials}")
    
    # Show progress for larger problems
    # show_progress = n > 10000
    show_progress = True
    
    # Set default timeout if not provided, and remove show_progress from kwargs if present
    if 'timeout' not in pcg_kwargs:
        pcg_kwargs['timeout'] = timeout
    pcg_kwargs.pop('show_progress', None)  # Remove to avoid duplicate argument
    
    for trial in tqdm(range(num_trials), desc="  PCG trials", leave=False):
        b = np.random.randn(A.shape[0])
        
        # Test learned solver
        if show_progress:
            print(f"  Trial {trial+1}/{num_trials}: testing learned solver...", flush=True)
        res_learned = run_pcg_analysis(A, ml_learned, b=b, show_progress=show_progress, **pcg_kwargs)
        learned_results.append(res_learned)
        
        # Test baseline solver
        if show_progress:
            print(f"  Trial {trial+1}/{num_trials}: testing baseline solver...", flush=True)
        res_baseline = run_pcg_analysis(A, ml_baseline, b=b, show_progress=show_progress, **pcg_kwargs)
        baseline_results.append(res_baseline)
        
        # Test tuned solver if provided
        if ml_tuned is not None:
            if show_progress:
                print(f"  Trial {trial+1}/{num_trials}: testing tuned solver...", flush=True)
            res_tuned = run_pcg_analysis(A, ml_tuned, b=b, show_progress=show_progress, **pcg_kwargs)
            tuned_results.append(res_tuned)
    
    print(f"  ✓ Completed {num_trials} PCG trials")
    
    # Aggregate statistics
    comparison = {
        'learned': {
            'avg_iterations': np.mean([r.iterations for r in learned_results]),
            'avg_reduction_rate': np.mean([r.reduction_rate for r in learned_results]),
            'success_rate': np.mean([r.converged for r in learned_results]),
            # Include residuals from first trial for plotting
            'residuals': learned_results[0].residuals if learned_results else [],
        },
        'baseline': {
            'avg_iterations': np.mean([r.iterations for r in baseline_results]),
            'avg_reduction_rate': np.mean([r.reduction_rate for r in baseline_results]),
            'success_rate': np.mean([r.converged for r in baseline_results]),
            'residuals': baseline_results[0].residuals if baseline_results else [],
        }
    }
    
    # Add tuned results if available
    if tuned_results:
        comparison['tuned'] = {
            'avg_iterations': np.mean([r.iterations for r in tuned_results]),
            'avg_reduction_rate': np.mean([r.reduction_rate for r in tuned_results]),
            'success_rate': np.mean([r.converged for r in tuned_results]),
            'residuals': tuned_results[0].residuals if tuned_results else [],
        }
    
    # Speedup
    if comparison['baseline']['avg_iterations'] > 0:
        comparison['speedup'] = (
            comparison['baseline']['avg_iterations'] / 
            comparison['learned']['avg_iterations']
        )
    else:
        comparison['speedup'] = 1.0
    
    return comparison

