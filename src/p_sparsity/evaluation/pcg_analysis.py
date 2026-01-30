"""PCG convergence analysis for AMG preconditioners."""

import numpy as np
import scipy.sparse.linalg as spla
from dataclasses import dataclass
from typing import List, Optional


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
    tol: float = 1e-8,
    verbose: bool = False
) -> PCGResult:
    """
    Run PCG with AMG preconditioner and analyze convergence.
    
    Args:
        A: System matrix (scipy.sparse)
        ml_solver: PyAMG multilevel solver
        b: Right-hand side vector (if None, use random)
        x0: Initial guess (if None, use zeros)
        max_iter: Maximum PCG iterations
        tol: Convergence tolerance
        verbose: Print iteration details
        
    Returns:
        PCGResult with convergence history
    """
    n = A.shape[0]
    
    # Setup problem
    if b is None:
        b = np.random.randn(n)
    if x0 is None:
        x0 = np.zeros(n)
    
    # Storage for residuals
    residuals = []
    
    def callback(xk):
        """Store residual at each iteration."""
        r = b - A @ xk
        res_norm = np.linalg.norm(r)
        residuals.append(res_norm)
        if verbose:
            # print(f"  PCG iter {len(residuals)}: residual = {res_norm:.2e}")
            pass
    
    # Run PCG with AMG preconditioner
    M = ml_solver.aspreconditioner()
    
    x, info = spla.cg(
        A, b, x0=x0,
        M=M,
        maxiter=max_iter,
        tol=tol,
        callback=callback
    )
    
    # Compute convergence metrics
    converged = (info == 0)
    final_res = residuals[-1] if residuals else np.inf
    
    # Average reduction rate: ||r_k|| / ||r_{k-1}||
    if len(residuals) > 1:
        reduction_factors = [residuals[i] / residuals[i-1] 
                           for i in range(1, len(residuals))]
        avg_reduction = np.mean(reduction_factors)
    else:
        avg_reduction = 1.0
    
    return PCGResult(
        residuals=residuals,
        iterations=len(residuals),
        converged=converged,
        final_residual=final_res,
        reduction_rate=avg_reduction
    )


def compare_pcg_performance(
    A,
    ml_learned,
    ml_baseline,
    num_trials: int = 5,
    **pcg_kwargs
) -> dict:
    """
    Compare PCG performance between learned and baseline AMG.
    
    Args:
        A: System matrix
        ml_learned: Learned AMG solver
        ml_baseline: Baseline AMG solver
        num_trials: Number of random RHS to test
        **pcg_kwargs: Additional arguments for run_pcg_analysis
        
    Returns:
        dict with comparison statistics
    """
    learned_results = []
    baseline_results = []
    
    for trial in range(num_trials):
        b = np.random.randn(A.shape[0])
        
        # Test learned solver
        res_learned = run_pcg_analysis(A, ml_learned, b=b, **pcg_kwargs)
        learned_results.append(res_learned)
        
        # Test baseline solver
        res_baseline = run_pcg_analysis(A, ml_baseline, b=b, **pcg_kwargs)
        baseline_results.append(res_baseline)
    
    # Aggregate statistics
    comparison = {
        'learned': {
            'avg_iterations': np.mean([r.iterations for r in learned_results]),
            'avg_reduction_rate': np.mean([r.reduction_rate for r in learned_results]),
            'success_rate': np.mean([r.converged for r in learned_results]),
        },
        'baseline': {
            'avg_iterations': np.mean([r.iterations for r in baseline_results]),
            'avg_reduction_rate': np.mean([r.reduction_rate for r in baseline_results]),
            'success_rate': np.mean([r.converged for r in baseline_results]),
        }
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
