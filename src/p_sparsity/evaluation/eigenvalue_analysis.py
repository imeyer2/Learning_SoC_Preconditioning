"""Eigenvalue and spectral analysis for AMG preconditioners."""

import numpy as np
import scipy.sparse.linalg as spla
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class EigenvalueResult:
    """Results from eigenvalue analysis."""
    eigenvalues: np.ndarray
    condition_number: float
    spectral_radius: float
    min_eigenvalue: float
    max_eigenvalue: float
    
    def __repr__(self):
        return (f"EigenvalueResult(cond={self.condition_number:.2e}, "
                f"rho={self.spectral_radius:.4f})")


def estimate_spectral_properties_from_vcycle(
    A,
    ml_solver,
    num_vecs: int = 10,
    max_iters: int = 20
) -> EigenvalueResult:
    """
    Fast spectral property estimation from V-cycle convergence behavior.
    Cheaper than computing eigenvalues - uses residual ratios instead.
    
    Args:
        A: System matrix
        ml_solver: PyAMG solver
        num_vecs: Number of random test vectors
        max_iters: Max V-cycle iterations per vector
        
    Returns:
        EigenvalueResult with estimated spectral properties
    """
    n = A.shape[0]
    convergence_factors = []
    
    for _ in range(num_vecs):
        # Random initial residual
        x0 = np.random.randn(n)
        b = A @ x0
        x = np.zeros(n)
        
        residuals = []
        for _ in range(max_iters):
            r = b - A @ x
            res_norm = np.linalg.norm(r)
            residuals.append(res_norm)
            
            if res_norm < 1e-12:
                break
                
            # One V-cycle
            x += ml_solver.solve(r, maxiter=1, tol=1e-16)
        
        # Estimate convergence factor from geometric mean
        if len(residuals) > 2:
            ratios = np.array(residuals[1:]) / np.array(residuals[:-1])
            ratios = ratios[ratios > 0]  # Filter numerical noise
            if len(ratios) > 0:
                # Geometric mean is more stable than arithmetic mean
                conv_factor = np.exp(np.mean(np.log(ratios)))
                convergence_factors.append(conv_factor)
    
    if len(convergence_factors) > 0:
        # Average convergence factor (spectral radius of error propagator)
        avg_conv_factor = np.mean(convergence_factors)
        spectral_radius = avg_conv_factor
        
        # Use convergence factor as the main metric
        # Lower is better: rho=0 is instant convergence, rho=1 is no convergence
        max_conv = np.max(convergence_factors)
        min_conv = np.min(convergence_factors)
        
        # Asymptotic convergence rate: -log(rho)
        # Higher is better: larger values mean faster convergence
        convergence_rate = -np.log(avg_conv_factor) if avg_conv_factor > 0 else 10.0
        
        return EigenvalueResult(
            eigenvalues=np.array([min_conv, avg_conv_factor, max_conv]),
            min_eigenvalue=min_conv,
            max_eigenvalue=max_conv,
            condition_number=convergence_rate,  # Store convergence rate in condition_number field
            spectral_radius=spectral_radius
        )
    else:
        # Fallback
        return EigenvalueResult(
            eigenvalues=np.array([0.9]),
            min_eigenvalue=0.9,
            max_eigenvalue=0.9,
            condition_number=10.0,
            spectral_radius=0.9
        )


def run_eigenvalue_analysis(
    A,
    ml_solver,
    k: int = 20,
    which: str = 'LM',
    compute_condition: bool = True
) -> EigenvalueResult:
    """
    Compute eigenvalues of preconditioned system M^{-1}A.
    WARNING: This is expensive for large matrices. Consider using
    estimate_spectral_properties_from_vcycle() instead.
    
    Args:
        A: System matrix
        ml_solver: PyAMG solver (used as preconditioner)
        k: Number of eigenvalues to compute
        which: Which eigenvalues ('LM'=largest magnitude, 'SM'=smallest, 'LR'=largest real)
        compute_condition: Whether to compute condition number (requires full spectrum)
        
    Returns:
        EigenvalueResult with spectral properties
    """
    n = A.shape[0]
    M = ml_solver.aspreconditioner()
    
    # Define linear operator for M^{-1}A
    def matvec(v):
        return M @ (A @ v)
    
    MA_op = spla.LinearOperator((n, n), matvec=matvec)
    
    # Compute eigenvalues
    if k < n - 1:
        try:
            eigenvalues, _ = spla.eigs(MA_op, k=k, which=which)
        except:
            # Fallback: compute fewer eigenvalues
            eigenvalues, _ = spla.eigs(MA_op, k=min(k//2, n-2), which=which)
    else:
        # For small problems, use full eigendecomposition
        MA_dense = MA_op @ np.eye(n)
        eigenvalues = np.linalg.eigvals(MA_dense)
    
    # Real parts (preconditioned system should be positive definite)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 0]  # Filter out numerical noise
    
    # Compute metrics
    if len(eigenvalues) > 0:
        min_eig = np.min(eigenvalues)
        max_eig = np.max(eigenvalues)
        spectral_radius = max_eig
        
        if compute_condition:
            condition_number = max_eig / min_eig if min_eig > 1e-14 else np.inf
        else:
            condition_number = np.nan
    else:
        min_eig = max_eig = spectral_radius = condition_number = np.nan
    
    return EigenvalueResult(
        eigenvalues=eigenvalues,
        condition_number=condition_number,
        spectral_radius=spectral_radius,
        min_eigenvalue=min_eig,
        max_eigenvalue=max_eig
    )


def compute_convergence_factor_bound(
    A,
    ml_solver,
    k: int = 20
) -> Tuple[float, float]:
    """
    Estimate convergence factor bounds from eigenvalues.
    
    For CG: rho ≤ (sqrt(kappa) - 1) / (sqrt(kappa) + 1)
    For Richardson: rho = 1 - 2*lambda_min / (lambda_min + lambda_max)
    
    Args:
        A: System matrix
        ml_solver: PyAMG solver
        k: Number of eigenvalues to compute
        
    Returns:
        (cg_bound, richardson_bound): Theoretical convergence rates
    """
    eig_result = run_eigenvalue_analysis(A, ml_solver, k=k)
    
    kappa = eig_result.condition_number
    if not np.isfinite(kappa):
        return np.nan, np.nan
    
    # CG bound
    sqrt_kappa = np.sqrt(kappa)
    cg_bound = (sqrt_kappa - 1) / (sqrt_kappa + 1)
    
    # Richardson bound
    lmin = eig_result.min_eigenvalue
    lmax = eig_result.max_eigenvalue
    richardson_bound = 1 - 2 * lmin / (lmin + lmax)
    
    return cg_bound, richardson_bound


def compare_spectral_properties(
    A,
    ml_learned,
    ml_baseline,
    k: int = 20,
    method: str = 'fast'
) -> dict:
    """
    Compare spectral properties of learned vs baseline preconditioner.
    
    Args:
        A: System matrix
        ml_learned: Learned AMG solver
        ml_baseline: Baseline AMG solver
        k: Number of eigenvalues to compute (only for method='eigenvalues')
        method: 'fast' (convergence-based, cheap) or 'eigenvalues' (accurate, expensive)
        
    Returns:
        dict with comparison statistics
    """
    print(f"  Spectral analysis method: {method}")
    
    # Choose analysis method
    if method == 'fast':
        # Fast convergence-based estimation (cheap for large problems)
        print("  → Estimating learned solver convergence...", end=" ", flush=True)
        eig_learned = estimate_spectral_properties_from_vcycle(A, ml_learned, num_vecs=10, max_iters=15)
        print("done")
        print("  → Estimating baseline solver convergence...", end=" ", flush=True)
        eig_baseline = estimate_spectral_properties_from_vcycle(A, ml_baseline, num_vecs=10, max_iters=15)
        print("done")
        # Estimate bounds from convergence factors
        cg_bound_learned = eig_learned.spectral_radius
        rich_bound_learned = eig_learned.spectral_radius
        cg_bound_baseline = eig_baseline.spectral_radius
        rich_bound_baseline = eig_baseline.spectral_radius
    else:
        # Full eigenvalue analysis (expensive but accurate)
        print(f"  → Computing {k} eigenvalues for learned solver...", end=" ", flush=True)
        eig_learned = run_eigenvalue_analysis(A, ml_learned, k=k)
        print("done")
        cg_bound_learned, rich_bound_learned = compute_convergence_factor_bound(
            A, ml_learned, k=k
        )
        print(f"  → Computing {k} eigenvalues for baseline solver...", end=" ", flush=True)
        eig_baseline = run_eigenvalue_analysis(A, ml_baseline, k=k)
        print("done")
        cg_bound_baseline, rich_bound_baseline = compute_convergence_factor_bound(
            A, ml_baseline, k=k
        )
    
    # For fast method, condition_number field actually stores convergence rate
    metric_name = 'convergence_rate' if method == 'fast' else 'condition_number'
    
    comparison = {
        'learned': {
            'condition_number': eig_learned.condition_number,
            'spectral_radius': eig_learned.spectral_radius,
            'cg_convergence_bound': cg_bound_learned,
            'richardson_bound': rich_bound_learned,
            'metric_type': metric_name,
            'eigenvalues': eig_learned.eigenvalues,  # Include eigenvalues for plotting
        },
        'baseline': {
            'condition_number': eig_baseline.condition_number,
            'spectral_radius': eig_baseline.spectral_radius,
            'cg_convergence_bound': cg_bound_baseline,
            'richardson_bound': rich_bound_baseline,
            'metric_type': metric_name,
            'eigenvalues': eig_baseline.eigenvalues,  # Include eigenvalues for plotting
        }
    }
    
    # Improvement ratios
    if eig_baseline.condition_number > 0:
        comparison['condition_improvement'] = (
            eig_baseline.condition_number / eig_learned.condition_number
        )
    else:
        comparison['condition_improvement'] = np.nan
    
    return comparison


def analyze_eigenvalue_clustering(eigenvalues: np.ndarray, num_bins: int = 10) -> dict:
    """
    Analyze how eigenvalues are clustered (good clustering => fast CG).
    
    Args:
        eigenvalues: Array of eigenvalues
        num_bins: Number of bins for histogram
        
    Returns:
        dict with clustering metrics
    """
    if len(eigenvalues) < 2:
        return {'entropy': 0.0, 'num_clusters': 0}
    
    # Histogram
    hist, bin_edges = np.histogram(eigenvalues, bins=num_bins)
    
    # Normalized histogram (probability distribution)
    prob = hist / hist.sum()
    prob = prob[prob > 0]  # Remove empty bins
    
    # Entropy: lower entropy => better clustering
    entropy = -np.sum(prob * np.log(prob))
    
    # Count clusters (non-empty bins)
    num_clusters = np.sum(hist > 0)
    
    return {
        'entropy': entropy,
        'num_clusters': num_clusters,
        'histogram': hist,
        'bin_edges': bin_edges,
    }
