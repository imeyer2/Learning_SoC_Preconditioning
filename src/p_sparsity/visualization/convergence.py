"""Convergence visualization."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from typing import List, Optional, Tuple


def plot_convergence_curves(
    residuals_dict: dict,
    figsize: Tuple[int, int] = (10, 6),
    semilogy: bool = True,
    title: str = "Convergence Comparison"
) -> Figure:
    """
    Plot convergence curves for multiple solvers.
    
    Args:
        residuals_dict: dict mapping solver names to lists of residuals
        figsize: Figure size
        semilogy: Use log scale for y-axis
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, residuals in residuals_dict.items():
        iterations = range(len(residuals))
        if semilogy:
            ax.semilogy(iterations, residuals, marker='o', label=name, linewidth=2)
        else:
            ax.plot(iterations, residuals, marker='o', label=name, linewidth=2)
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual Norm")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    return fig


def plot_reduction_factors(
    reduction_dict: dict,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Error Reduction Factors"
) -> Figure:
    """
    Plot error reduction factors per iteration.
    
    Args:
        reduction_dict: dict mapping solver names to lists of reduction factors
        figsize: Figure size
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, reductions in reduction_dict.items():
        iterations = range(1, len(reductions) + 1)  # Start from 1
        ax.plot(iterations, reductions, marker='s', label=name, linewidth=2)
    
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='No reduction')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reduction Factor")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_vcycle_comparison(
    vcycle_results_dict: dict,
    figsize: Tuple[int, int] = (12, 5)
) -> Figure:
    """
    Compare V-cycle performance across multiple solvers.
    
    Args:
        vcycle_results_dict: dict mapping solver names to VCycleResult objects
        figsize: Figure size
        
    Returns:
        matplotlib Figure with 2 subplots
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Error norms
    for name, result in vcycle_results_dict.items():
        cycles = range(len(result.error_norms))
        axes[0].semilogy(cycles, result.error_norms, marker='o', 
                        label=f"{name} (avg={result.avg_reduction_factor:.3f})",
                        linewidth=2)
    
    axes[0].set_xlabel("V-cycle")
    axes[0].set_ylabel("Error Norm")
    axes[0].set_title("Error Convergence")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, which='both')
    
    # Plot 2: Energy norms
    for name, result in vcycle_results_dict.items():
        cycles = range(len(result.energy_norms))
        axes[1].semilogy(cycles, result.energy_norms, marker='s',
                        label=name, linewidth=2)
    
    axes[1].set_xlabel("V-cycle")
    axes[1].set_ylabel("Energy Norm")
    axes[1].set_title("Energy Norm Convergence")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    return fig


def plot_eigenvalue_spectrum(
    eigenvalues_dict: dict,
    figsize: Tuple[int, int] = (12, 5),
    num_bins: int = 30
) -> Figure:
    """
    Plot eigenvalue spectra for preconditioned systems.
    
    Args:
        eigenvalues_dict: dict mapping solver names to eigenvalue arrays
        figsize: Figure size
        num_bins: Number of histogram bins
        
    Returns:
        matplotlib Figure with 2 subplots
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Eigenvalue scatter
    for name, eigs in eigenvalues_dict.items():
        axes[0].scatter(np.real(eigs), np.imag(eigs), alpha=0.6, 
                       label=name, s=20)
    
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_xlabel("Real part")
    axes[0].set_ylabel("Imaginary part")
    axes[0].set_title("Eigenvalue Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Eigenvalue histogram (real part)
    for name, eigs in eigenvalues_dict.items():
        axes[1].hist(np.real(eigs), bins=num_bins, alpha=0.5, 
                    label=name, density=True)
    
    axes[1].set_xlabel("Eigenvalue (real part)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Eigenvalue Histogram")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_condition_number_comparison(
    condition_numbers: dict,
    figsize: Tuple[int, int] = (8, 6)
) -> Figure:
    """
    Bar plot comparing condition numbers.
    
    Args:
        condition_numbers: dict mapping solver names to condition numbers
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    names = list(condition_numbers.keys())
    values = list(condition_numbers.values())
    
    bars = ax.bar(names, values, alpha=0.7, edgecolor='black')
    
    # Color bars: lower is better
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(values)))
    sorted_idx = np.argsort(values)
    for i, bar in enumerate(bars):
        bar.set_color(colors[np.where(sorted_idx == i)[0][0]])
    
    ax.set_ylabel("Condition Number")
    ax.set_title("Condition Number Comparison (lower is better)")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1e}',
               ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_eigenvalue_spectra_comparison(
    eigenvalues_baseline: np.ndarray,
    eigenvalues_learned: np.ndarray,
    title: str = "Eigenvalue Spectra of $M^{-1}A$",
    figsize: Tuple[int, int] = (14, 5),
) -> Figure:
    """
    Plot eigenvalue spectra comparison like the reference script.
    
    Creates two subplots:
    1. Sorted eigenvalues with condition numbers
    2. Eigenvalue density histogram
    
    Args:
        eigenvalues_baseline: Eigenvalues for baseline preconditioner
        eigenvalues_learned: Eigenvalues for learned preconditioner
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure with 2 subplots
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Get real parts and sort
    ev_baseline = np.sort(np.real(eigenvalues_baseline))
    ev_learned = np.sort(np.real(eigenvalues_learned))
    
    # Compute condition numbers (filter near-zeros for null space)
    def compute_condition(eigs):
        pos_eigs = eigs[eigs > 1e-6]
        if len(pos_eigs) == 0:
            return np.inf
        return pos_eigs.max() / pos_eigs.min()
    
    cond_baseline = compute_condition(ev_baseline)
    cond_learned = compute_condition(ev_learned)
    
    # Plot 1: Sorted eigenvalues
    axes[0].plot(ev_baseline, label=f"Baseline ($\\kappa$={cond_baseline:.1f})", 
                 marker='.', linestyle='--', alpha=0.6, markersize=3)
    axes[0].plot(ev_learned, label=f"Learned ($\\kappa$={cond_learned:.1f})", 
                 marker='x', linestyle='-', alpha=0.6, markersize=3)
    axes[0].axhline(1.0, color='k', linestyle=':', alpha=0.3)
    axes[0].set_title(f"{title}")
    axes[0].set_ylabel("Eigenvalue $\\lambda$")
    axes[0].set_xlabel("Index")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Eigenvalue density histogram
    axes[1].hist(ev_baseline, bins=50, alpha=0.5, label="Baseline", density=True, color='red')
    axes[1].hist(ev_learned, bins=50, alpha=0.5, label="Learned", density=True, color='blue')
    axes[1].axvline(1.0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_title("Eigenvalue Density Histogram")
    axes[1].set_xlabel("Eigenvalue $\\lambda$")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_pcg_convergence_comparison(
    residuals_baseline: List[float],
    residuals_learned: List[float],
    residuals_tuned: Optional[List[float]] = None,
    title: str = "PCG Convergence Comparison",
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """
    Plot PCG residual convergence curves like the reference script.
    
    Args:
        residuals_baseline: Residual history for baseline (theta=0.0)
        residuals_learned: Residual history for learned preconditioner
        residuals_tuned: Optional residual history for tuned (theta=0.25)
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.semilogy(range(len(residuals_baseline)), residuals_baseline, 
                label=f"PCG + Baseline (θ=0.0) [{len(residuals_baseline)} iters]",
                linestyle='--', alpha=0.8, linewidth=2)
    
    if residuals_tuned is not None:
        ax.semilogy(range(len(residuals_tuned)), residuals_tuned,
                    label=f"PCG + Tuned (θ=0.25) [{len(residuals_tuned)} iters]",
                    linestyle='-.', alpha=0.8, linewidth=2)
    
    ax.semilogy(range(len(residuals_learned)), residuals_learned,
                label=f"PCG + Learned [{len(residuals_learned)} iters]",
                linestyle='-', alpha=0.9, linewidth=2.5, color='green')
    
    ax.set_xlabel("PCG Iteration")
    ax.set_ylabel("Residual Norm")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    return fig
