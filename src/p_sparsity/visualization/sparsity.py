"""Sparsity pattern visualization."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from typing import Optional, Tuple


def plot_sparsity_pattern(
    matrix,
    title: str = "Sparsity Pattern",
    figsize: Tuple[int, int] = (8, 8),
    markersize: float = 0.5,
    ax: Optional[plt.Axes] = None
) -> Figure:
    """
    Plot sparsity pattern of a sparse matrix.
    
    Args:
        matrix: Sparse matrix (scipy.sparse or dense)
        title: Plot title
        figsize: Figure size
        markersize: Size of markers for nonzeros
        ax: Matplotlib axes (if None, create new figure)
        
    Returns:
        matplotlib Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Convert to COO format for plotting
    if hasattr(matrix, 'tocoo'):
        coo = matrix.tocoo()
        rows, cols = coo.row, coo.col
    else:
        # Dense matrix
        rows, cols = np.nonzero(matrix)
    
    # Plot
    ax.plot(cols, rows, 'k.', markersize=markersize)
    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_ylim(-0.5, matrix.shape[0] - 0.5)
    ax.invert_yaxis()  # Matrix convention: (0,0) at top-left
    ax.set_aspect('equal')
    ax.set_title(f"{title}\n({len(rows)} nonzeros, {len(rows)/matrix.nnz*100:.1f}% sparsity)")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    
    plt.tight_layout()
    return fig


def plot_C_comparison(
    A,
    C_learned,
    C_baseline,
    figsize: Tuple[int, int] = (15, 5),
    sample_size: Optional[int] = None
) -> Figure:
    """
    Compare learned C matrix with baseline C matrix.
    
    Args:
        A: System matrix
        C_learned: Learned strength matrix
        C_baseline: Baseline strength matrix (e.g., from standard AMG)
        figsize: Figure size
        sample_size: If not None, only plot this many rows/cols (for large matrices)
        
    Returns:
        matplotlib Figure with 3 subplots
    """
    # Sample if needed
    if sample_size is not None and A.shape[0] > sample_size:
        idx = np.sort(np.random.choice(A.shape[0], sample_size, replace=False))
        A = A[idx, :][:, idx]
        C_learned = C_learned[idx, :][:, idx]
        C_baseline = C_baseline[idx, :][:, idx]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot A
    plot_sparsity_pattern(A, title="System Matrix A", ax=axes[0])
    
    # Plot learned C
    plot_sparsity_pattern(C_learned, title="Learned C", ax=axes[1])
    
    # Plot baseline C
    plot_sparsity_pattern(C_baseline, title="Baseline C (Standard AMG)", ax=axes[2])
    
    plt.tight_layout()
    return fig


def plot_sparsity_histogram(
    matrices: dict,
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Plot histogram of nonzeros per row for multiple matrices.
    
    Args:
        matrices: dict mapping names to sparse matrices
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, matrix in matrices.items():
        # Count nonzeros per row
        if hasattr(matrix, 'tocsr'):
            csr = matrix.tocsr()
            nnz_per_row = np.diff(csr.indptr)
        else:
            nnz_per_row = np.count_nonzero(matrix, axis=1)
        
        # Plot histogram
        ax.hist(nnz_per_row, bins=30, alpha=0.6, label=name, density=True)
    
    ax.set_xlabel("Nonzeros per row")
    ax.set_ylabel("Density")
    ax.set_title("Sparsity Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_strength_connections(
    A,
    C,
    row_idx: int,
    figsize: Tuple[int, int] = (12, 5)
) -> Figure:
    """
    Visualize which connections are marked as "strong" for a specific row.
    
    Args:
        A: System matrix
        C: Strength matrix
        row_idx: Which row to visualize
        figsize: Figure size
        
    Returns:
        matplotlib Figure with 2 subplots
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Extract row from A and C
    if hasattr(A, 'tocsr'):
        A_row = A.tocsr()[row_idx, :].toarray().flatten()
        C_row = C.tocsr()[row_idx, :].toarray().flatten()
    else:
        A_row = A[row_idx, :]
        C_row = C[row_idx, :]
    
    # Find nonzero indices
    A_nonzero = np.nonzero(A_row)[0]
    C_nonzero = np.nonzero(C_row)[0]
    
    # Plot 1: A values
    axes[0].stem(A_nonzero, A_row[A_nonzero], label='A entries')
    axes[0].stem(C_nonzero, A_row[C_nonzero], linefmt='r-', 
                markerfmt='ro', label='Strong connections (C)')
    axes[0].set_title(f"Row {row_idx}: Matrix Entries")
    axes[0].set_xlabel("Column index")
    axes[0].set_ylabel("Value")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Strength indicator
    strength_indicator = np.zeros_like(A_row)
    strength_indicator[C_nonzero] = 1
    axes[1].stem(A_nonzero, strength_indicator[A_nonzero])
    axes[1].set_title(f"Row {row_idx}: Strength Indicator")
    axes[1].set_xlabel("Column index")
    axes[1].set_ylabel("Is Strong Connection")
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
