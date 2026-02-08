"""Sparsity pattern visualization."""

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from matplotlib.figure import Figure
from typing import Optional, Tuple, Dict


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


def plot_sparse_density_heatmap(
    M,
    title: str = "Sparsity Density",
    figsize: Tuple[int, int] = (8, 8),
    bins: int = 512,
    log_scale: bool = True,
    crop: Optional[Tuple[int, int]] = None,
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """
    Heatmap-style visualization of sparsity: bin (row,col) locations of nonzeros.
    
    This is much better than spy() for large matrices as it doesn't densify.
    
    Args:
        M: Sparse matrix
        title: Plot title
        figsize: Figure size
        bins: Resolution of the heatmap in each dimension
        log_scale: Use log1p(counts) to reveal structure
        crop: If not None, restrict to top-left (max_rows, max_cols) region
        ax: Optional matplotlib axes
        
    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if not sp.isspmatrix(M):
        M = sp.csr_matrix(M)

    # Convert to COO for easy access to coordinates
    Mc = M.tocoo(copy=False)
    r = Mc.row.astype(np.int64, copy=False)
    c = Mc.col.astype(np.int64, copy=False)
    nrows, ncols = Mc.shape

    if crop is not None:
        max_r, max_c = crop
        max_r = min(int(max_r), nrows)
        max_c = min(int(max_c), ncols)
        keep = (r < max_r) & (c < max_c)
        r = r[keep]
        c = c[keep]
        nrows, ncols = max_r, max_c

    if r.size == 0:
        ax.set_title(f"{title}\n(empty)")
        return fig

    # Map coordinates -> bin indices
    br = np.minimum((r * bins) // max(nrows, 1), bins - 1)
    bc = np.minimum((c * bins) // max(ncols, 1), bins - 1)

    # Accumulate counts into a bins x bins grid
    H = np.zeros((bins, bins), dtype=np.int32)
    np.add.at(H, (br, bc), 1)

    if log_scale:
        H_plot = np.log1p(H.astype(np.float64))
        scale_note = "log1p(count)"
    else:
        H_plot = H.astype(np.float64)
        scale_note = "count"

    im = ax.imshow(H_plot, origin="upper", aspect="auto", cmap='viridis')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=scale_note)
    ax.set_title(f"{title}\nshape={M.shape}, nnz={M.nnz}, bins={bins}")
    ax.set_xlabel("column (binned)")
    ax.set_ylabel("row (binned)")
    
    plt.tight_layout()
    return fig


def plot_C_overlay_heatmap(
    C_baseline,
    C_learned,
    title: str = "Strength Matrix Overlay",
    figsize: Tuple[int, int] = (8, 8),
    bins: int = 512,
) -> Figure:
    """
    Overlay density heatmap: C_baseline=Red, C_learned=Blue, overlap=Purple.
    
    Args:
        C_baseline: Baseline strength matrix
        C_learned: Learned strength matrix
        title: Plot title
        figsize: Figure size
        bins: Resolution of the heatmap
        
    Returns:
        matplotlib Figure
    """
    if not sp.isspmatrix(C_baseline):
        C_baseline = sp.csr_matrix(C_baseline)
    if not sp.isspmatrix(C_learned):
        C_learned = sp.csr_matrix(C_learned)

    def get_hist(M, shape, bins):
        Mc = M.tocoo()
        r = Mc.row
        c = Mc.col
        nrows, ncols = shape
        br = np.minimum((r * bins) // max(nrows, 1), bins - 1)
        bc = np.minimum((c * bins) // max(ncols, 1), bins - 1)
        H = np.zeros((bins, bins), dtype=np.float32)
        np.add.at(H, (br, bc), 1)
        return np.log1p(H)

    # Use max shape
    nrows = max(C_baseline.shape[0], C_learned.shape[0])
    ncols = max(C_baseline.shape[1], C_learned.shape[1])
    
    H1 = get_hist(C_baseline, (nrows, ncols), bins)
    H2 = get_hist(C_learned, (nrows, ncols), bins)

    # Normalize independently for visibility
    H1_norm = H1 / (H1.max() + 1e-9)
    H2_norm = H2 / (H2.max() + 1e-9)

    # RGB: R = Baseline (Red), B = Learned (Blue)
    img = np.zeros((bins, bins, 3), dtype=np.float32)
    img[..., 0] = H1_norm  # Red channel
    img[..., 2] = H2_norm  # Blue channel

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, origin="upper", interpolation="nearest")
    ax.set_title(f"{title}\nRed=Baseline, Blue=Learned, Purple=Both")
    ax.set_xlabel("column (binned)")
    ax.set_ylabel("row (binned)")
    plt.tight_layout()
    return fig


def plot_P_comparison(
    P_baseline,
    P_learned,
    title: str = "Prolongation Operator Comparison",
    figsize: Tuple[int, int] = (14, 6),
    bins: int = 512,
    crop: Optional[Tuple[int, int]] = None,
) -> Figure:
    """
    Side-by-side comparison of prolongation operators P.
    
    Args:
        P_baseline: Baseline prolongation matrix
        P_learned: Learned prolongation matrix
        title: Plot title
        figsize: Figure size
        bins: Resolution for heatmaps
        crop: Optional (max_rows, max_cols) crop
        
    Returns:
        matplotlib Figure with 2 subplots
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    plot_sparse_density_heatmap(
        P_baseline,
        title="P (Baseline AMG)",
        bins=bins,
        crop=crop,
        ax=axes[0]
    )
    
    plot_sparse_density_heatmap(
        P_learned,
        title="P (Learned AMG)",
        bins=bins,
        crop=crop,
        ax=axes[1]
    )
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def compute_C_overlap_stats(C_baseline, C_learned) -> Dict[str, float]:
    """
    Compare two binary strength matrices and compute overlap statistics.
    
    Args:
        C_baseline: Baseline strength matrix
        C_learned: Learned strength matrix
        
    Returns:
        dict with IoU, intersection counts, etc.
    """
    # Handle sparse matrices properly
    is_sparse = sp.isspmatrix(C_baseline) or sp.isspmatrix(C_learned)
    
    if is_sparse:
        # Convert to CSR and binarize
        c1 = sp.csr_matrix(C_baseline)
        c2 = sp.csr_matrix(C_learned)
        c1.data = (np.abs(c1.data) > 1e-9).astype(np.float64)
        c2.data = (np.abs(c2.data) > 1e-9).astype(np.float64)
        c1.eliminate_zeros()
        c2.eliminate_zeros()
        
        # Intersection via element-wise multiply
        inter = c1.multiply(c2)
        inter.eliminate_zeros()
        nnz_inter = inter.nnz
        nnz_c1 = c1.nnz
        nnz_c2 = c2.nnz
    else:
        # Dense arrays
        c1 = (np.abs(C_baseline) > 1e-9).astype(int)
        c2 = (np.abs(C_learned) > 1e-9).astype(int)
        inter = c1 * c2
        nnz_inter = np.count_nonzero(inter)
        nnz_c1 = np.count_nonzero(c1)
        nnz_c2 = np.count_nonzero(c2)
    
    # Union = c1 + c2 - inter
    nnz_union = nnz_c1 + nnz_c2 - nnz_inter
    iou = nnz_inter / max(nnz_union, 1)
    
    return {
        "iou": iou,
        "n_baseline": nnz_c1,
        "n_learned": nnz_c2,
        "n_intersection": nnz_inter,
        "pct_baseline_in_learned": nnz_inter / max(nnz_c1, 1),
        "pct_learned_in_baseline": nnz_inter / max(nnz_c2, 1),
    }


def plot_iteration_bar(
    iterations: Dict[str, int],
    title: str = "PCG Iterations",
    figsize: Tuple[int, int] = (8, 6),
) -> Figure:
    """
    Bar chart comparing iteration counts.
    
    Args:
        iterations: dict mapping solver names to iteration counts
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    names = list(iterations.keys())
    values = list(iterations.values())
    
    colors = ['#2ecc71' if v == min(values) else '#3498db' for v in values]
    bars = ax.bar(names, values, color=colors, edgecolor='black', alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel("Iterations")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_weight_vs_logit(
    edge_weights: np.ndarray,
    logits: np.ndarray,
    title: str = "Edge Weight vs Model Logit",
    figsize: Tuple[int, int] = (8, 6),
) -> Figure:
    """
    Scatter plot of edge weight vs model logit to check if trivial mapping learned.
    
    Args:
        edge_weights: Normalized edge weights
        logits: Model output logits
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(edge_weights, logits, alpha=0.3, s=5, c='blue')
    ax.set_xlabel("Normalized Edge Weight")
    ax.set_ylabel("Learned Logit")
    ax.set_title(f"{title}\nCorrelation check: Is it just learning weights?")
    ax.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    if len(edge_weights) > 0:
        corr = np.corrcoef(edge_weights.flatten(), logits.flatten())[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig
