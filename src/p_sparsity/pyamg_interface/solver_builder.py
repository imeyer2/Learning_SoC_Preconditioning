"""
PyAMG solver building utilities.

Constructs C matrices from selected edges and builds PyAMG solvers.
"""

from typing import Optional, Tuple
import numpy as np
import scipy.sparse as sp
import torch
import pyamg

from .sampling import sample_deterministic_topk


def C_from_selected_edges(
    A: sp.csr_matrix,
    edge_index_cpu: torch.Tensor,
    selected_mask: np.ndarray,
) -> sp.csr_matrix:
    """
    Build symmetric binary strength matrix C from selected edges.
    
    Args:
        A: Sparse matrix
        edge_index_cpu: (2, E) edge connectivity (CPU tensor)
        selected_mask: (E,) bool array of selected edges
        
    Returns:
        C: Symmetric binary CSR matrix with diagonal ones
    """
    r = edge_index_cpu[0].cpu().numpy()
    c = edge_index_cpu[1].cpu().numpy()
    
    sel = selected_mask.astype(bool)
    # Exclude diagonal edges
    keep = sel & (r != c)
    
    rr = r[keep]
    cc = c[keep]
    vals = np.ones(rr.shape[0], dtype=np.float64)
    
    n = A.shape[0]
    C = sp.csr_matrix((vals, (rr, cc)), shape=(n, n))
    
    # Make symmetric
    C = C.maximum(C.T)
    
    # Add diagonal
    C = C + sp.eye(n, format="csr")
    
    return C


def build_B_for_pyamg(B_extra: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    """
    Build near-nullspace candidates for PyAMG from learned B vectors.
    
    Always includes constant vector; optionally adds learned columns.
    
    Args:
        B_extra: (N, B_extra) learned B candidates (optional)
        
    Returns:
        B: (N, 1+B_extra) numpy array with constant + learned vectors
    """
    if B_extra is None:
        return None
    
    Bx = B_extra.detach().cpu().numpy().astype(np.float64)
    n, p = Bx.shape
    
    # Normalize columns
    for j in range(p):
        col = Bx[:, j]
        col = col - col.mean()
        denom = np.linalg.norm(col) + 1e-12
        Bx[:, j] = col / denom
    
    # Prepend constant vector
    ones = np.ones((n, 1), dtype=np.float64)
    B = np.hstack([ones, Bx])
    
    return B


def build_pyamg_solver(
    A: sp.csr_matrix,
    C: sp.csr_matrix,
    B: Optional[np.ndarray] = None,
    coarse_solver: str = "splu",
) -> pyamg.multilevel.MultilevelSolver:
    """
    Build PyAMG Smoothed Aggregation solver using predefined C matrix.
    
    Args:
        A: System matrix
        C: Strength-of-connection matrix (predefined)
        B: Near-nullspace candidates (optional)
        coarse_solver: Coarse grid solver (splu, lu, cg, etc.)
        
    Returns:
        ml: PyAMG multilevel solver
    """
    strength = ("predefined", {"C": C})
    
    if B is None:
        ml = pyamg.smoothed_aggregation_solver(
            A,
            strength=strength,
            coarse_solver=coarse_solver
        )
    else:
        ml = pyamg.smoothed_aggregation_solver(
            A,
            strength=strength,
            B=B,
            coarse_solver=coarse_solver
        )
    
    return ml


def build_C_from_model(
    A: sp.csr_matrix,
    grid_n: int,
    model: torch.nn.Module,
    k_per_row: int,
    device: str = "cpu",
) -> Tuple[sp.csr_matrix, Optional[np.ndarray]]:
    """
    Build C matrix and B candidates from trained model.
    
    Runs model inference and deterministically selects top-k edges per row.
    
    Args:
        A: Sparse matrix
        grid_n: Grid size
        model: Trained AMG policy
        k_per_row: Number of edges to select per row
        device: Device for inference
        
    Returns:
        C: Strength matrix
        B: Near-nullspace candidates (or None)
    """
    from ..data import node_features_for_policy
    from torch_geometric.utils import from_scipy_sparse_matrix
    
    # Build node coordinates
    n = A.shape[0]
    grid_n_actual = int(np.sqrt(n))
    coords = np.zeros((n, 2), dtype=np.float64)
    coords[:, 0] = np.tile(np.linspace(0, 1, grid_n_actual), grid_n_actual)
    coords[:, 1] = np.repeat(np.linspace(0, 1, grid_n_actual), grid_n_actual)
    
    # Build features
    feature_config = {
        "use_relaxed_vectors": True,
        "use_coordinates": True,
        "num_vecs": 4,
        "relax_iters": 5,
        "relaxation_scheme": "jacobi",
        "omega": 2.0 / 3.0,
    }
    x = node_features_for_policy(A, coords, feature_config).to(device)
    
    # Build graph
    ei, ew = from_scipy_sparse_matrix(A)
    ew = ew.float()
    max_w = ew.abs().max()
    if max_w > 0:
        ew = ew / max_w
    
    ei = ei.to(device)
    ew = ew.to(device)
    
    # Run model
    model.eval()
    with torch.no_grad():
        logits, B_extra = model(x, ei, ew)
    
    # Deterministic top-k selection
    from .sampling import build_row_groups
    row_groups = build_row_groups(ei.cpu(), num_nodes=A.shape[0])
    selected = sample_deterministic_topk(logits, row_groups, k_per_row)
    
    # Build C matrix
    C = C_from_selected_edges(A, ei.cpu(), selected.cpu().numpy())
    
    # Build B candidates
    B = build_B_for_pyamg(B_extra)
    
    return C, B
