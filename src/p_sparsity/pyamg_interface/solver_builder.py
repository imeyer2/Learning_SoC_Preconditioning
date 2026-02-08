"""
PyAMG solver building utilities.

Constructs C matrices from selected edges and builds PyAMG solvers.

Three hierarchy strategies are supported:
1. HYBRID (default): GNN for fine level, PyAMG standard for coarser levels
2. FULL_GNN: GNN computes C at every level (slower, requires iterative build)
3. PREDEFINED_ONLY: Only GNN C, PyAMG builds rest (can fail with bad C)
"""

from typing import Optional, Tuple, List, Literal
import numpy as np
import scipy.sparse as sp
import torch
import pyamg

from .sampling import sample_deterministic_topk


# Hierarchy strategy types
HierarchyStrategy = Literal["hybrid", "full_gnn", "predefined_only"]


def print_hierarchy_info(ml: pyamg.multilevel.MultilevelSolver, prefix: str = "") -> dict:
    """
    Print and return AMG hierarchy information.
    
    Args:
        ml: PyAMG multilevel solver
        prefix: Optional prefix for print statements
        
    Returns:
        dict with hierarchy stats
    """
    num_levels = len(ml.levels)
    dofs = [lvl.A.shape[0] for lvl in ml.levels]
    nnzs = [lvl.A.nnz for lvl in ml.levels]
    
    # Coarsening factors
    coarsening_factors = []
    for i in range(num_levels - 1):
        cf = dofs[i] / dofs[i+1] if dofs[i+1] > 0 else float('inf')
        coarsening_factors.append(cf)
    
    # Operator complexity
    total_nnz = sum(nnzs)
    op_complexity = total_nnz / nnzs[0] if nnzs[0] > 0 else float('inf')
    
    # Grid complexity
    total_dofs = sum(dofs)
    grid_complexity = total_dofs / dofs[0] if dofs[0] > 0 else float('inf')
    
    print(f"{prefix}AMG Hierarchy: {num_levels} levels")
    print(f"{prefix}  Level | DOFs      | NNZ        | Coarsening Factor")
    print(f"{prefix}  ------|-----------|------------|------------------")
    for i, (d, n) in enumerate(zip(dofs, nnzs)):
        cf = f"{coarsening_factors[i]:.2f}x" if i < len(coarsening_factors) else "(coarsest)"
        print(f"{prefix}  {i:5d} | {d:9d} | {n:10d} | {cf}")
    print(f"{prefix}  Operator complexity: {op_complexity:.3f}")
    print(f"{prefix}  Grid complexity: {grid_complexity:.3f}")
    print(f"{prefix}  Coarsest DOFs: {dofs[-1]} (direct solve)")
    
    # Warning if coarse grid is large
    if dofs[-1] > 500:
        print(f"{prefix}  ⚠️  WARNING: Coarse grid has {dofs[-1]} DOFs - direct solve may be expensive!")
    
    return {
        'num_levels': num_levels,
        'dofs_per_level': dofs,
        'nnz_per_level': nnzs,
        'coarsening_factors': coarsening_factors,
        'operator_complexity': op_complexity,
        'grid_complexity': grid_complexity,
    }


def get_hierarchy_summary(ml: pyamg.multilevel.MultilevelSolver) -> str:
    """
    Get a one-line summary of the hierarchy.
    
    Returns:
        String like "3 lvls: 1024→256→16 (OC=1.28)"
    """
    dofs = [lvl.A.shape[0] for lvl in ml.levels]
    nnzs = [lvl.A.nnz for lvl in ml.levels]
    op_complexity = sum(nnzs) / nnzs[0] if nnzs[0] > 0 else 0
    dof_str = "→".join(str(d) for d in dofs)
    return f"{len(dofs)} lvls: {dof_str} (OC={op_complexity:.2f})"


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
    max_levels: Optional[int] = None,
    max_coarse: int = 10,
    hierarchy_strategy: HierarchyStrategy = "hybrid",
    fallback_strength: Tuple = ('symmetric', {'theta': 0.0}),
) -> pyamg.multilevel.MultilevelSolver:
    """
    Build PyAMG Smoothed Aggregation solver using predefined C matrix.
    
    Args:
        A: System matrix
        C: Strength-of-connection matrix (predefined by GNN)
        B: Near-nullspace candidates (optional)
        coarse_solver: Coarse grid solver (splu, lu, cg, etc.)
        max_levels: Maximum number of levels in hierarchy (None = unlimited)
        max_coarse: Maximum size of coarsest grid before stopping
        hierarchy_strategy: How to handle multi-level hierarchy:
            - "hybrid": GNN C for level 0, fallback_strength for coarser levels
            - "predefined_only": Use GNN C only (may produce bad hierarchies)
            - "full_gnn": Not supported here (use build_pyamg_solver_full_gnn)
        fallback_strength: Strength method for coarser levels in hybrid mode
        
    Returns:
        ml: PyAMG multilevel solver
    """
    if hierarchy_strategy == "full_gnn":
        raise ValueError(
            "full_gnn strategy requires build_pyamg_solver_full_gnn() "
            "which iteratively builds C at each level"
        )
    
    if hierarchy_strategy == "hybrid":
        # GNN for level 0, standard PyAMG for deeper levels
        strength = [
            ('predefined', {'C': C}),  # Level 0: GNN-generated C
            fallback_strength,          # Level 1+: PyAMG standard  
        ]
    else:  # predefined_only
        strength = ('predefined', {'C': C})
    
    kwargs = {
        "strength": strength,
        "coarse_solver": coarse_solver,
        "max_coarse": max_coarse,
    }
    if max_levels is not None:
        kwargs["max_levels"] = max_levels
    if B is not None:
        kwargs["B"] = B
    
    ml = pyamg.smoothed_aggregation_solver(A, **kwargs)
    
    return ml


def build_pyamg_solver_full_gnn(
    A: sp.csr_matrix,
    model: torch.nn.Module,
    k_per_row: int,
    B: Optional[np.ndarray] = None,
    coarse_solver: str = "splu",
    max_levels: int = 10,
    max_coarse: int = 10,
    device: str = "cpu",
    use_learned_k: bool = False,
) -> pyamg.multilevel.MultilevelSolver:
    """
    Build PyAMG solver with GNN-generated C at EVERY level.
    
    This iteratively:
    1. Runs GNN on current A to get C
    2. Uses C to build aggregates and prolongation
    3. Computes coarse A
    4. Repeats until coarse enough
    
    Args:
        A: Fine-level system matrix
        model: Trained AMG policy (must accept variable-size graphs)
        k_per_row: Number of edges to select per row (used if use_learned_k=False)
        B: Near-nullspace candidates (optional)
        coarse_solver: Coarse grid solver
        max_levels: Maximum hierarchy depth
        max_coarse: Stop when coarse grid has fewer unknowns
        device: Device for GNN inference
        use_learned_k: If True and model predicts k, use learned per-node k values
        
    Returns:
        ml: PyAMG multilevel solver with GNN C at each level
    """
    from ..data import node_features_for_policy
    from torch_geometric.utils import from_scipy_sparse_matrix
    from .sampling import build_row_groups
    
    # Collect C matrices for each level
    C_list = []
    current_A = A.copy()
    
    for level in range(max_levels - 1):
        n = current_A.shape[0]
        if n <= max_coarse:
            break
            
        # Build features for current level
        # Note: coordinates are approximated for coarse levels
        coords = _estimate_coords_for_coarse_level(current_A, level)
        
        feature_config = {
            "use_relaxed_vectors": True,
            "use_coordinates": True,
            "num_vecs": 4,
            "relax_iters": 5,
            "relaxation_scheme": "jacobi",
            "omega": 2.0 / 3.0,
        }
        x = node_features_for_policy(current_A, coords, feature_config).to(device)
        
        # Build graph
        ei, ew = from_scipy_sparse_matrix(current_A)
        ew = ew.float()
        max_w = ew.abs().max()
        if max_w > 0:
            ew = ew / max_w
        ei = ei.to(device)
        ew = ew.to(device)
        
        # Run GNN
        model.eval()
        with torch.no_grad():
            output = model(x, ei, ew)
        
        # Handle both dict-based and legacy tuple-based returns
        if isinstance(output, dict):
            logits = output['edge_logits']
            k_per_node = output.get('k_per_node')
        else:
            logits = output[0]
            k_per_node = None
        
        # Determine which k to use
        if use_learned_k and k_per_node is not None:
            k_to_use = k_per_node
        else:
            k_to_use = k_per_row
        
        # Select top-k edges
        row_groups = build_row_groups(ei.cpu(), num_nodes=n)
        selected = sample_deterministic_topk(logits, row_groups, k_to_use)
        
        # Build C for this level
        C = C_from_selected_edges(current_A, ei.cpu(), selected.cpu().numpy())
        C_list.append(C)
        
        # Compute coarse A (approximate - using standard aggregation)
        # This is needed to continue the loop
        temp_ml = pyamg.smoothed_aggregation_solver(
            current_A, 
            strength=('predefined', {'C': C}),
            max_levels=2,
            max_coarse=1
        )
        if len(temp_ml.levels) > 1:
            current_A = temp_ml.levels[1].A
        else:
            break
    
    # Build final solver with all C matrices
    strength_list = [('predefined', {'C': C}) for C in C_list]
    # Add fallback for any remaining levels
    strength_list.append(('symmetric', {'theta': 0.0}))
    
    kwargs = {
        "strength": strength_list,
        "coarse_solver": coarse_solver,
        "max_coarse": max_coarse,
        "max_levels": max_levels,
    }
    if B is not None:
        kwargs["B"] = B
    
    ml = pyamg.smoothed_aggregation_solver(A, **kwargs)
    
    return ml


def _estimate_coords_for_coarse_level(A: sp.csr_matrix, level: int) -> np.ndarray:
    """
    Estimate node coordinates for coarse level matrices.
    
    For coarse levels, we don't have exact coordinates. We use spectral
    embedding as a reasonable approximation.
    
    Args:
        A: Coarse level matrix
        level: Hierarchy level (0 = fine)
        
    Returns:
        coords: (n, 2) estimated coordinates
    """
    n = A.shape[0]
    
    if n <= 100:
        # For very coarse grids, just use random layout
        return np.random.randn(n, 2)
    
    # Use simple spectral embedding (Laplacian eigenvectors)
    try:
        from scipy.sparse.linalg import eigsh
        
        # Build graph Laplacian
        D = sp.diags(np.array(A.sum(axis=1)).flatten())
        L = D - A
        
        # Get 2 smallest non-trivial eigenvectors
        _, V = eigsh(L.tocsr(), k=3, which='SM', tol=1e-3)
        coords = V[:, 1:3]  # Skip constant eigenvector
        
        # Normalize to [0, 1]
        coords = coords - coords.min(axis=0)
        max_range = coords.max(axis=0) + 1e-10
        coords = coords / max_range
        
    except Exception:
        # Fallback to random
        coords = np.random.randn(n, 2)
    
    return coords


def build_C_from_model(
    A: sp.csr_matrix,
    grid_n: int,
    model: torch.nn.Module,
    k_per_row: int,
    device: str = "cpu",
    use_learned_k: bool = False,
) -> Tuple[sp.csr_matrix, Optional[np.ndarray]]:
    """
    Build C matrix and B candidates from trained model.
    
    Runs model inference and deterministically selects top-k edges per row.
    If the model predicts per-node k values, those can be used instead of fixed k_per_row.
    
    Args:
        A: Sparse matrix
        grid_n: Grid size (number of physical nodes per dimension, NOT DOFs)
        model: Trained AMG policy
        k_per_row: Number of edges to select per row (used if use_learned_k=False or model doesn't predict k)
        device: Device for inference
        use_learned_k: If True and model predicts k, use the learned per-node k values
        
    Returns:
        C: Strength matrix
        B: Near-nullspace candidates (or None)
    """
    from ..data import node_features_for_policy
    from torch_geometric.utils import from_scipy_sparse_matrix
    
    # Build node coordinates
    # For vector problems (elasticity), n = grid_n^2 * dofs_per_node
    # So we need to repeat coordinates for each DOF
    n = A.shape[0]
    num_physical_nodes = grid_n * grid_n
    dofs_per_node = n // num_physical_nodes
    
    # Generate coordinates for physical nodes
    physical_coords = np.zeros((num_physical_nodes, 2), dtype=np.float64)
    physical_coords[:, 0] = np.tile(np.linspace(0, 1, grid_n), grid_n)
    physical_coords[:, 1] = np.repeat(np.linspace(0, 1, grid_n), grid_n)
    
    # Repeat for each DOF (e.g., for elasticity: x-dof and y-dof at same location)
    if dofs_per_node > 1:
        coords = np.repeat(physical_coords, dofs_per_node, axis=0)
    else:
        coords = physical_coords
    
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
        output = model(x, ei, ew)
    
    # Handle both dict-based and legacy tuple-based returns
    if isinstance(output, dict):
        logits = output['edge_logits']
        B_extra = output['B_extra']
        k_per_node = output.get('k_per_node')  # None if learn_k=False
    else:
        logits, B_extra = output[:2]
        k_per_node = None
    
    # Determine which k to use
    if use_learned_k and k_per_node is not None:
        k_to_use = k_per_node
    else:
        k_to_use = k_per_row
    
    # Deterministic top-k selection
    from .sampling import build_row_groups
    row_groups = build_row_groups(ei.cpu(), num_nodes=A.shape[0])
    selected = sample_deterministic_topk(logits, row_groups, k_to_use)
    
    # Build C matrix
    C = C_from_selected_edges(A, ei.cpu(), selected.cpu().numpy())
    
    # Build B candidates
    B = build_B_for_pyamg(B_extra)
    
    return C, B
