#!/usr/bin/env python
"""
Reproducible comparison of C matrix construction methods.

IMPORTANT: Tests AMG as a PRECONDITIONER (not as a solver!)
Uses PCG with AMG preconditioning, which is the practical use case.

Run from project root:
    python scripts/compare_c_methods.py
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg
import pyamg
import torch
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from p_sparsity.utils.config import load_config
from p_sparsity.models import build_policy_from_config
from p_sparsity.data import node_features_for_policy
from p_sparsity.pyamg_interface.sampling import sample_deterministic_topk, build_row_groups
from p_sparsity.pyamg_interface.solver_builder import C_from_selected_edges, build_B_for_pyamg
from torch_geometric.utils import from_scipy_sparse_matrix


def get_device():
    """Get best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon)")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")


def pcg_solve(A, b, ml, rtol=1e-8, maxiter=200):
    """
    Solve Ax=b using PCG with AMG preconditioner.
    
    Returns:
        x: Solution
        info: Convergence info (0 = success)
        iters: Number of iterations
        residuals: List of residual norms
    """
    # Get preconditioner from AMG hierarchy
    M = ml.aspreconditioner(cycle='V')
    
    # Track iterations
    residuals = []
    
    def callback(xk):
        r = b - A @ xk
        residuals.append(np.linalg.norm(r))
    
    x, info = cg(A, b, M=M, rtol=rtol, maxiter=maxiter, callback=callback)
    
    return x, info, len(residuals), residuals


def main():
    grid_n = 256
    k_per_row = 5  # Need enough connections for proper hierarchy
    seed = 42

    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    
    device = get_device()

    print(f'Generating {grid_n}x{grid_n} anisotropic problem...')
    A = pyamg.gallery.stencil_grid(
        pyamg.gallery.diffusion_stencil_2d(epsilon=0.01, theta=np.pi/4),
        (grid_n, grid_n), format='csr'
    )
    n = A.shape[0]
    b = rng.standard_normal(n)
    print(f'n={n}, nnz={A.nnz}')
    print('\n*** Using AMG as PRECONDITIONER for PCG (not as solver) ***\n')

    # 1. Baseline - no preconditioning (plain CG)
    print('Running plain CG (no preconditioner)...')
    t0 = time.time()
    res_plain = []
    def cb_plain(xk):
        res_plain.append(np.linalg.norm(b - A @ xk))
    x, info = cg(A, b, rtol=1e-8, maxiter=500, callback=cb_plain)
    plain_iters = len(res_plain)
    print(f'  Time: {time.time()-t0:.2f}s')
    print(f'Plain CG: {plain_iters} iters (converged={info==0})')

    # 2. Baseline AMG preconditioner
    print('\nRunning PCG with baseline AMG preconditioner...')
    t0 = time.time()
    ml = pyamg.smoothed_aggregation_solver(A, B=np.ones((n,1)), max_levels=10)
    print(f'  Hierarchy: {[lvl.A.shape[0] for lvl in ml.levels]}')
    x, info, iters, res = pcg_solve(A, b, ml)
    print(f'  Time: {time.time()-t0:.2f}s')
    baseline = iters
    print(f'Baseline AMG-PCG: {baseline} iters (converged={info==0})')

    # 3. Tuned AMG preconditioner
    print('\nRunning PCG with tuned AMG preconditioner...')
    t0 = time.time()
    ml = pyamg.smoothed_aggregation_solver(A, strength=('symmetric', {'theta': 0.25}), B=np.ones((n,1)), max_levels=10)
    print(f'  Hierarchy: {[lvl.A.shape[0] for lvl in ml.levels]}')
    x, info, iters, res = pcg_solve(A, b, ml)
    print(f'  Time: {time.time()-t0:.2f}s')
    print(f'Tuned AMG-PCG: {iters} iters, speedup={baseline/iters:.2f}x (converged={info==0})')

    # 4. Random C (hybrid) - using CSR structure for speed
    print('\nBuilding random C...')
    indptr = A.indptr
    indices = A.indices
    C_rows, C_cols = [], []
    for i in range(n):
        start, end = indptr[i], indptr[i+1]
        nbrs = indices[start:end]
        nbrs = nbrs[nbrs != i]
        if len(nbrs) > 0:
            k = min(k_per_row, len(nbrs))
            sel = rng.choice(nbrs, size=k, replace=False)
            C_rows.extend([i]*k)
            C_cols.extend(sel)
    C = sp.csr_matrix((np.ones(len(C_rows)), (C_rows, C_cols)), shape=(n,n))
    C = C.maximum(C.T) + sp.eye(n)
    
    print('Running PCG with random C AMG preconditioner (hybrid)...')
    t0 = time.time()
    # Use hybrid: random C for level 0, PyAMG for rest
    strength_random = [
        ('predefined', {'C': C}),
        ('symmetric', {'theta': 0.0}),
    ]
    ml = pyamg.smoothed_aggregation_solver(A, strength=strength_random, B=np.ones((n,1)), max_levels=10)
    print(f'  Hierarchy: {[lvl.A.shape[0] for lvl in ml.levels]}')
    x, info, iters, res = pcg_solve(A, b, ml)
    print(f'  Time: {time.time()-t0:.2f}s')
    print(f'Random C AMG-PCG: {iters} iters, speedup={baseline/iters:.2f}x (converged={info==0})')

    # 5. Random GNN (hybrid)
    print('\nBuilding random GNN C...')
    t0 = time.time()
    model_cfg = load_config('configs/model/gat_default.yaml')
    model = build_policy_from_config(model_cfg).to(device)
    print(f'  Model creation: {time.time()-t0:.2f}s')
    
    t0 = time.time()
    coords = np.zeros((n, 2))
    coords[:, 0] = np.tile(np.linspace(0, 1, grid_n), grid_n)
    coords[:, 1] = np.repeat(np.linspace(0, 1, grid_n), grid_n)
    feature_cfg = {
        'use_relaxed_vectors': True, 
        'use_coordinates': True, 
        'num_vecs': 4, 
        'relax_iters': 5, 
        'relaxation_scheme': 'jacobi', 
        'omega': 2/3
    }
    x = node_features_for_policy(A, coords, feature_cfg).to(device)
    print(f'  Feature computation: {time.time()-t0:.2f}s')
    
    t0 = time.time()
    ei, ew = from_scipy_sparse_matrix(A)
    ew = ew.float() / ew.abs().max()
    ei = ei.to(device)
    ew = ew.to(device)
    print(f'  Graph conversion: {time.time()-t0:.2f}s')
    
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        logits, B_extra = model(x, ei, ew)
    if device.type != 'cpu':
        torch.cuda.synchronize() if device.type == 'cuda' else torch.mps.synchronize()
    print(f'  GNN forward pass: {time.time()-t0:.2f}s')
    
    t0 = time.time()
    # Move back to CPU for sampling
    logits_cpu = logits.cpu()
    ei_cpu = ei.cpu()
    row_groups = build_row_groups(ei_cpu, num_nodes=n)
    selected = sample_deterministic_topk(logits_cpu, row_groups, k_per_row)
    C_gnn = C_from_selected_edges(A, ei_cpu, selected.numpy())
    B_gnn = build_B_for_pyamg(B_extra.cpu() if B_extra is not None else None)
    print(f'  C matrix construction: {time.time()-t0:.2f}s')
    
    print('Running PCG with random GNN AMG preconditioner (hybrid)...')
    t0 = time.time()
    
    # Use HYBRID strategy: GNN for level 0, PyAMG standard for coarser levels
    strength_hybrid = [
        ('predefined', {'C': C_gnn}),   # Level 0: GNN-generated C
        ('symmetric', {'theta': 0.0}),   # Level 1+: PyAMG standard
    ]
    
    ml = pyamg.smoothed_aggregation_solver(
        A, 
        strength=strength_hybrid, 
        B=B_gnn if B_gnn is not None else np.ones((n,1)), 
        max_levels=10
    )
    print(f'  Hierarchy: {[lvl.A.shape[0] for lvl in ml.levels]}')
    x, info, iters, res = pcg_solve(A, b, ml)
    print(f'  Time: {time.time()-t0:.2f}s')
    print(f'Random GNN AMG-PCG (hybrid): {iters} iters, speedup={baseline/iters:.2f}x (converged={info==0})')

    print()
    print('=' * 60)
    print(f'SUMMARY ({grid_n}x{grid_n} anisotropic, eps=0.01, theta=pi/4)')
    print('=' * 60)
    print('AMG used as PRECONDITIONER for PCG (not as standalone solver)')


if __name__ == "__main__":
    main()
