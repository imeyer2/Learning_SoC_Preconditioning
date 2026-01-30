"""
Legacy code from main.py that still needs to be modularized.

This file contains the remaining pieces from the original main.py:
- RL training logic (REINFORCE)
- PyAMG solver building
- Reward computation
- Evaluation functions
- Visualization functions

TODO: Break this into proper modules:
- src/p_sparsity/rl/ (training algorithms, rewards, baselines)
- src/p_sparsity/pyamg_interface/ (solver building, sampling)
- src/p_sparsity/evaluation/ (PCG, V-cycle, eigenvalue analysis)
- src/p_sparsity/visualization/ (all plotting functions)
"""

# This is the content from the original main.py lines that handle:
# - PyAMG solver building (build_pyamg_solver, C_from_selected_edges, etc.)
# - Sampling strategies (sample_topk_without_replacement)
# - Reward computation (compute_reward, one_vcycle_error_reduce_ratio)
# - Training loop (train_policy)
# - Evaluation (evaluate_on_case, pcg_iterations, etc.)
# - Visualization (all plot_ functions)

# Import the original code
import sys
import os

# Get the full content from main.py that we need to preserve
# This will be properly modularized in the next phase

# For now, we'll create placeholder imports
from typing import *
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyamg

# ==============================================================================
# SECTION 1: PyAMG INTERFACE - Solver Building and Sampling
# ==============================================================================

def build_row_groups(edge_index: torch.Tensor, num_nodes: int) -> List[torch.Tensor]:
    """Build row groups for sampling."""
    row = edge_index[0].cpu().numpy()
    buckets: List[List[int]] = [[] for _ in range(num_nodes)]
    for e, r in enumerate(row):
        buckets[int(r)].append(e)
    return [torch.tensor(b, dtype=torch.long) for b in buckets]


def sample_topk_without_replacement(
    logits: torch.Tensor,
    row_groups: List[torch.Tensor],
    k: int,
    temperature: float = 1.0,
    gumbel: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample top-k edges per row without replacement."""
    device = logits.device
    E = logits.numel()
    selected = torch.zeros(E, dtype=torch.bool, device=device)
    logp_total = torch.zeros((), dtype=torch.float32, device=device)
    
    tau = max(float(temperature), 1e-6)
    
    for edges_i in row_groups:
        if edges_i.numel() and int(edges_i.max()) >= E:
            raise RuntimeError(f"row_groups contain edge id {int(edges_i.max())} but logits has E={E}")
        if edges_i.numel() == 0:
            continue
        edges_i = edges_i.to(device)
        
        kk = min(k, edges_i.numel())
        if kk <= 0:
            continue
        
        local_logits = logits[edges_i] / tau
        remaining = edges_i.clone()
        rem_logits = local_logits.clone()
        
        for _t in range(kk):
            if gumbel:
                u = torch.rand_like(rem_logits)
                g = -torch.log(-torch.log(u + 1e-12) + 1e-12)
                sample_logits = rem_logits + g
            else:
                sample_logits = rem_logits
            
            probs = F.softmax(rem_logits, dim=0)
            j = torch.argmax(sample_logits)
            chosen_edge = remaining[j]
            
            selected[chosen_edge] = True
            logp_total = logp_total + torch.log(probs[j] + 1e-12)
            
            keep = torch.ones(remaining.numel(), dtype=torch.bool, device=device)
            keep[j] = False
            remaining = remaining[keep]
            rem_logits = rem_logits[keep]
            if remaining.numel() == 0:
                break
    
    return selected, logp_total


def C_from_selected_edges(A: sp.csr_matrix, edge_index_cpu: torch.Tensor, selected_mask: np.ndarray) -> sp.csr_matrix:
    """Build symmetric binary strength matrix C."""
    r = edge_index_cpu[0].cpu().numpy()
    c = edge_index_cpu[1].cpu().numpy()
    
    sel = selected_mask.astype(bool)
    keep = sel & (r != c)
    
    rr = r[keep]
    cc = c[keep]
    vals = np.ones(rr.shape[0], dtype=np.float64)
    
    n = A.shape[0]
    C = sp.csr_matrix((vals, (rr, cc)), shape=(n, n))
    C = C.maximum(C.T)
    C = C + sp.eye(n, format="csr")
    return C


def build_B_for_pyamg(B_extra: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    """Build near-nullspace candidates for PyAMG."""
    if B_extra is None:
        return None
    
    Bx = B_extra.detach().cpu().numpy().astype(np.float64)
    n, p = Bx.shape
    for j in range(p):
        col = Bx[:, j]
        col = col - col.mean()
        denom = np.linalg.norm(col) + 1e-12
        Bx[:, j] = col / denom
    
    ones = np.ones((n, 1), dtype=np.float64)
    B = np.hstack([ones, Bx])
    return B


def build_pyamg_solver(A: sp.csr_matrix, C: sp.csr_matrix, B: Optional[np.ndarray],
                      coarse_solver: str = "splu"):
    """Build SA solver using predefined strength matrix C."""
    strength = ("predefined", {"C": C})
    if B is None:
        ml = pyamg.smoothed_aggregation_solver(A, strength=strength, coarse_solver=coarse_solver)
    else:
        ml = pyamg.smoothed_aggregation_solver(A, strength=strength, B=B, coarse_solver=coarse_solver)
    return ml


# ==============================================================================
# SECTION 2: REWARD COMPUTATION
# ==============================================================================

def energy_norm_sq(A: sp.csr_matrix, e: np.ndarray) -> float:
    """Compute A-energy norm squared."""
    Ae = A @ e
    return float(e @ Ae)


def one_vcycle_error_reduce_ratio(ml, A: sp.csr_matrix, e0: np.ndarray, cycle: str = "V") -> float:
    """Apply one V-cycle and return energy reduction ratio."""
    b0 = np.zeros_like(e0)
    try:
        e1 = ml.solve(b0, x0=e0, tol=0.0, maxiter=1, accel=None, cycle=cycle)
    except TypeError:
        e1 = ml.solve(b0, x0=e0, tol=1e-30, maxiter=1, accel=None, cycle=cycle)
    
    num = energy_norm_sq(A, e1)
    den = energy_norm_sq(A, e0) + 1e-30
    return num / den


# NOTE: The rest of the original code (training loop, evaluation, visualization)
# should be properly modularized into the appropriate modules. For now, this file
# serves as a reference point during the migration.

# Users should import from the new modular structure instead of using this file directly.
