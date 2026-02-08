#!/usr/bin/env python
"""
SOLUTION: How to use GNN-predicted C with PyAMG for deep hierarchies.

The root cause of 2-level hierarchies is found in PyAMG's source code:
When strength=('predefined', {'C': C}), PyAMG AUTOMATICALLY sets max_levels=2!

This is by design: a single predefined C matrix only works for one level.
For deeper hierarchies, you need C matrices for EACH level.

This script demonstrates the FIX.
"""
import numpy as np
import scipy.sparse as sp
import pyamg
from pyamg.aggregation.aggregate import standard_aggregation


def main():
    # Create test problem
    grid_n = 64
    A = pyamg.gallery.stencil_grid(
        pyamg.gallery.diffusion_stencil_2d(epsilon=0.01, theta=np.pi/4),
        (grid_n, grid_n), format='csr'
    )
    n = A.shape[0]
    print(f"Test problem: {grid_n}x{grid_n} anisotropic grid, n={n}")
    
    # Baseline hierarchy
    ml_baseline = pyamg.smoothed_aggregation_solver(A, B=np.ones((n,1)), max_levels=10)
    print(f"\nBaseline hierarchy: {[lvl.A.shape[0] for lvl in ml_baseline.levels]}")
    
    # Get GNN C (using symmetric as surrogate)
    C_gnn = pyamg.strength.symmetric_strength_of_connection(A, theta=0.0)
    
    print("\n" + "="*70)
    print("PROBLEM: Single predefined C => 2 levels (max_levels forced to 2)")
    print("="*70)
    
    # BAD: This gives only 2 levels!
    ml_bad = pyamg.smoothed_aggregation_solver(
        A, 
        strength=('predefined', {'C': C_gnn}),
        B=np.ones((n,1)), 
        max_levels=10  # This gets OVERWRITTEN to 2!
    )
    print(f"With predefined (BROKEN): {[lvl.A.shape[0] for lvl in ml_bad.levels]}")
    
    print("\n" + "="*70)
    print("SOLUTION 1: Use predefined C for level 0, then 'symmetric' for rest")
    print("="*70)
    
    # CORRECT: Provide a LIST of strength methods for each level
    # Level 0: Use GNN C (predefined)
    # Level 1+: Use symmetric strength (computed by PyAMG)
    
    ml_good = pyamg.smoothed_aggregation_solver(
        A, 
        strength=[
            ('predefined', {'C': C_gnn}),  # Level 0 -> 1: Use GNN C
            ('symmetric', {'theta': 0.0}),  # Level 1 -> 2: PyAMG computes
            # ('symmetric', ...) is repeated for all subsequent levels
        ],
        B=np.ones((n,1)), 
        max_levels=10,
    )
    print(f"With list [predefined, symmetric]: {[lvl.A.shape[0] for lvl in ml_good.levels]}")
    
    # Test solve
    b = np.random.randn(n)
    res_baseline = []
    ml_baseline.solve(b, tol=1e-8, maxiter=200, residuals=res_baseline)
    
    res_good = []
    ml_good.solve(b, tol=1e-8, maxiter=200, residuals=res_good)
    
    print(f"\nBaseline iters: {len(res_baseline)}")
    print(f"GNN+symmetric iters: {len(res_good)}")
    
    print("\n" + "="*70)
    print("SOLUTION 2: Compute predefined C at EACH level (full GNN control)")
    print("="*70)
    
    # For full control, compute C for each coarse level too
    # This requires access to the coarse A matrices, which we don't have a priori
    # So we need to build the hierarchy level-by-level
    
    def compute_gnn_C(A_level):
        """Simulate GNN computing C for a matrix at any level."""
        # In reality, this would be your GNN inference
        # For now, use symmetric strength as surrogate
        return pyamg.strength.symmetric_strength_of_connection(A_level, theta=0.0)
    
    # Build list of C matrices level by level
    print("\nBuilding C matrices for each level:")
    C_list = []
    A_current = A
    max_lvls = 5
    max_coarse = 10
    
    for lvl in range(max_lvls - 1):
        if A_current.shape[0] <= max_coarse:
            break
        
        # Compute C for this level
        C_lvl = compute_gnn_C(A_current)
        C_list.append(('predefined', {'C': C_lvl}))
        print(f"  Level {lvl}: A.shape={A_current.shape}, C.nnz={C_lvl.nnz}")
        
        # Build this level's aggregation and coarse A (to get next level's A)
        AggOp, _ = standard_aggregation(C_lvl)
        
        # Build tentative P
        B_current = np.ones((A_current.shape[0], 1))
        T, B_coarse = pyamg.aggregation.fit_candidates(AggOp, B_current)
        
        # Simple Galerkin coarsening (no smoothing for size estimation)
        A_coarse = T.T @ A_current @ T
        A_current = A_coarse
    
    print(f"\nComputed {len(C_list)} predefined C matrices")
    
    # Now build the solver with all predefined C matrices
    ml_full = pyamg.smoothed_aggregation_solver(
        A, 
        strength=C_list,
        B=np.ones((n,1)), 
        max_levels=len(C_list) + 1,
    )
    print(f"With all predefined C's: {[lvl.A.shape[0] for lvl in ml_full.levels]}")
    
    res_full = []
    ml_full.solve(b, tol=1e-8, maxiter=200, residuals=res_full)
    print(f"Full GNN control iters: {len(res_full)}")
    
    print("\n" + "="*70)
    print("SUMMARY OF FINDINGS")
    print("="*70)
    print("""
ROOT CAUSE:
-----------
PyAMG's levelize_strength_or_aggregation() function in util/utils.py
at line 1836-1838 contains:

    if to_levelize[0] == 'predefined':
        to_levelize = [to_levelize]
        max_levels = 2   # <-- FORCES 2 LEVELS!
        max_coarse = 0

When you pass strength=('predefined', {'C': C}), PyAMG interprets this as:
"User wants ONE level of predefined coarsening, then stop."

SOLUTIONS:
----------
1. EASIEST: Use a LIST for strength parameter:
   strength=[
       ('predefined', {'C': C_gnn}),  # Level 0: GNN
       ('symmetric', {'theta': 0.0}), # Level 1+: Standard
   ]
   This uses GNN for fine->coarse, then PyAMG's standard for the rest.

2. FULL CONTROL: Pre-compute C for each level:
   - Build hierarchy level by level
   - At each level, run GNN on the coarse A to get C
   - Pass all C matrices as a list to strength parameter

3. ALTERNATIVE: Use 'aggregate' predefined instead of 'strength' predefined:
   - Compute aggregates directly from GNN C
   - Pass aggregates to PyAMG, let it compute strength for smoothing

WHAT MAKES A VALID C MATRIX:
----------------------------
1. Must be SYMMETRIC (C = C.T)
2. Should have DIAGONAL entries (ones on diagonal)  
3. Off-diagonal C[i,j] > 0 means i and j are "strongly connected"
4. Should respect A's sparsity pattern (C[i,j] only if A[i,j] != 0)
5. More connections = larger aggregates = better coarsening ratio

For GNN outputs:
- Select edges that represent STRONG algebraic connections
- Not just random k edges - should prioritize large |A[i,j]| values
- Symmetric selection is important for good aggregation

WHY YOUR CURRENT GNN MAY PRODUCE BAD RESULTS:
---------------------------------------------
Even if the hierarchy depth issue is fixed, the GNN might select edges that:
1. Don't represent strong connections (random selection)
2. Create disconnected subgraphs (poor aggregation)
3. Miss important anisotropic couplings

The GNN should learn to mimic what symmetric_strength_of_connection does:
select edges where |A[i,j]| >= theta * sqrt(|A[i,i]| * |A[j,j]|)
""")


if __name__ == "__main__":
    main()
