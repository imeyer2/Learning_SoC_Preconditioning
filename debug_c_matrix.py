#!/usr/bin/env python
"""
Debug script to understand why GNN C matrices produce bad hierarchies.
"""
import numpy as np
import scipy.sparse as sp
import pyamg
from pyamg.aggregation.aggregate import standard_aggregation


def analyze_c_matrix(C, name):
    """Analyze C matrix properties."""
    n = C.shape[0]
    print(f"\n{'='*60}")
    print(f"Analysis of {name}")
    print(f"{'='*60}")
    
    # Basic properties
    print(f"Shape: {C.shape}")
    print(f"nnz: {C.nnz}")
    print(f"nnz/row: {C.nnz/n:.2f}")
    print(f"Format: {C.format}")
    
    # Check symmetry
    diff = abs(C - C.T)
    print(f"Symmetric: {diff.nnz == 0}")
    
    # Check diagonal
    diag = C.diagonal()
    print(f"Diagonal nonzeros: {(diag != 0).sum()}/{n}")
    print(f"Diagonal all ones: {np.all(diag == 1)}")
    
    # Check values
    unique_vals = np.unique(C.data)
    print(f"Unique values: {unique_vals[:10]}{'...' if len(unique_vals) > 10 else ''}")
    print(f"All binary: {np.all((C.data == 0) | (C.data == 1))}")
    
    # Off-diagonal connections per row (excluding diagonal)
    C_no_diag = C - sp.diags(C.diagonal())
    offdiag_per_row = np.diff(C_no_diag.indptr)
    print(f"Off-diag connections/row: min={offdiag_per_row.min()}, max={offdiag_per_row.max()}, mean={offdiag_per_row.mean():.1f}")
    print(f"Rows with 0 off-diag connections: {(offdiag_per_row == 0).sum()}")
    
    return C


def test_aggregation(C, name):
    """Test what aggregation produces from C."""
    print(f"\n--- Testing aggregation for {name} ---")
    
    # Ensure CSR format
    C_csr = C.tocsr()
    
    try:
        AggOp, Cpts = standard_aggregation(C_csr)
        num_aggs = AggOp.shape[1]
        print(f"Number of aggregates: {num_aggs}")
        print(f"Coarsening ratio: {C.shape[0] / num_aggs:.2f}x")
        
        # Check aggregate sizes
        agg_sizes = np.array(AggOp.sum(axis=0)).flatten()
        print(f"Aggregate sizes: min={agg_sizes.min()}, max={agg_sizes.max()}, mean={agg_sizes.mean():.1f}")
        
        # Check for isolated nodes (not in any aggregate)
        node_membership = np.array(AggOp.sum(axis=1)).flatten()
        isolated = (node_membership == 0).sum()
        print(f"Isolated nodes (not aggregated): {isolated}")
        
        return num_aggs
        
    except Exception as e:
        print(f"Aggregation failed: {e}")
        return None


def main():
    # Create test problem
    grid_n = 64
    A = pyamg.gallery.stencil_grid(
        pyamg.gallery.diffusion_stencil_2d(epsilon=0.01, theta=np.pi/4),
        (grid_n, grid_n), format='csr'
    )
    n = A.shape[0]
    print(f"Test problem: {grid_n}x{grid_n} anisotropic grid, n={n}")
    
    # 1. Baseline: PyAMG symmetric strength
    print("\n" + "="*70)
    print("1. BASELINE: PyAMG Symmetric Strength (theta=0)")
    print("="*70)
    
    C_baseline = pyamg.strength.symmetric_strength_of_connection(A, theta=0.0)
    analyze_c_matrix(C_baseline, "C_baseline")
    test_aggregation(C_baseline, "C_baseline")
    
    # Build full hierarchy
    ml = pyamg.smoothed_aggregation_solver(A, B=np.ones((n,1)), max_levels=10)
    print(f"\nFull hierarchy: {[lvl.A.shape[0] for lvl in ml.levels]}")
    
    # 2. Create a "bad" C matrix similar to GNN output
    print("\n" + "="*70)
    print("2. SIMULATED GNN C (k=5 random per row)")
    print("="*70)
    
    rng = np.random.default_rng(42)
    indptr = A.indptr
    indices = A.indices
    C_rows, C_cols = [], []
    k = 5
    for i in range(n):
        start, end = indptr[i], indptr[i+1]
        nbrs = indices[start:end]
        nbrs = nbrs[nbrs != i]  # Exclude self
        if len(nbrs) > 0:
            k_actual = min(k, len(nbrs))
            sel = rng.choice(nbrs, size=k_actual, replace=False)
            C_rows.extend([i]*k_actual)
            C_cols.extend(sel)
    
    C_gnn = sp.csr_matrix((np.ones(len(C_rows)), (C_rows, C_cols)), shape=(n,n))
    C_gnn = C_gnn.maximum(C_gnn.T) + sp.eye(n, format='csr')
    
    analyze_c_matrix(C_gnn, "C_gnn_random")
    test_aggregation(C_gnn, "C_gnn_random")
    
    # Build hierarchy with predefined C
    print("\nBuilding hierarchy with predefined C_gnn...")
    ml_gnn = pyamg.smoothed_aggregation_solver(
        A, 
        strength=('predefined', {'C': C_gnn}),
        B=np.ones((n,1)), 
        max_levels=10
    )
    print(f"Hierarchy: {[lvl.A.shape[0] for lvl in ml_gnn.levels]}")
    
    # 3. Compare structure
    print("\n" + "="*70)
    print("3. STRUCTURAL COMPARISON")
    print("="*70)
    
    print(f"\nBaseline C nnz/row: {C_baseline.nnz / n:.2f}")
    print(f"GNN C nnz/row: {C_gnn.nnz / n:.2f}")
    
    # Check connectivity
    print("\n--- Connectivity Analysis ---")
    
    # For baseline, check how many connections come from A
    A_pattern = (A != 0).astype(float)
    baseline_in_A = (C_baseline.multiply(A_pattern)).nnz
    print(f"Baseline connections in A: {baseline_in_A}/{C_baseline.nnz} ({100*baseline_in_A/C_baseline.nnz:.1f}%)")
    
    gnn_in_A = (C_gnn.multiply(A_pattern)).nnz
    print(f"GNN connections in A: {gnn_in_A}/{C_gnn.nnz} ({100*gnn_in_A/C_gnn.nnz:.1f}%)")
    
    # 4. Test: what if we use the EXACT sparsity pattern of PyAMG's C?
    print("\n" + "="*70)
    print("4. TEST: Binary version of baseline C")
    print("="*70)
    
    C_binary = C_baseline.copy()
    C_binary.data[:] = 1.0
    
    analyze_c_matrix(C_binary, "C_binary_baseline")
    test_aggregation(C_binary, "C_binary_baseline")
    
    ml_binary = pyamg.smoothed_aggregation_solver(
        A, 
        strength=('predefined', {'C': C_binary}),
        B=np.ones((n,1)), 
        max_levels=10
    )
    print(f"Hierarchy with binary baseline C: {[lvl.A.shape[0] for lvl in ml_binary.levels]}")
    
    # 5. Key insight: Check the AGGREGATION for level 1
    print("\n" + "="*70)
    print("5. LEVEL 1 AGGREGATION ANALYSIS")
    print("="*70)
    
    # For baseline
    AggOp_baseline, _ = standard_aggregation(C_baseline.tocsr())
    n1_baseline = AggOp_baseline.shape[1]
    
    # For GNN
    AggOp_gnn, _ = standard_aggregation(C_gnn.tocsr())
    n1_gnn = AggOp_gnn.shape[1]
    
    print(f"Baseline: {n} -> {n1_baseline} ({n/n1_baseline:.2f}x coarsening)")
    print(f"GNN:      {n} -> {n1_gnn} ({n/n1_gnn:.2f}x coarsening)")
    
    # What does the coarse A look like for GNN?
    print("\n--- Why does GNN hierarchy stop at level 2? ---")
    
    # Get level 1 matrix from ml_gnn
    A1 = ml_gnn.levels[1].A
    print(f"Level 1 A: {A1.shape}, nnz={A1.nnz}, nnz/row={A1.nnz/A1.shape[0]:.1f}")
    
    # Compute strength for level 1
    C1_baseline = pyamg.strength.symmetric_strength_of_connection(A1, theta=0.0)
    print(f"Level 1 C (symmetric, theta=0): nnz={C1_baseline.nnz}, nnz/row={C1_baseline.nnz/A1.shape[0]:.1f}")
    
    # Test aggregation on level 1
    AggOp1, _ = standard_aggregation(C1_baseline.tocsr())
    print(f"Level 1 -> Level 2 aggregation: {A1.shape[0]} -> {AggOp1.shape[1]}")
    
    print("\n" + "="*70)
    print("6. DIAGNOSIS: Why poor coarsening?")
    print("="*70)
    
    # The issue: when we use predefined C, it ONLY applies to level 0!
    # For subsequent levels, PyAMG still computes its own strength
    # But with our weird aggregates, the coarse matrices are pathological
    
    print("""
KEY INSIGHT:
- 'predefined' C ONLY affects level 0 -> level 1 coarsening
- For subsequent levels, PyAMG computes strength from the coarse A matrices
- If level 1 aggregation is poor (too many aggregates), then:
  - The coarse A matrix is too large
  - Its structure may be pathological (very sparse or very dense rows)
  - Standard strength measures may fail to find good connections
  
LIKELY CAUSES of bad GNN coarsening:
1. GNN selects RANDOM k edges, not STRONG connections
2. Standard_aggregation forms aggregates greedily:
   - Pick unaggregated node as seed
   - Add strongly connected neighbors
   - If neighbors already aggregated, node stays isolated or forms tiny aggregate
3. With random selection, nodes may select neighbors that are FAR away in the graph
   - This creates many small/isolated aggregates
   - Poor coarsening ratio (~3x instead of ~8x)
""")
    
    # Demonstrate: what happens if we select edges with ACTUAL weights?
    print("\n" + "="*70)
    print("7. FIX: Select top-k edges by A[i,j] VALUE (not random)")
    print("="*70)
    
    C_rows, C_cols = [], []
    k = 5
    for i in range(n):
        start, end = indptr[i], indptr[i+1]
        row_j = indices[start:end]
        row_val = np.abs(A.data[start:end])
        
        # Exclude diagonal
        mask = row_j != i
        row_j = row_j[mask]
        row_val = row_val[mask]
        
        if len(row_j) > 0:
            # Select top-k by absolute value
            k_actual = min(k, len(row_j))
            top_k = np.argsort(row_val)[-k_actual:]  # Largest values
            sel = row_j[top_k]
            C_rows.extend([i]*k_actual)
            C_cols.extend(sel)
    
    C_topk = sp.csr_matrix((np.ones(len(C_rows)), (C_rows, C_cols)), shape=(n,n))
    C_topk = C_topk.maximum(C_topk.T) + sp.eye(n, format='csr')
    
    analyze_c_matrix(C_topk, "C_topk_by_value")
    test_aggregation(C_topk, "C_topk_by_value")
    
    ml_topk = pyamg.smoothed_aggregation_solver(
        A, 
        strength=('predefined', {'C': C_topk}),
        B=np.ones((n,1)), 
        max_levels=10
    )
    print(f"Hierarchy with top-k by value: {[lvl.A.shape[0] for lvl in ml_topk.levels]}")


if __name__ == "__main__":
    main()
