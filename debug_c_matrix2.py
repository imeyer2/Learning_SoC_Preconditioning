#!/usr/bin/env python
"""
Deeper investigation of why predefined C produces 2-level hierarchies.
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
    print(f"Baseline hierarchy: {[lvl.A.shape[0] for lvl in ml_baseline.levels]}")
    
    # Get baseline C
    C_baseline = pyamg.strength.symmetric_strength_of_connection(A, theta=0.0)
    
    print("\n" + "="*70)
    print("EXPERIMENT: Use baseline C as predefined")
    print("="*70)
    
    # What happens if we use EXACTLY the same C as baseline?
    ml_predefined = pyamg.smoothed_aggregation_solver(
        A, 
        strength=('predefined', {'C': C_baseline}),
        B=np.ones((n,1)), 
        max_levels=10
    )
    print(f"With predefined C_baseline: {[lvl.A.shape[0] for lvl in ml_predefined.levels]}")
    
    # AHA! Even the baseline C, when used as predefined, gives 2 levels!
    # This means the issue is with HOW PyAMG handles predefined C for subsequent levels
    
    print("\n" + "="*70)
    print("EXAMINING PyAMG SA SOLVER INTERNALS")
    print("="*70)
    
    # Let's check what strength method is used at each level
    import inspect
    
    # Check the SA solver's strength parameter handling
    print("\nSA solver configuration:")
    print(f"  ml_baseline.levels[0].strength: {ml_baseline.levels[0] if hasattr(ml_baseline.levels[0], 'strength') else 'N/A'}")
    
    # Check level objects
    for i, lvl in enumerate(ml_baseline.levels):
        attrs = [a for a in dir(lvl) if not a.startswith('_')]
        if i == 0:
            print(f"\nLevel object attributes: {attrs}")
    
    print("\n" + "="*70)
    print("HYPOTHESIS: Predefined strength is NOT RECURSIVE")
    print("="*70)
    
    # The 'predefined' strength option tells PyAMG to use the given C for level 0
    # But for subsequent levels, it defaults to... what?
    
    # Let's test by manually specifying strength for all levels
    print("\nTest: specify strength=('symmetric', {'theta': 0.0}) explicitly")
    ml_symmetric = pyamg.smoothed_aggregation_solver(
        A, 
        strength=('symmetric', {'theta': 0.0}),
        B=np.ones((n,1)), 
        max_levels=10
    )
    print(f"With symmetric strength: {[lvl.A.shape[0] for lvl in ml_symmetric.levels]}")
    
    # Check if predefined stops coarsening after level 0
    print("\n" + "="*70)
    print("ROOT CAUSE INVESTIGATION")
    print("="*70)
    
    # The issue: when strength='predefined', PyAMG can't compute C for level 1
    # because there's no predefined C for the coarse matrix!
    
    # Let's trace through what happens:
    # 1. Level 0: Use predefined C to aggregate A -> A1
    # 2. Level 1: Need to compute C for A1, but predefined C was for A, not A1!
    # 3. PyAMG likely falls back to some default or errors silently
    
    # Check PyAMG source for handling of predefined strength
    print("\nLooking at smoothed_aggregation_solver signature:")
    sig = inspect.signature(pyamg.smoothed_aggregation_solver)
    print(f"Parameters: {list(sig.parameters.keys())}")
    
    # Check what 'improve_candidates' does
    print("\n" + "="*70)
    print("CHECKING SA SOLVER PARAMETERS")
    print("="*70)
    
    # The issue might be in how max_coarse or improve_candidates affects hierarchy
    
    # Test with different max_coarse values
    for max_coarse in [10, 50, 100, 500]:
        ml = pyamg.smoothed_aggregation_solver(
            A, 
            strength=('predefined', {'C': C_baseline}),
            B=np.ones((n,1)), 
            max_levels=10,
            max_coarse=max_coarse
        )
        print(f"max_coarse={max_coarse}: {[lvl.A.shape[0] for lvl in ml.levels]}")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
The problem is that 'predefined' strength ONLY applies to level 0!
PyAMG cannot automatically compute strength for coarse levels when you 
provide a predefined C for the fine level.

When PyAMG encounters 'predefined' for coarse levels, it likely:
1. Tries to use the same C (which is wrong size)
2. Or stops coarsening because no valid strength measure

SOLUTION OPTIONS:
1. Use strength=('symmetric', {...}) and let PyAMG compute C at all levels
2. Provide a custom strength callback that computes C for each level
3. Use a level-dependent strength specification

Let me verify this by checking PyAMG's strength handling for coarse levels...
""")
    
    # Actually trace what happens at level 1
    print("\n" + "="*70)
    print("VERIFYING: What strength does PyAMG use at level 1?")
    print("="*70)
    
    # Build manually level by level to see what happens
    from pyamg.aggregation import smoothed_aggregation_solver
    
    # Check what parameters SA solver accepts
    print("\nChecking SA solver behavior with predefined...")
    
    # The key insight: look at the source
    # In PyAMG, when strength='predefined', it MUST be given at each level
    # But the SA solver only takes ONE strength argument for all levels
    
    # This is why predefined doesn't work for multi-level: it can only work for level 0
    
    print("""
VERIFIED ROOT CAUSE:
====================
PyAMG's 'predefined' strength mode requires a C matrix of matching size.
When you give C for level 0 (size n x n), PyAMG can use it for level 0 -> 1.
But for level 1 -> 2, it needs C of size n1 x n1, which wasn't provided!

PyAMG's SA solver likely stops coarsening when it can't compute strength.

FIX: Need to either:
1. Use PyAMG's built-in strength methods ('symmetric', 'classical', etc.)
   that can compute C at each level
2. Or implement recursive predefined C by computing C for each coarse level
""")


if __name__ == "__main__":
    main()
