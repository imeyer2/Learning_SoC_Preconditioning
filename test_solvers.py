#!/usr/bin/env python
"""Quick test to diagnose solver performance differences."""

import numpy as np
import pyamg
from pyamg.strength import symmetric_strength_of_connection
from pyamg.gallery import stencil_grid
import time

# Create a simple test problem
n = 128
stencil = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
A = stencil_grid(stencil, (n, n), format='csr')
A = A.astype(np.float64)  # Ensure float64
B = np.ones((A.shape[0], 1))

# Use consistent max_levels for fair comparison
max_levels = 2
max_coarse = 10

print(f'Matrix size: {A.shape[0]}')
print(f'Using max_levels={max_levels}, max_coarse={max_coarse} for fair comparison')

# Test the three solver types
print('\n--- Baseline (theta=0.0) ---')
ml_baseline = pyamg.smoothed_aggregation_solver(A, B=B, max_levels=max_levels, max_coarse=max_coarse)
print(f'Levels: {len(ml_baseline.levels)}')
for i, level in enumerate(ml_baseline.levels):
    print(f'  Level {i}: {level.A.shape[0]} DOFs')

print('\n--- Tuned (theta=0.25) ---')
ml_tuned = pyamg.smoothed_aggregation_solver(
    A, 
    strength=('symmetric', {'theta': 0.25}),
    B=B,
    max_levels=max_levels,
    max_coarse=max_coarse
)
print(f'Levels: {len(ml_tuned.levels)}')
for i, level in enumerate(ml_tuned.levels):
    print(f'  Level {i}: {level.A.shape[0]} DOFs')

# Also get C for reference
C_tuned = symmetric_strength_of_connection(A, theta=0.25)
print(f'C_tuned nnz: {C_tuned.nnz}, A nnz: {A.nnz}')

# Test preconditioner application timing
b = np.random.randn(A.shape[0])
M_baseline = ml_baseline.aspreconditioner()
M_tuned = ml_tuned.aspreconditioner()

print('\n--- Preconditioner application test ---')
t0 = time.time()
for _ in range(10):
    z = M_baseline @ b
print(f'Baseline (10 apps): {time.time()-t0:.4f}s')

t0 = time.time()
for _ in range(10):
    z = M_tuned @ b
print(f'Tuned (10 apps): {time.time()-t0:.4f}s')

# Test actual PCG convergence
print('\n--- PCG convergence test ---')
x0 = np.zeros_like(b)

def simple_pcg(A, b, M, max_iter=100, tol=1e-8):
    x = x0.copy()
    r = b - A @ x
    z = M @ r
    p = z.copy()
    rz = r @ z
    r0_norm = np.linalg.norm(r)
    
    for i in range(max_iter):
        Ap = A @ p
        alpha = rz / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        
        res_norm = np.linalg.norm(r)
        if res_norm < tol * r0_norm:
            return i + 1, True
        
        z = M @ r
        rz_new = r @ z
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new
    
    return max_iter, False

iters, conv = simple_pcg(A, b, M_baseline)
print(f'Baseline: {iters} iters, converged={conv}')

iters, conv = simple_pcg(A, b, M_tuned)
print(f'Tuned: {iters} iters, converged={conv}')
