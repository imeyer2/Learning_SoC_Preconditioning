#!/usr/bin/env python
"""Test if random C selection (no model) also gives speedup over baseline."""

import numpy as np
import scipy.sparse as sp
import pyamg

# Generate a test problem using pyamg's gallery
A = pyamg.gallery.stencil_grid(
    pyamg.gallery.diffusion_stencil_2d(epsilon=0.01, theta=np.pi/4),
    (256, 256),
    format='csr'
)
n = A.shape[0]
print(f'Matrix size: {n} x {n}')

# Random number generator
rng = np.random.default_rng(42)

# Random b
b = np.random.randn(n)

# Baseline PyAMG (default)
ml_baseline = pyamg.smoothed_aggregation_solver(A, B=np.ones((n,1)), max_levels=10)
residuals_baseline = []
x = ml_baseline.solve(b, tol=1e-8, maxiter=200, residuals=residuals_baseline)
print(f'Baseline (default): {len(residuals_baseline)} iters')

# Create PURELY RANDOM C (k random edges per row, no model involved)
k_per_row = 5
rows, cols = A.nonzero()
C_data = []
C_rows = []
C_cols = []

for i in range(n):
    # Get neighbors of node i
    mask = rows == i
    neighbors = cols[mask]
    neighbors = neighbors[neighbors != i]  # exclude diagonal
    
    if len(neighbors) > 0:
        # Random selection of k neighbors
        k = min(k_per_row, len(neighbors))
        selected = rng.choice(neighbors, size=k, replace=False)
        for j in selected:
            C_data.append(1.0)
            C_rows.append(i)
            C_cols.append(j)

C_random = sp.csr_matrix((C_data, (C_rows, C_cols)), shape=(n, n))
C_random = C_random.maximum(C_random.T)  # symmetrize
C_random = C_random + sp.eye(n)

# PyAMG with purely random C
ml_random_C = pyamg.smoothed_aggregation_solver(
    A, 
    strength=('predefined', {'C': C_random}),
    B=np.ones((n,1)),
    max_levels=10
)
residuals_random_C = []
x = ml_random_C.solve(b, tol=1e-8, maxiter=200, residuals=residuals_random_C)
print(f'Random C (no model): {len(residuals_random_C)} iters')

# PyAMG with symmetric strength (tuned)
ml_tuned = pyamg.smoothed_aggregation_solver(
    A,
    strength=('symmetric', {'theta': 0.25}),
    B=np.ones((n,1)),
    max_levels=10
)
residuals_tuned = []
x = ml_tuned.solve(b, tol=1e-8, maxiter=200, residuals=residuals_tuned)
print(f'Tuned (theta=0.25): {len(residuals_tuned)} iters')

# Create WEIGHT-BASED C (select k strongest edges per row)
C_data2 = []
C_rows2 = []
C_cols2 = []

for i in range(n):
    # Get neighbors of node i
    mask = rows == i
    neighbors = cols[mask]
    neighbors = neighbors[neighbors != i]  # exclude diagonal
    
    if len(neighbors) > 0:
        # Get weights for these edges
        weights = np.abs(np.array([A[i, j] for j in neighbors]))
        
        # Select k strongest
        k = min(k_per_row, len(neighbors))
        strongest_idx = np.argsort(weights)[-k:]
        selected = neighbors[strongest_idx]
        
        for j in selected:
            C_data2.append(1.0)
            C_rows2.append(i)
            C_cols2.append(j)

C_weight = sp.csr_matrix((C_data2, (C_rows2, C_cols2)), shape=(n, n))
C_weight = C_weight.maximum(C_weight.T)  # symmetrize
C_weight = C_weight + sp.eye(n)

# PyAMG with weight-based C
ml_weight_C = pyamg.smoothed_aggregation_solver(
    A, 
    strength=('predefined', {'C': C_weight}),
    B=np.ones((n,1)),
    max_levels=10
)
residuals_weight_C = []
x = ml_weight_C.solve(b, tol=1e-8, maxiter=200, residuals=residuals_weight_C)
print(f'Weight-based C (top-k strongest): {len(residuals_weight_C)} iters')

print(f'\n--- Summary ---')
print(f'Speedup from random C: {len(residuals_baseline)/len(residuals_random_C):.2f}x')
print(f'Speedup from weight-based C: {len(residuals_baseline)/len(residuals_weight_C):.2f}x')
print(f'Speedup from tuned: {len(residuals_baseline)/len(residuals_tuned):.2f}x')
