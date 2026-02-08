#!/usr/bin/env python
"""Debug script to compare trained vs random model on the same problem."""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

np.random.seed(12345)
torch.manual_seed(12345)

from p_sparsity.utils import load_config
from p_sparsity.models import build_policy_from_config
from p_sparsity.case_studies import AnisotropicGenerator
from p_sparsity.pyamg_interface import build_C_from_model, build_pyamg_solver
import scipy.sparse.linalg as spla
import pyamg

# Create test problem
gen = AnisotropicGenerator()
instance = gen.generate(32, params={'theta': 1.5, 'epsilon': 0.01})
A = instance.A
grid_size = instance.grid_size
print(f'Problem: n={A.shape[0]}, nnz={A.nnz}')

# Load trained model
model_cfg = load_config('configs/model/gat_default.yaml')
trained_model = build_policy_from_config(model_cfg)
ckpt = torch.load(
    'case_studies/results/study_1_anisotropic/case_study_1_anisotropic/variation_A/checkpoints/best_model.pt', 
    weights_only=False
)
trained_model.load_state_dict(ckpt['model_state_dict'])
trained_model.eval()

# Create random model (same seed as Variation D uses - seed 42 from config)
torch.manual_seed(42)
np.random.seed(42)
random_model = build_policy_from_config(model_cfg)
random_model.eval()

# Build C matrices
k_per_row = 3
C_trained, B_trained = build_C_from_model(A, grid_size, trained_model, k_per_row, device='cpu')
C_random, B_random = build_C_from_model(A, grid_size, random_model, k_per_row, device='cpu')

print(f'C_trained: nnz={C_trained.nnz}, density={C_trained.nnz/A.shape[0]**2:.4f}')
print(f'C_random: nnz={C_random.nnz}, density={C_random.nnz/A.shape[0]**2:.4f}')

# Check if they're different
diff = (C_trained != C_random).sum()
print(f'C matrices differ in {diff} entries')

# Build solvers
ml_trained = build_pyamg_solver(A, C_trained, B_trained)
ml_random = build_pyamg_solver(A, C_random, B_random)
ml_baseline = pyamg.smoothed_aggregation_solver(A)

print(f'\nHierarchy levels: trained={len(ml_trained.levels)}, random={len(ml_random.levels)}, baseline={len(ml_baseline.levels)}')

# Solve with same b
np.random.seed(99999)
b = np.random.randn(A.shape[0])

def count_iters(A, b, ml):
    iters = [0]
    def callback(x):
        iters[0] += 1
    x, info = spla.cg(A, b, M=ml.aspreconditioner(), rtol=1e-8, callback=callback)
    return iters[0]

iters_trained = count_iters(A, b, ml_trained)
iters_random = count_iters(A, b, ml_random)
iters_baseline = count_iters(A, b, ml_baseline)

print(f'\nIterations: trained={iters_trained}, random={iters_random}, baseline={iters_baseline}')
print(f'Speedup: trained={iters_baseline/iters_trained:.2f}x, random={iters_baseline/iters_random:.2f}x')
print(f'Trained vs Random: {iters_random/iters_trained:.2f}x')

# Run on multiple problems
print("\n--- Testing on 10 random problems ---")
trained_wins = 0
random_wins = 0
ties = 0

for i in range(10):
    theta = np.random.uniform(0, 2*np.pi)
    epsilon = np.exp(np.random.uniform(np.log(0.001), np.log(0.5)))
    instance = gen.generate(32, params={'theta': theta, 'epsilon': epsilon})
    A = instance.A
    grid_size = instance.grid_size
    
    C_trained, B_trained = build_C_from_model(A, grid_size, trained_model, k_per_row, device='cpu')
    C_random, B_random = build_C_from_model(A, grid_size, random_model, k_per_row, device='cpu')
    
    ml_trained = build_pyamg_solver(A, C_trained, B_trained)
    ml_random = build_pyamg_solver(A, C_random, B_random)
    ml_baseline = pyamg.smoothed_aggregation_solver(A)
    
    b = np.random.randn(A.shape[0])
    
    it_trained = count_iters(A, b, ml_trained)
    it_random = count_iters(A, b, ml_random)
    it_baseline = count_iters(A, b, ml_baseline)
    
    if it_trained < it_random:
        trained_wins += 1
        winner = "TRAINED"
    elif it_random < it_trained:
        random_wins += 1
        winner = "RANDOM"
    else:
        ties += 1
        winner = "TIE"
    
    print(f"  Problem {i+1}: theta={theta:.2f}, eps={epsilon:.4f} -> trained={it_trained}, random={it_random}, baseline={it_baseline} [{winner}]")

print(f"\nSummary: Trained wins={trained_wins}, Random wins={random_wins}, Ties={ties}")
