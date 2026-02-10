#!/usr/bin/env python
"""Quick test to verify gradient flow through sampling function."""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from p_sparsity.pyamg_interface.sampling import sample_topk_fully_vectorized, build_row_csr

def test_sampling_gradients():
    print("Testing gradient flow through sample_topk_fully_vectorized...")
    
    # Create simple logits with gradients
    torch.manual_seed(42)
    logits = torch.randn(20, requires_grad=True)
    edge_index = torch.tensor([
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4],
        [1, 2, 3, 0, 2, 4, 0, 1, 3, 4, 0, 2, 4, 1, 0, 1, 2, 3, 0, 1]
    ])
    num_nodes = 5
    row_ptr, perm = build_row_csr(edge_index, num_nodes)
    
    # Sample
    selected_mask, logp_sum = sample_topk_fully_vectorized(
        logits, row_ptr, perm, k=2, temperature=1.0, gumbel=True
    )
    
    print(f'  logits.requires_grad: {logits.requires_grad}')
    print(f'  logp_sum.requires_grad: {logp_sum.requires_grad}')
    print(f'  logp_sum value: {logp_sum.item():.4f}')
    
    # Backprop
    logp_sum.backward()
    
    print(f'  logits.grad is None: {logits.grad is None}')
    if logits.grad is not None:
        print(f'  logits.grad norm: {logits.grad.norm().item():.6f}')
        nonzero = (logits.grad != 0).sum().item()
        print(f'  logits.grad nonzero count: {nonzero}/{logits.numel()}')
        
        if nonzero > 0:
            print('  ✅ PASS: Gradients flow through sampling')
            return True
    
    print('  ❌ FAIL: No gradients from sampling')
    return False


if __name__ == "__main__":
    test_sampling_gradients()
