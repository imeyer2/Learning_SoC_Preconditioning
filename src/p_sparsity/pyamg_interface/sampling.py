"""
Edge sampling strategies for policy network.

Implements stochastic sampling of edges per row for building C matrix.
Supports both fixed k (same for all nodes) and per-node k values.

Optimized with vectorized operations for faster sampling.
"""

from typing import List, Tuple, Union, Optional
import torch
import torch.nn.functional as F
import numpy as np


def build_row_groups(edge_index: torch.Tensor, num_nodes: int) -> List[torch.Tensor]:
    """
    Group edges by source node for per-row sampling.
    
    Args:
        edge_index: (2, E) edge connectivity
        num_nodes: Number of nodes
        
    Returns:
        row_groups: List of edge index tensors, one per node
    """
    row = edge_index[0].cpu().numpy()
    buckets: List[List[int]] = [[] for _ in range(num_nodes)]
    
    for e, r in enumerate(row):
        buckets[int(r)].append(e)
    
    return [torch.tensor(b, dtype=torch.long) for b in buckets]


def build_row_csr(edge_index: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build CSR-format row pointers for vectorized segment operations.
    
    Args:
        edge_index: (2, E) edge connectivity  
        num_nodes: Number of nodes
        
    Returns:
        row_ptr: (N+1,) CSR row pointers
        perm: (E,) permutation to sort edges by source node
    """
    row = edge_index[0]
    device = row.device
    
    # Sort edges by source node
    perm = torch.argsort(row)
    sorted_row = row[perm]
    
    # Count edges per node
    E = row.numel()
    counts = torch.zeros(num_nodes, dtype=torch.long, device=device)
    counts.scatter_add_(0, sorted_row, torch.ones(E, dtype=torch.long, device=device))
    
    # Build CSR row pointers
    row_ptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
    row_ptr[1:] = torch.cumsum(counts, dim=0)
    
    return row_ptr, perm


def sample_topk_vectorized(
    logits: torch.Tensor,
    row_ptr: torch.Tensor,
    perm: torch.Tensor,
    k: Union[int, torch.Tensor],
    temperature: float = 1.0,
    gumbel: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized sampling using Gumbel-top-k trick.
    
    This is MUCH faster than the sequential version because:
    1. All Gumbel noise is generated in one batch
    2. Top-k per segment is done with vectorized segment operations
    3. Log-probability computed in parallel
    
    Args:
        logits: (E,) edge logits from policy
        row_ptr: (N+1,) CSR row pointers from build_row_csr
        perm: (E,) permutation from build_row_csr
        k: Number of edges to sample per row
        temperature: Temperature for softmax
        gumbel: Whether to use Gumbel noise for sampling
        
    Returns:
        selected_mask: (E,) bool tensor indicating selected edges (in original order)
        logprob_sum: Scalar tensor with sum of log probabilities
    """
    device = logits.device
    E = logits.numel()
    N = row_ptr.numel() - 1
    tau = max(float(temperature), 1e-6)
    
    # Permute logits to sorted order
    sorted_logits = logits[perm] / tau
    
    # Handle per-node k values  
    if isinstance(k, torch.Tensor):
        k_per_node = k.to(device)
    else:
        k_per_node = torch.full((N,), k, dtype=torch.long, device=device)
    
    # Compute segment lengths
    lengths = row_ptr[1:] - row_ptr[:-1]
    
    # Clamp k to available edges per node
    k_actual = torch.minimum(k_per_node, lengths)
    
    # Create node assignment for each edge (which node does each edge belong to)
    node_ids = torch.zeros(E, dtype=torch.long, device=device)
    for i in range(N):
        start, end = row_ptr[i].item(), row_ptr[i+1].item()
        if end > start:
            node_ids[start:end] = i
    
    # Add Gumbel noise for sampling
    if gumbel:
        u = torch.rand(E, device=device)
        g = -torch.log(-torch.log(u + 1e-12) + 1e-12)
        perturbed_logits = sorted_logits + g
    else:
        perturbed_logits = sorted_logits
    
    # For each segment, select top-k by perturbed logits
    selected_sorted = torch.zeros(E, dtype=torch.bool, device=device)
    logp_total = torch.zeros((), dtype=torch.float32, device=device)
    
    # Compute softmax probabilities per segment for log-prob
    # Use segment-wise softmax: exp(x_i) / sum_j exp(x_j) for j in segment
    max_per_seg = torch.segment_reduce(sorted_logits, 'max', lengths=lengths)
    max_per_seg = torch.where(lengths > 0, max_per_seg, torch.zeros_like(max_per_seg))
    
    # Expand max back to edges for numerical stability
    expanded_max = max_per_seg[node_ids]
    exp_logits = torch.exp(sorted_logits - expanded_max)
    
    # Sum exp per segment
    sum_exp = torch.zeros(N, device=device)
    sum_exp.scatter_add_(0, node_ids, exp_logits)
    
    # Probabilities
    probs = exp_logits / (sum_exp[node_ids] + 1e-12)
    
    # Process each segment to select top-k
    # This loop is over nodes but does minimal work per node
    for i in range(N):
        start = row_ptr[i].item()
        end = row_ptr[i+1].item()
        kk = k_actual[i].item()
        
        if kk <= 0 or start >= end:
            continue
        
        # Get top-k indices within this segment
        segment_perturbed = perturbed_logits[start:end]
        if kk >= end - start:
            # Select all
            selected_sorted[start:end] = True
            logp_total = logp_total + torch.log(probs[start:end] + 1e-12).sum()
        else:
            # Select top-k
            _, topk_local = torch.topk(segment_perturbed, kk)
            topk_global = start + topk_local
            selected_sorted[topk_global] = True
            
            # Log prob of selecting these k edges (approximation for without-replacement)
            # For Gumbel-top-k, this is sum of log-probs of selected items
            logp_total = logp_total + torch.log(probs[topk_global] + 1e-12).sum()
    
    # Unpermute to original edge order
    inv_perm = torch.argsort(perm)
    selected_mask = selected_sorted[inv_perm]
    
    return selected_mask, logp_total


def sample_topk_fully_vectorized(
    logits: torch.Tensor,
    row_ptr: torch.Tensor,
    perm: torch.Tensor,
    k: Union[int, torch.Tensor],
    temperature: float = 1.0,
    gumbel: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fully vectorized Gumbel-top-k sampling without any Python loops.
    
    Uses a ranking approach: for each edge, compute its rank within its segment,
    then select edges with rank < k.
    
    Args:
        logits: (E,) edge logits from policy
        row_ptr: (N+1,) CSR row pointers
        perm: (E,) permutation from build_row_csr
        k: Number of edges to sample per row
        temperature: Temperature for softmax
        gumbel: Whether to use Gumbel noise
        
    Returns:
        selected_mask: (E,) bool tensor (original order)
        logprob_sum: Scalar log probability
    """
    device = logits.device
    E = logits.numel()
    N = row_ptr.numel() - 1
    tau = max(float(temperature), 1e-6)
    
    if E == 0:
        return torch.zeros(0, dtype=torch.bool, device=device), torch.tensor(0.0, device=device)
    
    # Permute logits to sorted-by-node order
    sorted_logits = logits[perm] / tau
    
    # Build node_ids: which segment each edge belongs to
    lengths = row_ptr[1:] - row_ptr[:-1]
    node_ids = torch.repeat_interleave(torch.arange(N, device=device), lengths)
    
    # Handle per-node k  
    if isinstance(k, torch.Tensor):
        k_per_node = k.to(device)
    else:
        k_per_node = torch.full((N,), k, dtype=torch.long, device=device)
    
    # Clamp k to segment lengths
    k_actual = torch.minimum(k_per_node, lengths)
    
    # Add Gumbel noise
    if gumbel:
        u = torch.rand(E, device=device)
        g = -torch.log(-torch.log(u + 1e-12) + 1e-12)
        perturbed = sorted_logits + g
    else:
        perturbed = sorted_logits
    
    # Compute rank of each edge within its segment (0 = highest perturbed value)
    # Strategy: negate perturbed, argsort gives ranking
    # But we need per-segment ranking...
    
    # Add large offset per segment to ensure segments don't mix
    # offset = node_id * (max_perturbed + 1) then argsort
    max_val = perturbed.max().item() + 1
    offset_perturbed = -perturbed + node_ids.float() * (max_val * 2)
    
    # Argsort gives global order, then compute rank
    order = torch.argsort(offset_perturbed)
    ranks = torch.zeros(E, dtype=torch.long, device=device)
    ranks[order] = torch.arange(E, device=device)
    
    # Convert global ranks to within-segment ranks
    # segment_start_rank[i] = row_ptr[i]
    segment_start_rank = row_ptr[node_ids]
    within_segment_rank = ranks - segment_start_rank
    
    # Select edges where within_segment_rank < k_actual[node_id]
    k_for_edge = k_actual[node_ids]
    selected_sorted = within_segment_rank < k_for_edge
    
    # Compute log probabilities using segment softmax
    # First compute segment-wise max for numerical stability
    max_per_seg = torch.segment_reduce(sorted_logits, 'max', lengths=lengths)
    # Handle empty segments
    max_per_seg = torch.where(lengths > 0, max_per_seg, torch.zeros_like(max_per_seg))
    expanded_max = max_per_seg[node_ids]
    
    exp_logits = torch.exp(sorted_logits - expanded_max)
    sum_exp = torch.zeros(N, device=device)
    sum_exp.scatter_add_(0, node_ids, exp_logits)
    probs = exp_logits / (sum_exp[node_ids] + 1e-12)
    
    # Sum log-probs of selected edges
    log_probs = torch.log(probs + 1e-12)
    logp_total = (log_probs * selected_sorted.float()).sum()
    
    # Unpermute to original order
    inv_perm = torch.argsort(perm)
    selected_mask = selected_sorted[inv_perm]
    
    return selected_mask, logp_total


def sample_topk_without_replacement(
    logits: torch.Tensor,
    row_groups: List[torch.Tensor],
    k: Union[int, torch.Tensor],
    temperature: float = 1.0,
    gumbel: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample up to k edges per row without replacement using Gumbel-Softmax trick.
    
    For each node i:
      - Sample k (or k[i] if per-node) edges from its outgoing edges sequentially
      - Remove sampled edges from pool (without replacement)
      - Track log probabilities for REINFORCE
    
    Args:
        logits: (E,) edge logits from policy
        row_groups: List of edge indices per node
        k: Number of edges to sample per row. Can be:
           - int: same k for all nodes
           - Tensor (N,): per-node k values
        temperature: Temperature for softmax
        gumbel: Whether to use Gumbel noise for sampling
        
    Returns:
        selected_mask: (E,) bool tensor indicating selected edges
        logprob_sum: Scalar tensor with sum of log probabilities
    """
    device = logits.device
    E = logits.numel()
    selected = torch.zeros(E, dtype=torch.bool, device=device)
    logp_total = torch.zeros((), dtype=torch.float32, device=device)
    
    tau = max(float(temperature), 1e-6)
    
    # Handle per-node k values
    is_per_node_k = isinstance(k, torch.Tensor)
    if is_per_node_k:
        k_values = k.cpu().numpy().astype(int)
    else:
        k_fixed = int(k)
    
    for node_idx, edges_i in enumerate(row_groups):
        if edges_i.numel() and int(edges_i.max()) >= E:
            raise RuntimeError(
                f"row_groups contain edge id {int(edges_i.max())} but logits has E={E}"
            )
        if edges_i.numel() == 0:
            continue
        
        edges_i = edges_i.to(device)
        
        # Get k for this node
        if is_per_node_k:
            k_this_node = int(k_values[node_idx])
        else:
            k_this_node = k_fixed
        
        # Number to sample in this row
        kk = min(k_this_node, edges_i.numel())
        if kk <= 0:
            continue
        
        # Copy local logits
        local_logits = logits[edges_i] / tau
        
        # Sequential sampling without replacement
        remaining = edges_i.clone()
        rem_logits = local_logits.clone()
        
        for _t in range(kk):
            if gumbel:
                # Gumbel-max trick for sampling
                u = torch.rand_like(rem_logits)
                g = -torch.log(-torch.log(u + 1e-12) + 1e-12)
                sample_logits = rem_logits + g
            else:
                sample_logits = rem_logits
            
            # Compute probabilities for log-likelihood
            probs = F.softmax(rem_logits, dim=0)
            
            # Sample (argmax with Gumbel noise gives sample from categorical)
            j = torch.argmax(sample_logits)
            chosen_edge = remaining[j]
            
            # Mark as selected
            selected[chosen_edge] = True
            
            # Accumulate log probability
            logp_total = logp_total + torch.log(probs[j] + 1e-12)
            
            # Remove chosen edge from pool
            keep = torch.ones(remaining.numel(), dtype=torch.bool, device=device)
            keep[j] = False
            remaining = remaining[keep]
            rem_logits = rem_logits[keep]
            
            if remaining.numel() == 0:
                break
    
    return selected, logp_total


def sample_deterministic_topk(
    logits: torch.Tensor,
    row_groups: List[torch.Tensor],
    k: Union[int, torch.Tensor],
) -> torch.Tensor:
    """
    Deterministically select top-k edges per row by logit value.
    
    Used for evaluation (no sampling, no log probabilities).
    
    Args:
        logits: (E,) edge logits from policy
        row_groups: List of edge indices per node
        k: Number of edges to select per row. Can be:
           - int: same k for all nodes
           - Tensor (N,): per-node k values
        
    Returns:
        selected_mask: (E,) bool tensor indicating selected edges
    """
    device = logits.device
    E = logits.numel()
    selected = np.zeros(E, dtype=bool)
    logits_np = logits.cpu().numpy()
    
    # Handle per-node k values
    is_per_node_k = isinstance(k, torch.Tensor)
    if is_per_node_k:
        k_values = k.cpu().numpy().astype(int)
    else:
        k_fixed = int(k)
    
    for node_idx, edges_i in enumerate(row_groups):
        if edges_i.numel() == 0:
            continue
        
        edges = edges_i.numpy()
        
        # Get k for this node
        if is_per_node_k:
            k_this_node = int(k_values[node_idx])
        else:
            k_this_node = k_fixed
        
        kk = min(k_this_node, edges.shape[0])
        if kk <= 0:
            continue
        
        # Get logit values for this row
        vals = logits_np[edges]
        
        # Sort descending
        top_indices = np.argsort(vals)[::-1]
        
        # Take top-k
        top = top_indices[:kk]
        selected[edges[top]] = True
    
    return torch.tensor(selected, dtype=torch.bool, device=device)
