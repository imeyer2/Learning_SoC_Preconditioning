"""
Edge sampling strategies for policy network.

Implements stochastic sampling of edges per row for building C matrix.
"""

from typing import List, Tuple
import torch
import torch.nn.functional as F


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


def sample_topk_without_replacement(
    logits: torch.Tensor,
    row_groups: List[torch.Tensor],
    k: int,
    temperature: float = 1.0,
    gumbel: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample up to k edges per row without replacement using Gumbel-Softmax trick.
    
    For each node i:
      - Sample k edges from its outgoing edges sequentially
      - Remove sampled edges from pool (without replacement)
      - Track log probabilities for REINFORCE
    
    Args:
        logits: (E,) edge logits from policy
        row_groups: List of edge indices per node
        k: Number of edges to sample per row
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
    
    for edges_i in row_groups:
        if edges_i.numel() and int(edges_i.max()) >= E:
            raise RuntimeError(
                f"row_groups contain edge id {int(edges_i.max())} but logits has E={E}"
            )
        if edges_i.numel() == 0:
            continue
        
        edges_i = edges_i.to(device)
        
        # Number to sample in this row
        kk = min(k, edges_i.numel())
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
    k: int,
) -> torch.Tensor:
    """
    Deterministically select top-k edges per row by logit value.
    
    Used for evaluation (no sampling, no log probabilities).
    
    Args:
        logits: (E,) edge logits from policy
        row_groups: List of edge indices per node
        k: Number of edges to select per row
        
    Returns:
        selected_mask: (E,) bool tensor indicating selected edges
    """
    import numpy as np
    
    device = logits.device
    E = logits.numel()
    selected = np.zeros(E, dtype=bool)
    logits_np = logits.cpu().numpy()
    
    for edges_i in row_groups:
        if edges_i.numel() == 0:
            continue
        
        edges = edges_i.numpy()
        kk = min(k, edges.shape[0])
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
