"""
AMG Edge Policy network.

Main policy model that outputs edge logits for selecting strong connections,
optionally learns B candidates for near-nullspace, and optionally predicts
per-node k values (number of edges to select per node).
"""

from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn_backbones import build_backbone
from .edge_features import EdgeFeatureEncoder, build_edge_mlp


class AMGEdgePolicy(nn.Module):
    """
    Policy network for AMG edge selection.
    
    Architecture:
    1. GNN backbone processes node features -> node embeddings
    2. Edge encoder computes edge features from node embeddings
    3. Edge MLP predicts logits for each edge
    4. Optional: B-head predicts near-nullspace candidates
    5. Optional: K-head predicts per-node k values (number of edges to select)
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # GNN backbone
        self.backbone = build_backbone(config)
        
        # Edge feature encoder
        self.edge_encoder = EdgeFeatureEncoder(config)
        
        # Edge MLP for logit prediction
        self.edge_mlp = build_edge_mlp(config, self.edge_encoder.feat_dim)
        
        # Optional B-candidate learning
        self.learn_B = config.get("learn_B", True)
        if self.learn_B:
            self.B_extra = config.get("B_extra", 2)
            self.B_head = self._build_B_head(config)
        else:
            self.B_extra = 0
            self.B_head = None
        
        # Optional learnable k (per-node number of edges to select)
        self.learn_k = config.get("learn_k", False)
        if self.learn_k:
            self.k_min = config.get("k_min", 1)
            self.k_max = config.get("k_max", 8)
            self.k_head = self._build_k_head(config)
        else:
            self.k_head = None
    
    def _build_B_head(self, config: dict) -> nn.Module:
        """Build MLP for B-candidate prediction."""
        hidden_dim = self.backbone.out_dim
        B_config = config.get("B_head", {})
        layers = B_config.get("layers", [64])
        activation = B_config.get("activation", "relu")
        
        modules = []
        in_dim = hidden_dim
        
        for out_dim in layers:
            modules.append(nn.Linear(in_dim, out_dim))
            if activation == "relu":
                modules.append(nn.ReLU())
            elif activation == "elu":
                modules.append(nn.ELU())
            in_dim = out_dim
        
        # Final layer to B_extra dimensions
        modules.append(nn.Linear(in_dim, self.B_extra))
        
        return nn.Sequential(*modules)
    
    def _build_k_head(self, config: dict) -> nn.Module:
        """
        Build MLP for per-node k prediction.
        
        Outputs a scalar per node that gets mapped to [k_min, k_max] via sigmoid.
        """
        hidden_dim = self.backbone.out_dim
        k_config = config.get("k_head", {})
        layers = k_config.get("layers", [32])
        activation = k_config.get("activation", "relu")
        
        modules = []
        in_dim = hidden_dim
        
        for out_dim in layers:
            modules.append(nn.Linear(in_dim, out_dim))
            if activation == "relu":
                modules.append(nn.ReLU())
            elif activation == "elu":
                modules.append(nn.ELU())
            in_dim = out_dim
        
        # Final layer outputs 1 value per node (logit for k)
        modules.append(nn.Linear(in_dim, 1))
        
        return nn.Sequential(*modules)
    
    def _compute_k_values(self, h: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-node k values from node embeddings.
        
        Uses soft discretization: sigmoid maps to [k_min, k_max], then we 
        use straight-through estimator for discrete k during forward pass.
        
        Args:
            h: (N, D) node embeddings
            temperature: Temperature for sigmoid (lower = sharper discretization)
            
        Returns:
            k_continuous: (N,) continuous k values in [k_min, k_max]
            k_discrete: (N,) discretized k values (integers)
        """
        # Get raw logits
        k_logits = self.k_head(h).squeeze(-1)  # (N,)
        
        # Map to [k_min, k_max] via sigmoid
        k_range = self.k_max - self.k_min
        k_continuous = self.k_min + k_range * torch.sigmoid(k_logits / temperature)
        
        # Discretize with straight-through estimator
        k_discrete = torch.round(k_continuous).long()
        k_discrete = torch.clamp(k_discrete, self.k_min, self.k_max)
        
        # Straight-through: gradient flows through continuous, but forward uses discrete
        k_ste = k_continuous + (k_discrete.float() - k_continuous).detach()
        
        return k_continuous, k_discrete, k_ste
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        return_internals: bool = False,
        k_temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Forward pass.
        
        Args:
            x: (N, F) node features
            edge_index: (2, E) edge connectivity
            edge_weight: (E,) edge weights
            return_internals: If True, also return attention weights and embeddings
            k_temperature: Temperature for k-head sigmoid (lower = sharper)
            
        Returns:
            Dict containing:
                - edge_logits: (E,) logits for edge selection
                - B_extra: (N, B_extra) B-candidates if learn_B else None
                - k_per_node: (N,) per-node k values if learn_k else None  
                - k_continuous: (N,) continuous k for loss computation if learn_k else None
                - internals: Dict with 'attention_weights', 'node_embeddings' if return_internals
        """
        # 1. GNN backbone
        attention_weights = None
        if return_internals and hasattr(self.backbone, 'forward'):
            # Try to get attention weights from GAT backbone
            try:
                result = self.backbone(x, edge_index, return_attention=True)
                if isinstance(result, tuple):
                    h, attention_weights = result
                else:
                    h = result
            except TypeError:
                # Backbone doesn't support return_attention
                h = self.backbone(x, edge_index)
        else:
            h = self.backbone(x, edge_index)
        
        # Store node embeddings for diagnostics
        node_embeddings = h if return_internals else None
        
        # 2. Edge features
        edge_feat = self.edge_encoder(h, edge_index, edge_weight, x)
        
        # 3. Edge logits
        logits = self.edge_mlp(edge_feat).squeeze(-1)
        
        # 4. Mask diagonal edges (if present)
        row, col = edge_index
        diag = (row == col)
        logits = logits.masked_fill(diag, -1e9)
        
        # 5. Optional B-candidates
        B_extra = None
        if self.learn_B and self.B_head is not None:
            B_extra = self.B_head(h)
        
        # 6. Optional per-node k values
        k_per_node = None
        k_continuous = None
        k_ste = None
        if self.learn_k and self.k_head is not None:
            k_continuous, k_per_node, k_ste = self._compute_k_values(h, temperature=k_temperature)
        
        # Build output dict
        output = {
            'edge_logits': logits,
            'B_extra': B_extra,
            'k_per_node': k_per_node,
            'k_continuous': k_continuous,
            'k_ste': k_ste,  # For straight-through gradient
        }
        
        if return_internals:
            output['internals'] = {
                'attention_weights': attention_weights,
                'node_embeddings': node_embeddings,
            }
        
        return output
    
    def forward_legacy(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        return_internals: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Legacy forward pass for backward compatibility.
        Returns tuple (logits, B_extra) or (logits, B_extra, internals).
        """
        output = self.forward(x, edge_index, edge_weight, return_internals)
        
        if return_internals:
            return output['edge_logits'], output['B_extra'], output.get('internals', {})
        return output['edge_logits'], output['B_extra']


def build_policy_from_config(config: dict) -> AMGEdgePolicy:
    """
    Build AMG policy from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        policy: AMGEdgePolicy instance
    """
    return AMGEdgePolicy(config)
