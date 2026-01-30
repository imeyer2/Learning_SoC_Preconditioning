"""
AMG Edge Policy network.

Main policy model that outputs edge logits for selecting strong connections,
and optionally learns B candidates for near-nullspace.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn

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
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: (N, F) node features
            edge_index: (2, E) edge connectivity
            edge_weight: (E,) edge weights
            
        Returns:
            edge_logits: (E,) logits for edge selection
            B_extra: (N, B_extra) B-candidates if learn_B else None
        """
        # 1. GNN backbone
        h = self.backbone(x, edge_index)
        
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
        
        return logits, B_extra


def build_policy_from_config(config: dict) -> AMGEdgePolicy:
    """
    Build AMG policy from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        policy: AMGEdgePolicy instance
    """
    return AMGEdgePolicy(config)
