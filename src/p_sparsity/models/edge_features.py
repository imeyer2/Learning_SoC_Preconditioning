"""
Edge feature encoding for policy network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeFeatureEncoder(nn.Module):
    """
    Encodes edge features from node embeddings and graph structure.
    
    Features include:
    - Node embeddings (source and target)
    - Edge weight
    - Physical similarity (from relaxed input channels)
    - Direction features (from coordinates)
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # Determine feature dimension
        hidden_dim = config.get("hidden_dim", 64)
        edge_feat_config = config.get("edge_features", {})
        
        feat_dim = 2 * hidden_dim  # node_i, node_j
        
        if edge_feat_config.get("use_weight", True):
            feat_dim += 1
        if edge_feat_config.get("use_similarity", True):
            feat_dim += 1
        if edge_feat_config.get("use_direction", True):
            feat_dim += edge_feat_config.get("direction_components", 6)
        
        self.feat_dim = feat_dim
    
    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        node_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute edge features.
        
        Args:
            node_embeddings: (N, hidden_dim) from GNN backbone
            edge_index: (2, E) edge connectivity
            edge_weight: (E,) edge weights
            node_features: (N, F) original node features
            
        Returns:
            edge_features: (E, feat_dim)
        """
        row, col = edge_index
        features = []
        
        # Node embeddings
        node_i = node_embeddings[row]
        node_j = node_embeddings[col]
        features.extend([node_i, node_j])
        
        edge_feat_config = self.config.get("edge_features", {})
        
        # Edge weight
        if edge_feat_config.get("use_weight", True):
            w = edge_weight.unsqueeze(1).abs()
            features.append(w)
        
        # Physical similarity from relaxed input channels
        if edge_feat_config.get("use_similarity", True):
            # Assume first 4 channels are relaxed vectors
            phys_i = node_features[row, :4]
            phys_j = node_features[col, :4]
            sim = F.cosine_similarity(phys_i, phys_j, dim=1).unsqueeze(1)
            features.append(sim)
        
        # Direction features from coordinates
        if edge_feat_config.get("use_direction", True):
            # Assume channels 4:6 are coordinates (x, y)
            ci = node_features[row, 4:6]
            cj = node_features[col, 4:6]
            dxy = ci - cj
            dx = dxy[:, [0]]
            dy = dxy[:, [1]]
            adx = dx.abs()
            ady = dy.abs()
            dn = torch.sqrt(dx * dx + dy * dy + 1e-12)
            udx = dx / dn
            udy = dy / dn
            dir_feat = torch.cat([dx, dy, adx, ady, udx, udy], dim=1)
            features.append(dir_feat)
        
        return torch.cat(features, dim=1)


def build_edge_mlp(config: dict, input_dim: int) -> nn.Module:
    """
    Build MLP for edge logit prediction.
    
    Args:
        config: Model configuration
        input_dim: Input feature dimension
        
    Returns:
        mlp: Edge MLP module
    """
    mlp_config = config.get("edge_mlp", {})
    layers = mlp_config.get("layers", [128, 64, 1])
    activation = mlp_config.get("activation", "relu")
    
    modules = []
    in_dim = input_dim
    
    for i, out_dim in enumerate(layers):
        modules.append(nn.Linear(in_dim, out_dim))
        if i < len(layers) - 1:  # No activation after last layer
            if activation == "relu":
                modules.append(nn.ReLU())
            elif activation == "elu":
                modules.append(nn.ELU())
            elif activation == "gelu":
                modules.append(nn.GELU())
        in_dim = out_dim
    
    return nn.Sequential(*modules)
