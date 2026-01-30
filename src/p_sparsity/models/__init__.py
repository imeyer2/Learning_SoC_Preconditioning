"""
Model module for P-Sparsity.

Provides GNN-based policy networks with swappable backbones.
"""

from .amg_policy import AMGEdgePolicy, build_policy_from_config
from .gnn_backbones import GATBackbone, GCNBackbone, GraphSAGEBackbone
from .edge_features import EdgeFeatureEncoder

__all__ = [
    "AMGEdgePolicy",
    "build_policy_from_config",
    "GATBackbone",
    "GCNBackbone",
    "GraphSAGEBackbone",
    "EdgeFeatureEncoder",
]
