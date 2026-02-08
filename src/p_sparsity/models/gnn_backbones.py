"""
GNN backbone implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class GATBackbone(nn.Module):
    """Graph Attention Network backbone."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        heads_layer1: int = 2,
        heads_layer2: int = 1,
        concat_layer1: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = dropout
        
        self.conv1 = GATConv(
            in_channels,
            hidden_dim,
            heads=heads_layer1,
            concat=concat_layer1,
            dropout=dropout,
        )
        
        conv1_out = hidden_dim * heads_layer1 if concat_layer1 else hidden_dim
        self.conv2 = GATConv(
            conv1_out,
            hidden_dim,
            heads=heads_layer2,
            concat=False,
            dropout=dropout,
        )
        
        self.out_dim = hidden_dim
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            return_attention: If True, also return attention weights from last layer
            
        Returns:
            h: Node embeddings
            attn: (optional) Attention weights if return_attention=True
        """
        x = F.elu(self.conv1(x, edge_index))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        if return_attention:
            x, (edge_index_out, attn_weights) = self.conv2(
                x, edge_index, return_attention_weights=True
            )
            return x, attn_weights
        else:
            x = self.conv2(x, edge_index)
            return x


class GCNBackbone(nn.Module):
    """Graph Convolutional Network backbone."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.out_dim = hidden_dim
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GraphSAGEBackbone(nn.Module):
    """GraphSAGE backbone."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int = 2,
        aggregator: str = "mean",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_dim, aggr=aggregator))
        
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
        
        self.out_dim = hidden_dim
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def build_backbone(config: dict) -> nn.Module:
    """
    Build GNN backbone from configuration.
    
    Args:
        config: Model configuration dict
        
    Returns:
        backbone: GNN backbone module
    """
    backbone_type = config.get("backbone", "gat").lower()
    in_channels = config.get("input_channels", 6)
    hidden_dim = config.get("hidden_dim", 64)
    
    if backbone_type == "gat":
        gat_config = config.get("gat", {})
        return GATBackbone(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            heads_layer1=gat_config.get("heads_layer1", 2),
            heads_layer2=gat_config.get("heads_layer2", 1),
            concat_layer1=gat_config.get("concat_layer1", True),
            dropout=gat_config.get("dropout", 0.0),
        )
    elif backbone_type == "gcn":
        gcn_config = config.get("gcn", {})
        return GCNBackbone(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=gcn_config.get("num_layers", 2),
            dropout=gcn_config.get("dropout", 0.0),
        )
    elif backbone_type == "graphsage":
        sage_config = config.get("graphsage", {})
        return GraphSAGEBackbone(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=sage_config.get("num_layers", 2),
            aggregator=sage_config.get("aggregator", "mean"),
            dropout=sage_config.get("dropout", 0.0),
        )
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")
