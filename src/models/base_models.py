"""
Base model implementations for geometry learning experiments.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from typing import Optional, Dict, Any


class FixedGeometryGNN(nn.Module):
    """Baseline model with fixed Euclidean geometry."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 3,
                 dropout: float = 0.0,
                 activation: str = "relu"):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(gnn.GCNConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(gnn.GCNConv(hidden_dim, hidden_dim))
            
        # Output layer
        self.layers.append(gnn.GCNConv(hidden_dim, output_dim))
        
        self.dropout = dropout
        self.activation = self._get_activation(activation)
        
    def _get_activation(self, activation: str):
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, 
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index, edge_weight)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Final layer
        x = self.layers[-1](x, edge_index, edge_weight)
        
        return x
    
    def get_parameters_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": total - trainable
        }


