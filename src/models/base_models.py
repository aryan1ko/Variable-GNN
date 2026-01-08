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


class LearnableMetricGNN(nn.Module):
    """GNN with learnable edge metrics."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_edges: int,
                 num_layers: int = 3,
                 dropout: float = 0.0,
                 activation: str = "relu",
                 init_method: str = "uniform",
                 parameterization: str = "edge_weights"):
        super().__init__()
        
        # Network parameters
        self.layers = nn.ModuleList()
        self.layers.append(gnn.GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.layers.append(gnn.GCNConv(hidden_dim, hidden_dim))
            
        self.layers.append(gnn.GCNConv(hidden_dim, output_dim))
        
        self.dropout = dropout
        self.activation = self._get_activation(activation)
        
        # Geometry parameters
        self.parameterization = parameterization
        self._init_geometry_parameters(num_edges, init_method)
        
        # Store initial values for regularization
        self.register_buffer('initial_edge_weights', 
                           self.edge_weights.detach().clone())
        
    def _init_geometry_parameters(self, num_edges: int, init_method: str):
        if self.parameterization == "edge_weights":
            if init_method == "uniform":
                init_val = torch.ones(num_edges)
            elif init_method == "random":
                init_val = torch.randn(num_edges) * 0.1 + 1.0
            elif init_method == "small":
                init_val = torch.ones(num_edges) * 0.1
            else:
                raise ValueError(f"Unknown init method: {init_method}")
                
            self.edge_weights = nn.Parameter(init_val)
            
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")
    
    def _get_activation(self, activation: str):
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid()
        }
        return activations.get(activation, nn.ReLU())
    
    def get_metric(self) -> torch.Tensor:
        if self.parameterization == "edge_weights":
            return F.softplus(self.edge_weights)
        else:
            raise NotImplementedError()
    
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        edge_weight = self.get_metric()
        
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index, edge_weight)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        x = self.layers[-1](x, edge_index, edge_weight)
        
        return x
    
    def geometric_regularization(self,
                                 reg_type: str = "deviation",
                                 edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if reg_type == "deviation":
            current = self.get_metric()
            initial = F.softplus(self.initial_edge_weights)
            return torch.mean((current - initial) ** 2)
            
        elif reg_type == "smoothness":
            if edge_index is None:
                return torch.tensor(0.0, device=self.edge_weights.device)
                
            current = self.get_metric()
            src, dst = edge_index
            diff = current[src] - current[dst]
            return torch.mean(diff ** 2)
            
        elif reg_type == "curvature":
            if edge_index is None:
                return torch.tensor(0.0, device=self.edge_weights.device)
                
            current = self.get_metric()
            src, dst = edge_index
            mean_weights = torch.zeros(current.shape[0], device=current.device)
            mean_weights = mean_weights.index_add_(0, src, current) / \
                          torch.bincount(src, minlength=current.shape[0]).float().clamp(min=1)
            
            curv = 1 - (current / (mean_weights[src] + 1e-8))
            return torch.mean(curv ** 2)
            
        elif reg_type == "volume":
            current = self.get_metric()
            return torch.mean(torch.log(current + 1e-8) ** 2)
            
        else:
            return torch.tensor(0.0, device=self.edge_weights.device)
    
    def get_parameters_count(self) -> Dict[str, int]:
        network_params = sum(p.numel() for p in self.layers.parameters())
        geometry_params = self.edge_weights.numel()
        
        total = network_params + geometry_params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": total - trainable,
            "network": network_params,
            "geometry": geometry_params
        }


class FrozenMetricGNN(LearnableMetricGNN):
    """Ablation: Same as LearnableMetricGNN but geometry frozen."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_weights.requires_grad = False
    
    def get_metric(self) -> torch.Tensor:
        return F.softplus(self.edge_weights.detach())