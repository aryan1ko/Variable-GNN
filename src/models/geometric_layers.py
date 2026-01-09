"""
Custom geometric layers for metric learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from typing import Optional, Tuple


class MetricAwareGCNConv(gnn.GCNConv):
    """GCN convolution with explicit metric awareness."""
    
    def forward(self, x, edge_index, edge_weight=None, metric=None):
        if metric is not None:
            # Use metric to modulate edge weights
            if edge_weight is None:
                edge_weight = metric
            else:
                edge_weight = edge_weight * metric
                
        return super().forward(x, edge_index, edge_weight)


class RiemannianConv(nn.Module):
    """Simple Riemannian-inspired convolution layer."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.metric_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x, edge_index, edge_metric=None):
        x = self.linear(x)
        
        if edge_metric is not None:
            # Simple metric-aware aggregation
            row, col = edge_index
            x_j = x[col]
            
            if edge_metric.dim() == 1:
                # Edge metric weights
                weights = edge_metric.unsqueeze(1)
            else:
                weights = edge_metric
            
            aggregated = torch.zeros_like(x)
            aggregated = aggregated.index_add_(0, row, x_j * weights)
            
            # Combine with original features
            degree = torch.bincount(row, minlength=x.size(0)).float().unsqueeze(1)
            aggregated = aggregated / degree.clamp(min=1)
            
            x = x + self.metric_scale * aggregated
            
        return x