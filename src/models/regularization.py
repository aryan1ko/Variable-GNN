"""
Geometric regularization functions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


def curvature_regularization(edge_weights: torch.Tensor,
                            edge_index: torch.Tensor,
                            method: str = "ollivier") -> torch.Tensor:
    """Compute curvature regularization term."""
    if method == "ollivier":
        # Simple Ollivier-Ricci curvature approximation
        src, dst = edge_index
        w = edge_weights
        
        # Compute average neighbor weight for each node
        unique_nodes = torch.unique(src)
        node_avg = torch.zeros_like(w)
        
        for node in unique_nodes:
            mask = src == node
            if mask.any():
                node_avg[mask] = w[mask].mean()
        
        # Curvature: 1 - w_ij / avg_neighbor_weight
        curvature = 1 - w / (node_avg + 1e-8)
        return torch.mean(curvature ** 2)
    
    elif method == "forman":
        # Forman-Ricci curvature approximation
        src, dst = edge_index
        w = edge_weights
        
        # Simple Forman curvature: 2 - sum_{f∋e} w(f)/w(e)
        # For graph edges, approximate with node degrees
        degree_src = torch.bincount(src, minlength=w.size(0)).float()
        degree_dst = torch.bincount(dst, minlength=w.size(0)).float()
        
        forman_curv = 2 - (degree_src[src] + degree_dst[dst]) / (w + 1e-8)
        return torch.mean(forman_curv ** 2)
    
    else:
        raise ValueError(f"Unknown curvature method: {method}")


def smoothness_regularization(edge_weights: torch.Tensor,
                             edge_index: torch.Tensor,
                             order: int = 1) -> torch.Tensor:
    """Encourage smooth variation of metrics across the graph."""
    src, dst = edge_index
    
    if order == 1:
        # First-order smoothness
        diff = edge_weights[src] - edge_weights[dst]
        return torch.mean(diff ** 2)
    elif order == 2:
        # Second-order smoothness (Laplacian)
        w = edge_weights
        src_degrees = torch.bincount(src, minlength=w.size(0)).float()
        dst_degrees = torch.bincount(dst, minlength=w.size(0)).float()
        
        laplacian = w[src] * (1/src_degrees[src] + 1/dst_degrees[dst])
        return torch.mean(laplacian ** 2)
    
    else:
        raise ValueError(f"Unsupported smoothness order: {order}")


def volume_regularization(edge_weights: torch.Tensor,
                         method: str = "log") -> torch.Tensor:
    """Control the total 'volume' of the metric."""
    if method == "log":
        # Penalize deviation from log-normal distribution
        return torch.mean(torch.log(edge_weights + 1e-8) ** 2)
    
    elif method == "l2":
        # Simple L2 regularization on weights
        return torch.mean(edge_weights ** 2)
    
    elif method == "entropy":
        # Shannon entropy regularization
        w_norm = edge_weights / (edge_weights.sum() + 1e-8)
        entropy = -torch.sum(w_norm * torch.log(w_norm + 1e-8))
        return entropy
    
    else:
        raise ValueError(f"Unknown volume method: {method}")


def deviation_regularization(current_weights: torch.Tensor,
                           initial_weights: torch.Tensor,
                           method: str = "l2") -> torch.Tensor:
    """Penalize deviation from initial metric."""
    if method == "l2":
        return torch.mean((current_weights - initial_weights) ** 2)
    
    elif method == "relative":
        rel_diff = (current_weights - initial_weights) / (initial_weights + 1e-8)
        return torch.mean(rel_diff ** 2)
    
    elif method == "log_ratio":
        log_ratio = torch.log(current_weights / (initial_weights + 1e-8))
        return torch.mean(log_ratio ** 2)
    
    else:
        raise ValueError(f"Unknown deviation method: {method}")