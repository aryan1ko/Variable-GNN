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

