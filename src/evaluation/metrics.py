"""
Geometric evaluation metrics.
"""
import torch
import numpy as np
from scipy import stats
from typing import List, Dict, Any


class GeometricMetrics:
    """Compute geometric properties of learned metrics."""
    
    @staticmethod
    def effective_curvature(edge_weights: torch.Tensor,
                           edge_index: torch.Tensor,
                           method: str = "ollivier") -> float:
        w = edge_weights.cpu().numpy()
        src, dst = edge_index.cpu().numpy()
        
        if method == "ollivier":
            curvatures = []
            unique_nodes = np.unique(src)
            
            for node in unique_nodes:
                node_edges = w[src == node]
                if len(node_edges) > 1:
                    mean_w = np.mean(node_edges)
                    for w_ij in node_edges:
                        curv = 1 - (w_ij / mean_w)
                        curvatures.append(curv)
            
            return np.mean(curvatures) if curvatures else 0.0
        
        elif method == "forman":
            degree_src = np.bincount(src, minlength=len(w))
            degree_dst = np.bincount(dst, minlength=len(w))
            
            forman_curv = 2 - (degree_src[src] + degree_dst[dst]) / (w + 1e-8)
            return np.mean(forman_curv)
        
        else:
            raise ValueError(f"Unknown curvature method: {method}")
    
    @staticmethod
    def metric_complexity(edge_weights: torch.Tensor,
                         method: str = "entropy") -> float:
        w = edge_weights.cpu().numpy()
        
        if method == "entropy":
            w_norm = w / (w.sum() + 1e-8)
            entropy = -np.sum(w_norm * np.log(w_norm + 1e-8))
            return entropy
        
        elif method == "variance":
            return np.var(w)
        
        elif method == "l2":
            return np.mean(w ** 2)
        
        else:
            raise ValueError(f"Unknown complexity method: {method}")
