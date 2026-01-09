"""
Custom optimizers for geometry learning.
"""
import torch
import torch.optim as optim
from typing import List, Dict, Any


class GeometryAwareAdam(optim.Adam):
    """Adam optimizer with different learning rates for geometry parameters."""
    
    def __init__(self, params, geometry_params=None, geometry_lr=0.05, **kwargs):
        if geometry_params is not None:
            param_groups = [
                {'params': geometry_params, 'lr': geometry_lr},
                {'params': [p for p in params if p not in geometry_params]}
            ]
            param_groups[1].update(kwargs)
        else:
            param_groups = [{'params': params}]
            param_groups[0].update(kwargs)
        
        super().__init__(param_groups)


class MetricGradientClipper:
    """Special gradient clipping for metric parameters."""
    
    def __init__(self, max_norm: float = 1.0, metric_max_norm: float = 0.1):
        self.max_norm = max_norm
        self.metric_max_norm = metric_max_norm
    
    def clip(self, model):
        # Clip network gradients
        torch.nn.utils.clip_grad_norm_(
            [p for n, p in model.named_parameters() if 'edge_weights' not in n],
            max_norm=self.max_norm
        )
        
        # Clip metric gradients more aggressively
        for n, p in model.named_parameters():
            if 'edge_weights' in n and p.grad is not None:
                torch.nn.utils.clip_grad_norm_([p], max_norm=self.metric_max_norm)