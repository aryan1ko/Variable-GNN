#!/usr/bin/env python3
"""
Debug script to investigate why learnable metric is underperforming.
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/src')

from models.base_models import FixedGeometryGNN, LearnableMetricGNN
from data.datasets import SyntheticManifoldDataset, create_data_splits

def analyze_metric_behavior():
    """Analyze what's happening with the learnable metric."""
    
    print("🔍 DEBUGGING LEARNABLE METRIC PERFORMANCE")
    print("=" * 60)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset
    data = SyntheticManifoldDataset.two_moons(n_samples=500, noise=0.15)
    data = create_data_splits(data)
    data = data.to(device)
    
    # Initialize models
    input_dim = data.x.shape[1]
    output_dim = len(data.y.unique())
    num_edges = data.edge_index.shape[1]
    
    fixed_model = FixedGeometryGNN(input_dim, 64, output_dim, 3)
    learnable_model = LearnableMetricGNN(input_dim, 64, output_dim, num_edges, 3)
    
    fixed_model = fixed_model.to(device)
    learnable_model = learnable_model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    
    # Train fixed model
    print("\n1. TRAINING FIXED GEOMETRY MODEL")
    fixed_optimizer = torch.optim.Adam(fixed_model.parameters(), lr=0.01)
    train_model(fixed_model, data, fixed_optimizer, criterion, "Fixed", device)
    
    # Train learnable model with different regularization strengths
    print("\n2. TRAINING LEARNABLE METRIC MODEL WITH DIFFERENT REGULARIZATION")
    
    reg_weights = [0.0, 0.001, 0.01, 0.1, 1.0]
    results = {}
    
    for reg_weight in reg_weights:
        print(f"\n   Regularization weight: {reg_weight}")
        
        # Reinitialize model
        learnable_model = LearnableMetricGNN(input_dim, 64, output_dim, num_edges, 3)
        learnable_model = learnable_model.to(device)
        
        # Separate learning rates for network and geometry
        optimizer = torch.optim.Adam([
            {'params': learnable_model.layers.parameters(), 'lr': 0.01},
            {'params': [learnable_model.edge_weights], 'lr': 0.05}
        ])
        
        # Train
        history = train_model(learnable_model, data, optimizer, criterion, 
                             f"Learnable (λ={reg_weight})", device, 
                             reg_weight=reg_weight)
        results[reg_weight] = history
        
        # Analyze final metric
        final_metric = learnable_model.get_metric()
        print(f"     Metric stats: mean={final_metric.mean():.3f}, "
              f"std={final_metric.std():.3f}, "
              f"min={final_metric.min():.3f}, max={final_metric.max():.3f}")
    
    # Plot comparison
    plot_comparison(results)
    
    return results


if __name__ == "__main__":
    analyze_metric_behavior()