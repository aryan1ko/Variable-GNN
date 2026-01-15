#!/usr/bin/env python3
"""
Analyze when geometry learning helps vs hurts.
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/src')

from models.base_models import FixedGeometryGNN, LearnableMetricGNN
from data.datasets import SyntheticManifoldDataset, create_data_splits

def analyze_dataset_difficulty():
    """Test on datasets of varying difficulty."""
    
    print("📊 ANALYZING WHEN GEOMETRY LEARNING HELPS")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different noise levels and dataset complexities
    noise_levels = [0.05, 0.1, 0.15, 0.2, 0.25]
    n_neighbors_list = [5, 10, 15, 20]
    
    results = {'noise': {}, 'connectivity': {}}
    
    # 1. Vary noise level
    print("\n1. EFFECT OF NOISE LEVEL")
    print("-" * 40)
    
    for noise in noise_levels:
        print(f"\nNoise level: {noise}")
        
        data = SyntheticManifoldDataset.two_moons(n_samples=500, noise=noise)
        data = create_data_splits(data)
        data = data.to(device)
        
        # Train models
        fixed_acc = train_and_evaluate('fixed', data, device)
        learnable_acc = train_and_evaluate('learnable', data, device, reg_weight=0.1)
        
        improvement = learnable_acc - fixed_acc
        
        results['noise'][noise] = {
            'fixed': fixed_acc,
            'learnable': learnable_acc,
            'improvement': improvement
        }
        
        print(f"  Fixed: {fixed_acc:.3f}, Learnable: {learnable_acc:.3f}, "
              f"Improvement: {improvement:+.3f}")
    
    # 2. Vary graph connectivity
    print("\n\n2. EFFECT OF GRAPH CONNECTIVITY")
    print("-" * 40)
    
    for n_neighbors in n_neighbors_list:
        print(f"\nNeighbors: {n_neighbors}")
        
        data = SyntheticManifoldDataset.two_moons(n_samples=500, noise=0.15, 
                                                 n_neighbors=n_neighbors)
        data = create_data_splits(data)
        data = data.to(device)
        
        fixed_acc = train_and_evaluate('fixed', data, device)
        learnable_acc = train_and_evaluate('learnable', data, device, reg_weight=0.1)
        
        improvement = learnable_acc - fixed_acc
        
        results['connectivity'][n_neighbors] = {
            'fixed': fixed_acc,
            'learnable': learnable_acc,
            'improvement': improvement
        }
        
        print(f"  Fixed: {fixed_acc:.3f}, Learnable: {learnable_acc:.3f}, "
              f"Improvement: {improvement:+.3f}")
    
    # Plot results
    plot_analysis_results(results, noise_levels, n_neighbors_list)
    
    return results

def train_and_evaluate(model_type, data, device, reg_weight=0.1, epochs=5000):
    """Train a model and return test accuracy."""
    input_dim = data.x.shape[1]
    output_dim = len(data.y.unique())
    num_edges = data.edge_index.shape[1]
    
    if model_type == 'fixed':
        model = FixedGeometryGNN(input_dim, 64, output_dim, 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    else:
        model = LearnableMetricGNN(input_dim, 64, output_dim, num_edges, 3)
        optimizer = torch.optim.Adam([
            {'params': model.layers.parameters(), 'lr': 0.01},
            {'params': [model.edge_weights], 'lr': 0.01}
        ])
    
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        
        if model_type == 'learnable':
            reg_loss = model.geometric_regularization('smoothness', data.edge_index)
            loss = loss + reg_weight * reg_loss
        
        loss.backward()
        optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[data.test_mask].argmax(dim=1)
        acc = (pred == data.y[data.test_mask]).float().mean()
    
    return acc.item()


if __name__ == "__main__":
    analyze_dataset_difficulty()