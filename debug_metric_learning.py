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

def train_model(model, data, optimizer, criterion, name, device, reg_weight=0.0, epochs=50):
    """Train a model and return history."""
    history = {'train_acc': [], 'val_acc': [], 'test_acc': []}
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        
        # Add regularization for learnable metric
        if hasattr(model, 'geometric_regularization'):
            reg_loss = model.geometric_regularization('deviation', data.edge_index)
            loss = loss + reg_weight * reg_loss
        
        loss.backward()
        optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            
            for mask_name, mask in [('train', data.train_mask), 
                                   ('val', data.val_mask), 
                                   ('test', data.test_mask)]:
                pred = out[mask].argmax(dim=1)
                acc = (pred == data.y[mask]).float().mean()
                history[f'{mask_name}_acc'].append(acc.item())
    
    # Print final accuracies
    print(f"     Final - Train: {history['train_acc'][-1]:.3f}, "
          f"Val: {history['val_acc'][-1]:.3f}, "
          f"Test: {history['test_acc'][-1]:.3f}")
    
    return history

def plot_comparison(results):
    """Plot comparison of different regularization strengths."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Training accuracy
    for reg_weight, history in results.items():
        axes[0, 0].plot(history['train_acc'], 
                       label=f'λ={reg_weight}', 
                       alpha=0.8, linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Training Accuracy vs Regularization')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Test accuracy
    for reg_weight, history in results.items():
        axes[0, 1].plot(history['test_acc'], 
                       label=f'λ={reg_weight}', 
                       alpha=0.8, linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Test Accuracy vs Regularization')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Final test accuracy vs regularization
    reg_weights = list(results.keys())
    final_test_accs = [history['test_acc'][-1] for history in results.values()]
    
    axes[1, 0].plot(reg_weights, final_test_accs, 'o-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Regularization Weight (λ)')
    axes[1, 0].set_ylabel('Final Test Accuracy')
    axes[1, 0].set_title('Optimal Regularization Weight')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Overfitting gap (train - test)
    overfitting_gaps = [history['train_acc'][-1] - history['test_acc'][-1] 
                       for history in results.values()]
    
    axes[1, 1].plot(reg_weights, overfitting_gaps, 's-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Regularization Weight (λ)')
    axes[1, 1].set_ylabel('Overfitting Gap (Train - Test)')
    axes[1, 1].set_title('Overfitting vs Regularization')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('debug_regularization_effect.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    analyze_metric_behavior()