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

def plot_analysis_results(results, noise_levels, n_neighbors_list):
    """Plot analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Noise vs Accuracy
    fixed_accs = [results['noise'][n]['fixed'] for n in noise_levels]
    learnable_accs = [results['noise'][n]['learnable'] for n in noise_levels]
    
    axes[0, 0].plot(noise_levels, fixed_accs, 'o-', label='Fixed', linewidth=2)
    axes[0, 0].plot(noise_levels, learnable_accs, 's-', label='Learnable', linewidth=2)
    axes[0, 0].set_xlabel('Noise Level')
    axes[0, 0].set_ylabel('Test Accuracy')
    axes[0, 0].set_title('Effect of Noise on Performance')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Noise vs Improvement
    improvements = [results['noise'][n]['improvement'] for n in noise_levels]
    
    axes[0, 1].plot(noise_levels, improvements, 'o-', linewidth=2, color='purple')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Noise Level')
    axes[0, 1].set_ylabel('Improvement (Learnable - Fixed)')
    axes[0, 1].set_title('When Does Geometry Learning Help?')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Fill area where improvement > 0
    x = np.array(noise_levels)
    y = np.array(improvements)
    axes[0, 1].fill_between(x, y, where=y>0, color='green', alpha=0.3)
    axes[0, 1].fill_between(x, y, where=y<0, color='red', alpha=0.3)
    
    # Plot 3: Connectivity vs Accuracy
    fixed_accs_c = [results['connectivity'][n]['fixed'] for n in n_neighbors_list]
    learnable_accs_c = [results['connectivity'][n]['learnable'] for n in n_neighbors_list]
    
    axes[1, 0].plot(n_neighbors_list, fixed_accs_c, 'o-', label='Fixed', linewidth=2)
    axes[1, 0].plot(n_neighbors_list, learnable_accs_c, 's-', label='Learnable', linewidth=2)
    axes[1, 0].set_xlabel('Number of Neighbors')
    axes[1, 0].set_ylabel('Test Accuracy')
    axes[1, 0].set_title('Effect of Graph Connectivity')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Connectivity vs Improvement
    improvements_c = [results['connectivity'][n]['improvement'] for n in n_neighbors_list]
    
    axes[1, 1].plot(n_neighbors_list, improvements_c, 'o-', linewidth=2, color='purple')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Number of Neighbors')
    axes[1, 1].set_ylabel('Improvement (Learnable - Fixed)')
    axes[1, 1].set_title('Optimal Connectivity for Geometry Learning')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Fill area where improvement > 0
    x = np.array(n_neighbors_list)
    y = np.array(improvements_c)
    axes[1, 1].fill_between(x, y, where=y>0, color='green', alpha=0.3)
    axes[1, 1].fill_between(x, y, where=y<0, color='red', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('when_geometry_helps.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📈 KEY FINDINGS")
    print("=" * 60)
    
    # Find conditions where learnable helps
    helping_noise = [n for n in noise_levels if results['noise'][n]['improvement'] > 0]
    hurting_noise = [n for n in noise_levels if results['noise'][n]['improvement'] < 0]
    
    if helping_noise:
        print(f"✅ Geometry learning HELPS with noise levels: {helping_noise}")
    if hurting_noise:
        print(f"⚠️  Geometry learning HURTS with noise levels: {hurting_noise}")
    
    helping_conn = [n for n in n_neighbors_list if results['connectivity'][n]['improvement'] > 0]
    hurting_conn = [n for n in n_neighbors_list if results['connectivity'][n]['improvement'] < 0]
    
    if helping_conn:
        print(f"✅ Geometry learning HELPS with connectivity: {helping_conn} neighbors")
    if hurting_conn:
        print(f"⚠️  Geometry learning HURTS with connectivity: {hurting_conn} neighbors")

if __name__ == "__main__":
    analyze_dataset_difficulty()