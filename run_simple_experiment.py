#!/usr/bin/env python3
"""
Simple experiment runner that avoids import issues.
"""
import os
import sys
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/src')

# Now import directly
from models.base_models import FixedGeometryGNN, LearnableMetricGNN, FrozenMetricGNN
from data.datasets import SyntheticManifoldDataset, create_data_splits


def run_simple_experiment():
    """Run a simple version of the experiment."""
    print("Starting Simple Geometry Learning Experiment")
    print("=" * 60)
    
    # Configuration
    config = {
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_runs': 2,
        'epochs': 50,
        'hidden_dim': 64,
        'num_layers': 3,
        'lr': 0.01,
        'geometry_lr': 0.05,
        'geometry_weight': 0.01,
        'reg_type': 'deviation'
    }
    
    # Set device
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Create dataset
    print("\n Creating Two Moons dataset...")
    data = SyntheticManifoldDataset.two_moons(
        n_samples=500,
        noise=0.15,
        n_neighbors=10
    )
    
    # Create splits
    data = create_data_splits(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    data = data.to(device)
    
    # Initialize models
    input_dim = data.x.shape[1]
    output_dim = len(data.y.unique())
    num_edges = data.edge_index.shape[1]
    
    models = {
        'fixed_geometry': FixedGeometryGNN(input_dim, config['hidden_dim'], output_dim, config['num_layers']),
        'learnable_metric': LearnableMetricGNN(input_dim, config['hidden_dim'], output_dim, num_edges, config['num_layers']),
        'frozen_metric': FrozenMetricGNN(input_dim, config['hidden_dim'], output_dim, num_edges, config['num_layers'])
    }
    
    results = {}
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\n Training {model_name}...")
        model = model.to(device)
        
        train_accs, val_accs = [], []
        
        # Setup optimizer
        if model_name == 'learnable_metric':
            optimizer = torch.optim.Adam([
                {'params': model.layers.parameters(), 'lr': config['lr']},
                {'params': [model.edge_weights], 'lr': config['geometry_lr']}
            ])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(config['epochs']):
            model.train()
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            
            # Add regularization for learnable metric
            if model_name == 'learnable_metric':
                reg_loss = model.geometric_regularization(config['reg_type'], data.edge_index)
                loss = loss + config['geometry_weight'] * reg_loss
            
            loss.backward()
            optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                train_pred = out[data.train_mask].argmax(dim=1)
                val_pred = out[data.val_mask].argmax(dim=1)
                
                train_acc = (train_pred == data.y[data.train_mask]).float().mean()
                val_acc = (val_pred == data.y[data.val_mask]).float().mean()
            
            train_accs.append(train_acc.item())
            val_accs.append(val_acc.item())
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")
        
        # Final test evaluation
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            test_pred = out[data.test_mask].argmax(dim=1)
            test_acc = (test_pred == data.y[data.test_mask]).float().mean()
        
        results[model_name] = {
            'train_accs': train_accs,
            'val_accs': val_accs,
            'test_acc': test_acc.item(),
            'final_train_acc': train_accs[-1],
            'final_val_acc': val_accs[-1]
        }
        
        print(f"  Final Test Accuracy: {test_acc:.3f}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/simple_experiment_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    print("\n Generating plots...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Training curves
    colors = {'fixed_geometry': 'blue', 'learnable_metric': 'green', 'frozen_metric': 'red'}
    for model_name in models.keys():
        axes[0].plot(results[model_name]['train_accs'], 
                    label=model_name, color=colors[model_name], alpha=0.8)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Training Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Validation curves
    for model_name in models.keys():
        axes[1].plot(results[model_name]['val_accs'], 
                    label=model_name, color=colors[model_name], alpha=0.8)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Test accuracy comparison
    model_names = list(models.keys())
    test_accs = [results[name]['test_acc'] for name in model_names]
    
    x = np.arange(len(model_names))
    axes[2].bar(x, test_accs, color=[colors[name] for name in model_names], alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([name.replace('_', '\n') for name in model_names])
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Test Accuracy Comparison')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "results_plot.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "=" * 60)
    print(" RESULTS SUMMARY")
    print("=" * 60)
    for model_name in models.keys():
        acc = results[model_name]['test_acc']
        print(f"{model_name:20s}: {acc:.3f}")
    
    print(f"\n Results saved to: {output_dir}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    run_simple_experiment()