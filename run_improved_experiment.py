#!/usr/bin/env python3
"""
Improved experiment with better hyperparameters for learnable metric.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/src')

from models.base_models import FixedGeometryGNN, LearnableMetricGNN, FrozenMetricGNN
from data.datasets import SyntheticManifoldDataset, create_data_splits

def run_improved_experiment():
    """Run experiment with optimized hyperparameters."""
    print(" IMPROVED GEOMETRY LEARNING EXPERIMENT")
    print("=" * 60)
    
    # IMPROVED CONFIGURATION
    config = {
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_runs': 3,
        'epochs': 100,
        'hidden_dim': 64,
        'num_layers': 3,
        'lr': 0.01,
        'geometry_lr': 0.01,  # REDUCED from 0.05
        'geometry_weight': 0.1,  # INCREASED from 0.01
        'reg_type': 'smoothness',  # CHANGED from 'deviation'
        'weight_decay': 0.0005,
    }
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
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
        'learnable_metric': LearnableMetricGNN(input_dim, config['hidden_dim'], output_dim, num_edges, 
                                              config['num_layers'], init_method='small'),  # SMALL INIT
        'frozen_metric': FrozenMetricGNN(input_dim, config['hidden_dim'], output_dim, num_edges, 
                                        config['num_layers'], init_method='small')
    }
    
    results = {}
    all_histories = {}
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\n Training {model_name}...")
        
        # Multiple runs
        run_accuracies = []
        run_histories = []
        
        for run in range(config['n_runs']):
            print(f"  Run {run+1}/{config['n_runs']}")
            
            # Reinitialize model
            if model_name == 'fixed_geometry':
                model = FixedGeometryGNN(input_dim, config['hidden_dim'], output_dim, config['num_layers'])
            elif model_name == 'learnable_metric':
                model = LearnableMetricGNN(input_dim, config['hidden_dim'], output_dim, num_edges, 
                                          config['num_layers'], init_method='small')
            else:
                model = FrozenMetricGNN(input_dim, config['hidden_dim'], output_dim, num_edges, 
                                       config['num_layers'], init_method='small')
            
            model = model.to(device)
            
            # Setup optimizer
            if model_name == 'learnable_metric':
                optimizer = torch.optim.AdamW([  # CHANGED to AdamW
                    {'params': model.layers.parameters(), 'lr': config['lr']},
                    {'params': [model.edge_weights], 'lr': config['geometry_lr']}
                ], weight_decay=config['weight_decay'])
            else:
                optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], 
                                             weight_decay=config['weight_decay'])
            
            criterion = nn.CrossEntropyLoss()
            
            # Training history
            history = {'train': [], 'val': [], 'test': []}
            
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
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Evaluate every 10 epochs
                if epoch % 10 == 0 or epoch == config['epochs'] - 1:
                    model.eval()
                    with torch.no_grad():
                        out = model(data.x, data.edge_index)
                        
                        for mask_name, mask in [('train', data.train_mask), 
                                               ('val', data.val_mask), 
                                               ('test', data.test_mask)]:
                            pred = out[mask].argmax(dim=1)
                            acc = (pred == data.y[mask]).float().mean()
                            history[mask_name].append(acc.item())
            
            # Final test accuracy
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                test_pred = out[data.test_mask].argmax(dim=1)
                test_acc = (test_pred == data.y[data.test_mask]).float().mean()
            
            run_accuracies.append(test_acc.item())
            run_histories.append(history)
            
            # Print metric statistics for learnable model
            if model_name == 'learnable_metric' and hasattr(model, 'get_metric'):
                metric = model.get_metric()
                print(f"    Metric: mean={metric.mean():.3f}, std={metric.std():.3f}")
        
        # Store results
        results[model_name] = {
            'mean': np.mean(run_accuracies),
            'std': np.std(run_accuracies),
            'all': run_accuracies
        }
        all_histories[model_name] = run_histories
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/improved_experiment_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate improved plots
    print("\n Generating plots...")
    generate_improved_plots(results, all_histories, output_dir, config)
    
    # Print summary
    print("\n" + "=" * 60)
    print(" IMPROVED RESULTS SUMMARY")
    print("=" * 60)
    for model_name in models.keys():
        mean_acc = results[model_name]['mean']
        std_acc = results[model_name]['std']
        print(f"{model_name:20s}: {mean_acc:.3f} ± {std_acc:.3f}")
    
    # Compare with expected results
    print("\n PERFORMANCE COMPARISON")
    print("-" * 40)
    fixed_acc = results['fixed_geometry']['mean']
    learnable_acc = results['learnable_metric']['mean']
    improvement = learnable_acc - fixed_acc
    
    if improvement > 0:
        print(f" Learnable metric IMPROVES by {improvement:.3f}")
    else:
        print(f"⚠️  Learnable metric UNDERPERFORMS by {-improvement:.3f}")
    
    print(f"\n Results saved to: {output_dir}")
    print("=" * 60)
    
    return results

def generate_improved_plots(results, all_histories, output_dir, config):
    """Generate better plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    colors = {'fixed_geometry': 'blue', 'learnable_metric': 'green', 'frozen_metric': 'red'}
    
    # Plot 1: Average training curves
    for model_name, histories in all_histories.items():
        # Average across runs
        avg_train = np.mean([h['train'] for h in histories], axis=0)
        epochs = np.arange(0, config['epochs'] + 1, 10)
        
        axes[0, 0].plot(epochs[:len(avg_train)], avg_train, 
                       label=model_name, color=colors[model_name], alpha=0.8, linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Training Accuracy (Averaged)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0.5, 1.0])
    
    # Plot 2: Average test curves
    for model_name, histories in all_histories.items():
        avg_test = np.mean([h['test'] for h in histories], axis=0)
        epochs = np.arange(0, config['epochs'] + 1, 10)
        
        axes[0, 1].plot(epochs[:len(avg_test)], avg_test, 
                       label=model_name, color=colors[model_name], alpha=0.8, linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Test Accuracy (Averaged)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0.5, 1.0])
    
    # Plot 3: Final test accuracy comparison
    model_names = list(results.keys())
    means = [results[name]['mean'] for name in model_names]
    stds = [results[name]['std'] for name in model_names]
    
    x = np.arange(len(model_names))
    bars = axes[0, 2].bar(x, means, yerr=stds, capsize=5,
                         color=[colors[name] for name in model_names], alpha=0.8)
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels([name.replace('_', '\n') for name in model_names])
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].set_title('Final Test Accuracy Comparison')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    axes[0, 2].set_ylim([0.5, 1.0])
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{mean:.3f}', ha='center', va='bottom')
    
    # Plot 4: Overfitting gap (Train - Test)
    overfitting_gaps = {}
    for model_name, histories in all_histories.items():
        final_train = np.mean([h['train'][-1] for h in histories])
        final_test = np.mean([h['test'][-1] for h in histories])
        overfitting_gaps[model_name] = final_train - final_test
    
    x = np.arange(len(overfitting_gaps))
    axes[1, 0].bar(x, list(overfitting_gaps.values()),
                  color=[colors[name] for name in overfitting_gaps.keys()], alpha=0.8)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([name.replace('_', '\n') for name in overfitting_gaps.keys()])
    axes[1, 0].set_ylabel('Overfitting Gap')
    axes[1, 0].set_title('Overfitting (Train - Test)')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Convergence speed (epochs to reach 90% train accuracy)
    convergence_speeds = {}
    for model_name, histories in all_histories.items():
        speeds = []
        for history in histories:
            for i, acc in enumerate(history['train']):
                if acc >= 0.9:
                    speeds.append(i * 10)  # Convert to epoch number
                    break
            else:
                speeds.append(config['epochs'])  # Never reached
        
        convergence_speeds[model_name] = np.mean(speeds)
    
    x = np.arange(len(convergence_speeds))
    bars = axes[1, 1].bar(x, list(convergence_speeds.values()),
                         color=[colors[name] for name in convergence_speeds.keys()], alpha=0.8)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([name.replace('_', '\n') for name in convergence_speeds.keys()])
    axes[1, 1].set_ylabel('Epochs to 90% Accuracy')
    axes[1, 1].set_title('Convergence Speed')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, speed in zip(bars, convergence_speeds.values()):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{int(speed)}', ha='center', va='bottom')
    
    # Plot 6: Performance summary table
    axes[1, 2].axis('off')
    
    # Create summary text
    summary_text = "PERFORMANCE SUMMARY\n\n"
    for model_name in results.keys():
        mean = results[model_name]['mean']
        std = results[model_name]['std']
        summary_text += f"{model_name}:\n"
        summary_text += f"  Accuracy: {mean:.3f} ± {std:.3f}\n"
        
        if model_name in overfitting_gaps:
            summary_text += f"  Overfitting: {overfitting_gaps[model_name]:.3f}\n"
        
        if model_name in convergence_speeds:
            summary_text += f"  Convergence: {int(convergence_speeds[model_name])} epochs\n"
        
        summary_text += "\n"
    
    axes[1, 2].text(0.1, 0.95, summary_text, fontsize=10, 
                   verticalalignment='top', family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_dir / "improved_results.png", dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_improved_experiment()