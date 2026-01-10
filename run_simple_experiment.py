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
   