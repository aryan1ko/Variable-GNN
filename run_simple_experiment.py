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
    