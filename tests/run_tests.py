#!/usr/bin/env python3
"""
Minimal test runner to avoid import issues.
"""
import sys
import os
import unittest

# Add src to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

# Now import the modules
from models.base_models import FixedGeometryGNN, LearnableMetricGNN, FrozenMetricGNN
from data.datasets import SyntheticManifoldDataset, create_data_splits

import torch
import numpy as np

def test_fixed_geometry_gnn():
    """Test FixedGeometryGNN."""
    print("Testing FixedGeometryGNN...")
    model = FixedGeometryGNN(input_dim=10, hidden_dim=16, output_dim=2)
    x = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    output = model(x, edge_index)
    assert output.shape == (5, 2)
    print("✓ FixedGeometryGNN test passed")

def test_learnable_metric_gnn():
    """Test LearnableMetricGNN."""
    print("Testing LearnableMetricGNN...")
    model = LearnableMetricGNN(input_dim=10, hidden_dim=16, output_dim=2, num_edges=5)
    x = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    output = model(x, edge_index)
    assert output.shape == (5, 2)
    print("✓ LearnableMetricGNN test passed")

def test_two_moons_dataset():
    """Test dataset creation."""
    print("Testing Two Moons dataset...")
    data = SyntheticManifoldDataset.two_moons(n_samples=100)
    assert data.x.shape == (100, 2)
    assert data.y.shape == (100,)
    print("✓ Two Moons dataset test passed")

if __name__ == "__main__":
    print("Running minimal tests...")
    print(f"Python path: {sys.path}")
    
    try:
        test_fixed_geometry_gnn()
        test_learnable_metric_gnn()
        test_two_moons_dataset()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)