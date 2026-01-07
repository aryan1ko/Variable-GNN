"""
Test dataset creation and loading.
"""
import torch
import pytest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.data.datasets import SyntheticManifoldDataset, create_data_splits
except ImportError as e:
    print(f"Import error: {e}")
    print("Current Python path:", sys.path)
    # Try alternative import
    from data.datasets import SyntheticManifoldDataset, create_data_splits


def test_two_moons_dataset():
    """Test two moons dataset creation."""
    data = SyntheticManifoldDataset.two_moons(
        n_samples=100,
        noise=0.1,
        n_neighbors=5
    )
    
    assert hasattr(data, 'x')
    assert hasattr(data, 'y')
    assert hasattr(data, 'edge_index')
    
    assert data.x.shape == (100, 2)
    assert data.y.shape == (100,)
    
    # Check that graph is connected
    unique_nodes = torch.unique(data.edge_index)
    assert len(unique_nodes) == 100


def test_swiss_roll_dataset():
    """Test Swiss roll dataset creation."""
    data = SyntheticManifoldDataset.swiss_roll(
        n_samples=100,
        n_neighbors=5
    )
    
    assert data.x.shape == (100, 3)
    assert data.y.shape == (100,)
    
    # Check labels are in range [0, 3]
    assert torch.all(data.y >= 0)
    assert torch.all(data.y <= 3)


def test_create_data_splits():
    """Test data splitting function."""
    data = SyntheticManifoldDataset.two_moons(n_samples=100)
    data = create_data_splits(data)
    
    assert hasattr(data, 'train_mask')
    assert hasattr(data, 'val_mask')
    assert hasattr(data, 'test_mask')
    
    # Check mask sizes sum to total nodes
    assert data.train_mask.sum() + data.val_mask.sum() + data.test_mask.sum() == 100
    
    # Check no overlap
    assert not torch.any(data.train_mask & data.val_mask)
    assert not torch.any(data.train_mask & data.test_mask)
    assert not torch.any(data.val_mask & data.test_mask)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])