"""
Test model implementations.
"""
import torch
import pytest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.models.base_models import FixedGeometryGNN, LearnableMetricGNN, FrozenMetricGNN
except ImportError as e:
    print(f"Import error: {e}")
    print("Current Python path:", sys.path)
    # Try alternative import
    from models.base_models import FixedGeometryGNN, LearnableMetricGNN, FrozenMetricGNN


def test_fixed_geometry_gnn():
    """Test FixedGeometryGNN initialization and forward pass."""
    model = FixedGeometryGNN(
        input_dim=10,
        hidden_dim=16,
        output_dim=2,
        num_layers=3
    )
    
    # Test forward pass
    x = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    
    output = model(x, edge_index)
    
    assert output.shape == (5, 2)
    assert not torch.isnan(output).any()
    
    # Test parameter count
    params = model.get_parameters_count()
    assert params["trainable"] > 0
    assert params["trainable"] == params["total"]


def test_learnable_metric_gnn():
    """Test LearnableMetricGNN initialization and forward pass."""
    model = LearnableMetricGNN(
        input_dim=10,
        hidden_dim=16,
        output_dim=2,
        num_edges=5,
        num_layers=3
    )
    
    x = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    
    output = model(x, edge_index)
    
    assert output.shape == (5, 2)
    assert not torch.isnan(output).any()
    
    # Test metric getter
    metric = model.get_metric()
    assert metric.shape == (5,)
    assert torch.all(metric > 0)  # Should be positive
    
    # Test regularization
    reg_loss = model.geometric_regularization("deviation", edge_index)
    assert reg_loss.item() >= 0
    
    # Test parameter count includes geometry
    params = model.get_parameters_count()
    assert params["geometry"] == 5  # 5 edges


def test_frozen_metric_gnn():
    """Test FrozenMetricGNN (should have frozen geometry parameters)."""
    model = FrozenMetricGNN(
        input_dim=10,
        hidden_dim=16,
        output_dim=2,
        num_edges=5,
        num_layers=3
    )
    
    # Geometry parameters should not require grad
    assert not model.edge_weights.requires_grad
    
    # Network parameters should still require grad
    for name, param in model.named_parameters():
        if 'edge_weights' in name:
            assert not param.requires_grad
        else:
            assert param.requires_grad


def test_model_consistency():
    """Test that models have same network architecture."""
    fixed_model = FixedGeometryGNN(10, 16, 2, 3)
    learnable_model = LearnableMetricGNN(10, 16, 2, 5, 3)
    
    # Network layers should be same type
    assert len(fixed_model.layers) == len(learnable_model.layers)
    
    for fixed_layer, learnable_layer in zip(fixed_model.layers, learnable_model.layers):
        assert type(fixed_layer) == type(learnable_layer)
        assert fixed_layer.in_channels == learnable_layer.in_channels
        assert fixed_layer.out_channels == learnable_layer.out_channels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])