"""
Data loading and processing utilities.
"""
from .datasets import (
    SyntheticManifoldDataset,
    RealDatasetLoader,
    create_data_splits
)

__all__ = [
    "SyntheticManifoldDataset",
    "RealDatasetLoader",
    "create_data_splits"
]