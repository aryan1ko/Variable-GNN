"""
Dataset creation and loading utilities.
"""
import torch
from torch_geometric.data import Data
import numpy as np
from typing import Optional
from sklearn import datasets as skdatasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler


class SyntheticManifoldDataset:
    """Create synthetic manifold datasets."""
    
    @staticmethod
    def two_moons(n_samples: int = 1000,
                  noise: float = 0.1,
                  n_neighbors: int = 10) -> Data:
        X, y = skdatasets.make_moons(n_samples=n_samples, noise=noise)
        
        adj = kneighbors_graph(X, n_neighbors=n_neighbors, 
                              mode='connectivity', include_self=False)
        
        edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
        x = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, y=y)
    
    @staticmethod
    def swiss_roll(n_samples: int = 1000,
                   noise: float = 0.1,
                   n_neighbors: int = 15,
                   n_classes: int = 4) -> Data:
        X, t = skdatasets.make_swiss_roll(n_samples=n_samples, noise=noise)
        
        adj = kneighbors_graph(X, n_neighbors=n_neighbors,
                              mode='connectivity', include_self=False)
        
        edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
        x = torch.tensor(X, dtype=torch.float32)
        
        quantiles = np.quantile(t, np.linspace(0, 1, n_classes + 1)[1:-1])
        y = torch.tensor(np.digitize(t, quantiles), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, y=y)
    
    @staticmethod
    def sphere(n_samples: int = 1000,
               radius: float = 1.0,
               n_neighbors: int = 15) -> Data:
        phi = np.random.uniform(0, 2*np.pi, n_samples)
        costheta = np.random.uniform(-1, 1, n_samples)
        theta = np.arccos(costheta)
        
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        
        X = np.column_stack([x, y, z])
        
        labels = np.zeros(n_samples, dtype=int)
        labels[(x > 0) & (y > 0)] = 0
        labels[(x < 0) & (y > 0)] = 1
        labels[(x < 0) & (y < 0)] = 2
        labels[(x > 0) & (y < 0)] = 3
        
        adj = kneighbors_graph(X, n_neighbors=n_neighbors,
                              mode='connectivity', include_self=False)
        
        edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
        x = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, y=y)


class RealDatasetLoader:
    """Loader for real-world datasets."""
    
    @staticmethod
    def load_citation_network(name: str = "cora",
                              root: str = "data",
                              normalize_features: bool = True) -> Data:
        from torch_geometric.datasets import Planetoid
        
        dataset = Planetoid(root=root, name=name)
        data = dataset[0]
        
        if normalize_features:
            data.x = data.x / data.x.sum(1, keepdim=True).clamp(min=1)
            
        return data
    
    @staticmethod
    def load_tabular_dataset(name: str = "wine") -> Data:
        from sklearn.datasets import load_wine, load_breast_cancer
        
        if name == "wine":
            data = load_wine()
        elif name == "cancer":
            data = load_breast_cancer()
        else:
            raise ValueError(f"Unknown dataset: {name}")
            
        X, y = data.data, data.target
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        n_neighbors = min(10, len(X) // 10)
        adj = kneighbors_graph(X, n_neighbors=n_neighbors,
                              mode='connectivity', include_self=False)
        
        edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
        x = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, y=y)


def create_data_splits(data: Data,
                       train_ratio: float = 0.6,
                       val_ratio: float = 0.2,
                       test_ratio: float = 0.2,
                       seed: int = 42) -> Data:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    n_nodes = data.num_nodes
    indices = torch.randperm(n_nodes, generator=torch.Generator().manual_seed(seed))
    
    train_end = int(train_ratio * n_nodes)
    val_end = train_end + int(val_ratio * n_nodes)
    
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    return data