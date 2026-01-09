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

