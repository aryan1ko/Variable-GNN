"""
Geometric evaluation metrics.
"""
import torch
import numpy as np
from scipy import stats
from typing import List, Dict, Any


class GeometricMetrics:
    """Compute geometric properties of learned metrics."""
    
    @staticmethod
    def effective_curvature(edge_weights: torch.Tensor,
                           edge_index: torch.Tensor,
                           method: str = "ollivier") -> float:
        w = edge_weights.cpu().numpy()
        src, dst = edge_index.cpu().numpy()
        
        if method == "ollivier":
            curvatures = []
            unique_nodes = np.unique(src)
            
            for node in unique_nodes:
                node_edges = w[src == node]
                if len(node_edges) > 1:
                    mean_w = np.mean(node_edges)
                    for w_ij in node_edges:
                        curv = 1 - (w_ij / mean_w)
                        curvatures.append(curv)
            
            return np.mean(curvatures) if curvatures else 0.0
        
        elif method == "forman":
            degree_src = np.bincount(src, minlength=len(w))
            degree_dst = np.bincount(dst, minlength=len(w))
            
            forman_curv = 2 - (degree_src[src] + degree_dst[dst]) / (w + 1e-8)
            return np.mean(forman_curv)
        
        else:
            raise ValueError(f"Unknown curvature method: {method}")
    
    @staticmethod
    def metric_complexity(edge_weights: torch.Tensor,
                         method: str = "entropy") -> float:
        w = edge_weights.cpu().numpy()
        
        if method == "entropy":
            w_norm = w / (w.sum() + 1e-8)
            entropy = -np.sum(w_norm * np.log(w_norm + 1e-8))
            return entropy
        
        elif method == "variance":
            return np.var(w)
        
        elif method == "l2":
            return np.mean(w ** 2)
        
        else:
            raise ValueError(f"Unknown complexity method: {method}")
        
    @staticmethod
    def statistical_tests(accuracies_list: List[List[float]],
                         model_names: List[str],
                         alpha: float = 0.05) -> Dict[str, Any]:
        results = {}
        
        # Pairwise t-tests
        for i in range(len(accuracies_list)):
            for j in range(i + 1, len(accuracies_list)):
                t_stat, p_val = stats.ttest_ind(accuracies_list[i], accuracies_list[j])
                
                # Effect size (Cohen's d)
                mean_diff = np.mean(accuracies_list[i]) - np.mean(accuracies_list[j])
                pooled_std = np.sqrt((np.std(accuracies_list[i]) ** 2 + 
                                    np.std(accuracies_list[j]) ** 2) / 2)
                cohens_d = mean_diff / (pooled_std + 1e-8)
                
                key = f"{model_names[i]}_vs_{model_names[j]}"
                results[key] = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_val),
                    "significant": p_val < alpha,
                    "cohens_d": float(cohens_d),
                    "mean_diff": float(mean_diff)
                }
        
        return results
    
    @staticmethod
    def generalization_gap(train_accs: List[float],
                          test_accs: List[float]) -> Dict[str, float]:
        gaps = [train - test for train, test in zip(train_accs, test_accs)]
        return {
            "mean": float(np.mean(gaps)),
            "std": float(np.std(gaps)),
            "min": float(np.min(gaps)),
            "max": float(np.max(gaps))
        }
    
    @staticmethod
    def robustness_metrics(original_acc: float,
                          perturbed_accs: List[float]) -> Dict[str, float]:
        drops = [(original_acc - acc) / original_acc for acc in perturbed_accs]
        return {
            "mean_drop": float(np.mean(drops)),
            "max_drop": float(np.max(drops)),
            "std_drop": float(np.std(drops)),
            "robustness_score": 1.0 - float(np.mean(drops))  # Higher is better
        }
