# Learning Geometry on Fixed Topologies for Graph Neural Networks

This repository implements a novel paradigm for machine learning that optimizes the Riemannian geometry of model space while preserving topological structure. Rather than searching for optimal parameters within a fixed Euclidean space, we treat the metric tensor field itself as a learnable quantity, enabling the parameter space to adapt its geometric properties during training.

<img width="1492" height="489" alt="Screenshot 2026-01-09 at 10 56 41 PM" src="https://github.com/user-attachments/assets/0dc000d8-c692-4255-b486-fefe8b17b6f0" />
<img width="1499" height="901" alt="Screenshot 2026-01-07 at 1 57 39 PM" src="https://github.com/user-attachments/assets/52534ad7-df61-432e-803e-e3922c79f6cc" />


## Research Question

**Given a fixed topological structure, what performance and generalization gains are achievable through metric optimization?**

We isolate the effect of geometric learning by holding topology constant and comparing against traditional fixed-geometry baselines under controlled experimental conditions.

---

## Theoretical Foundation
Traditional machine learning optimizes parameters θ within a fixed geometric space, typically Euclidean R^n. The optimization trajectory follows standard gradient descent:

```
θ_{t+1} = θ_t - α ∇L(θ_t)
```

Our approach parameterizes the metric tensor G(θ) and performs optimization on a Riemannian manifold:

```
θ_{t+1} = θ_t - α G^{-1}(θ_t) ∇L(θ_t)
```

This framework exhibits a formal analogy to the Einstein-Hilbert action in general relativity, where the loss function plays the role of the action and data induces curvature in parameter space.

### Loss Formulation

```
L_total = L_data(θ) + λ_vol R_vol(G) + λ_smooth R_smooth(G)
```

where:
- **L_data**: Standard task loss (cross-entropy, MSE, etc.)
- **R_vol**: Volume-based regularization preventing metric degeneracy
- **R_smooth**: Smoothness penalty enforcing local geometric consistency

The regularization terms constrain the space of admissible geometries, preventing pathological solutions (collapsed metrics, unbounded curvature) while allowing meaningful geometric adaptation.

---

## Repository Structure

```
.
├── paper/                      # LaTeX source, figures, experimental results
├── src/
│   ├── models.py              # GNN architectures
│   ├── geometry.py            # Metric tensor parameterization
│   ├── regularization.py      # Geometric constraints
│   └── training.py            # Training procedures
├── tests/
│   ├── test_models.py
│   ├── test_geometry.py
│   └── run_tests.py
├── run_simple_experiment.py
├── run_improved_experiment.py
├── analyze_when_geometry_helps.py
├── debug_metric_learning.py
├── config.yaml
└── requirements.txt
```

---

## Installation

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

---

## Experiments

### Baseline Comparison

```bash
python run_simple_experiment.py
```

Compares fixed-geometry and learned-geometry variants under identical parameter budgets, optimization settings, and evaluation protocols.

### Convergence Analysis

```bash
python run_improved_experiment.py
```

Tracks generalization gap, convergence rate, and overfitting behavior across training.

### Geometric Analysis

```bash
python analyze_when_geometry_helps.py
```

Post-hoc analysis identifying regimes where metric learning improves generalization and characterizing failure modes (metric collapse, overfitting to geometric degrees of freedom).

### Diagnostics

```bash
python debug_metric_learning.py
```

Visualizes edge-length evolution, metric distortion patterns, and regularization behavior.

---

## Configuration

Experiments are specified via `config.yaml`:

```yaml
learning_rate: 0.001
geometry_learning_rate: 0.0001
volume_regularization: 0.1
smoothness_penalty: 0.05
epochs: 100
random_seed: 42
```

All hyperparameters are externalized to facilitate reproducibility and systematic ablation studies.

---

## Key Concepts

### Topology vs. Geometry

| Property | Topology | Geometry |
|----------|----------|----------|
| Definition | Connectivity structure | Distance measurement |
| Fixed in this work | Yes | No |
| Examples | Graph adjacency, layer connectivity | Edge lengths, metric tensor |
| Mathematical object | Combinatorial structure | Riemannian metric |

### Metric Parameterization

The metric tensor is parameterized via edge lengths in a discrete triangulation:

```
ds² = Σ_ij g_ij dθ^i dθ^j
```

Edge-based parameterization ensures:
- Positive definiteness
- Computational tractability
- Compatibility with automatic differentiation

### Regularization Necessity

Without geometric constraints, the learned metric can exhibit:
- **Degeneracy**: Metric determinant → 0
- **Unbounded distortion**: Extreme local stretching
- **Overfitting**: Memorizing training data through geometric degrees of freedom

Regularization terms impose soft constraints that balance data fidelity against geometric complexity, analogous to regularization in traditional ML but operating on the metric structure itself.

---

## Relationship to Existing Work

This approach connects to several research areas:

**Natural Gradient Descent**: Optimization in the space defined by the Fisher information metric. Our work generalizes this by learning arbitrary metrics rather than using a fixed information-geometric structure.

**Metric Learning**: Prior work learns task-specific distance functions. We learn the geometry of parameter space itself.

**Geometric Deep Learning**: Studies neural networks on non-Euclidean domains. We study non-Euclidean structure in parameter space.

**Information Geometry**: Studies manifold structure of probability distributions. We extend these ideas to deterministic parameter optimization.

---

## Experimental Results Summary

Preliminary findings indicate:

1. Metric learning provides consistent improvements in generalization when:
   - Parameter space dimensionality is high
   - Data manifolds exhibit intrinsic non-Euclidean structure
   - Training data is limited relative to model capacity

2. Gains diminish or reverse when:
   - Problems are inherently low-dimensional
   - Training data is abundant
   - Topology severely constrains geometric adaptation

3. Proper regularization is critical:
   - Volume regularization prevents metric collapse
   - Smoothness penalties prevent pathological local distortions
   - Hyperparameter selection significantly affects performance


---

## Reproducibility

All experiments are:
- Configuration-driven (no hardcoded hyperparameters)
- Seed-controlled (deterministic results)
- Version-tracked (pinned dependencies)
- Modular (clean separation of concerns)

To reproduce:
1. Clone repository
2. Install exact dependency versions
3. Execute experiment scripts
4. Compare against reported results in paper

## Future Work

- Extension to dynamic topology (meta-learners)
- Theoretical analysis of generalization bounds
- Efficient implementations for large-scale problems
- Applications to scientific model discovery
- Connection to physics-inspired architectures

---

## Citation

```bibtex
@article{
  title={Learning Geometry on Fixed Topologies for Graph Neural Networks},
  author={[Aryan Kondapally]},
  year={2025}
}
```

---

## License

MIT License. See LICENSE file for details.

---

## Contact

For questions regarding implementation details, experimental setup, or theoretical foundations, please email avk639@utexas.edu
