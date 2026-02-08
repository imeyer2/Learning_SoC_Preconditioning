# Random Baseline Analysis: Why "Random" GNN Models Show Speedups

## Key Finding

During ablation studies, we observed that a randomly initialized (untrained) GNN model still achieves significant speedups over default PyAMG baselines. This document explains why this occurs and what it means for the thesis.

## Experimental Results

### Comparison of C Matrix Construction Methods

| Method | Iterations (256×256 anisotropic) | Speedup vs Baseline |
|--------|----------------------------------|---------------------|
| **Baseline PyAMG (default)** | 78 | 1.00x |
| **Purely Random C** (random k edges per row, no model) | 120 | 0.65x (worse!) |
| **Random GNN** (untrained model) | 25 | 3.12x |
| **Trained GNN** | ~15-20 | ~4-5x |

### Key Observation

Purely random edge selection is **worse** than baseline, but a random GNN is **better**. Why?

## Root Cause Analysis

### The "Random" GNN Isn't Truly Random

The `build_C_from_model` function constructs node features using:

```python
feature_config = {
    "use_relaxed_vectors": True,
    "use_coordinates": True,
    "num_vecs": 4,
    "relax_iters": 5,
    "relaxation_scheme": "jacobi",
    "omega": 2.0 / 3.0,
}
x = node_features_for_policy(A, coords, feature_config)
```

These features contain **problem-specific information**:

1. **Relaxed error vectors**: Derived from Jacobi iterations on random initial vectors, these capture information about the near-nullspace of the matrix A
2. **Spatial coordinates**: Encode geometric structure of the problem
3. **Edge weights**: Normalized matrix entries passed to the GNN

### What the GNN Does (Even Untrained)

Even with random weights, the GNN architecture:

1. **Aggregates neighbor information** via attention mechanisms
2. **Combines problem-specific features** (relaxed vectors, coordinates, weights)
3. **Produces edge logits** that depend on the local structure

The attention mechanism naturally focuses on certain edges based on feature similarity, creating a structured (non-random) C matrix.

### Why This Matters

| Approach | What It Does | Result |
|----------|--------------|--------|
| **Purely random C** | Ignores all problem structure | Worse than baseline |
| **Random GNN** | Random transformation of meaningful features | Better than baseline |
| **Trained GNN** | Learned optimal transformation | Best performance |

## Implications for Thesis

### Positive Framing

This finding can be presented as **evidence that the architecture choice matters**:

> "The GNN architecture provides inherent value by aggregating problem-specific features into edge scores. Even without reinforcement learning, the attention-based message passing creates structured strength matrices that improve solver convergence. Training further optimizes this process by learning problem-aware scoring functions."

### Ablation Study Design

For fair ablation studies, compare:

1. **Baseline PyAMG** (default strength selection)
2. **Tuned PyAMG** (hand-tuned θ parameter)
3. **Random GNN** (untrained model with problem features)
4. **Trained GNN** (full approach)

The improvement from Random GNN → Trained GNN shows the **value of RL training**.
The improvement from Purely Random C → Random GNN shows the **value of the architecture**.

## Technical Details

### Feature Construction Pipeline

```
Matrix A → Relaxed Vectors → Node Features → GNN → Edge Logits → Top-k Selection → C Matrix
              ↑                    ↑
         (problem info)     (architecture provides
                            aggregation even with
                            random weights)
```

### Why Relaxed Vectors Matter

Relaxed vectors approximate the near-nullspace of A, which is exactly what AMG needs for good interpolation. By including these as features, even a random GNN has access to information about:

- Which directions are "smooth" (near-nullspace)
- Local anisotropy structure
- Coupling between DOFs

## Recommendations

1. **Always include purely random C baseline** to demonstrate architecture value
2. **Report both Random GNN and Trained GNN** to show training value
3. **Use same features for all comparisons** to isolate the learning contribution
4. **Consider this as a "feature engineering" contribution** in addition to the learning contribution

## Date

Analysis performed: February 5, 2026
