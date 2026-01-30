# Understanding B in AMG: Near-Nullspace Candidates

## What is B?

In Algebraic Multigrid (AMG), **B** is a matrix of **near-nullspace candidate vectors** that represent the smooth modes the multigrid hierarchy should preserve.

## The AMG Hierarchy

### What the Model Learns

The GNN policy learns two things:

1. **Edge Logits → C Matrix** (Primary Output)
   - Logits for each edge indicating "strength of connection"
   - Sampled to create binary strength matrix **C**
   - **C determines which nodes aggregate together**

2. **B Candidates** (Optional, Secondary Output)
   - Additional smooth vectors beyond the constant vector
   - Help PyAMG build better interpolation operators

### How PyAMG Uses Them

```
Model Output → PyAMG Processing → AMG Hierarchy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Edge Logits        →  Sample top-k      →  C matrix (n×n, binary)
                         per row               "strength connections"
                         
2. B candidates       →  Normalize &       →  B matrix (n×k)
   (optional)            prepend ones          "nullspace candidates"

3. C + B + A          →  PyAMG builds      →  Prolongation P
                         aggregation            Restriction R
                                               Coarse operators A_c
```

## Why Learn B?

### Standard AMG (No Learned B)
```python
# PyAMG default: only constant vector
B = np.ones((n, 1))  # Just [1, 1, 1, ..., 1]^T
```

### With Learned B
```python
# Model learns additional vectors
B_extra = model.B_head(node_embeddings)  # Shape: (n, 2)

# Combined:
B = [ones_vector, learned_vector_1, learned_vector_2]
    # Shape: (n, 3)
```

**Benefits:**
- Better interpolation of smooth errors
- Improved convergence for anisotropic problems
- Captures problem-specific smooth modes

## Example: Anisotropic Diffusion

For a problem with strong coupling in X-direction:

```
Standard B:          Learned B might include:
┌   ┐               ┌     ┐   ┌           ┐   ┌           ┐
│ 1 │               │  1  │   │ x-coord   │   │  smooth   │
│ 1 │               │  1  │   │ x-coord   │   │  mode in  │
│ 1 │               │  1  │   │ x-coord   │   │  X-dir    │
│ 1 │               │  1  │   │ x-coord   │   │ ...       │
│...│               │ ... │   │  ...      │   │           │
└   ┘               └     ┘   └           ┘   └           ┘
 constant            const      geometric      learned
                                coords         physics
```

The learned vectors help PyAMG understand the anisotropy!

## In the Code

### Model Architecture
```python
class AMGEdgePolicy(nn.Module):
    def __init__(self, ...):
        # Main output: edge logits
        self.edge_mlp = nn.Sequential(...)  # → logits
        
        # Optional: B candidates
        if learn_B:
            self.B_head = nn.Sequential(...)  # → B_extra
    
    def forward(self, x, edge_index, edge_weight):
        h = self.backbone(x, edge_index)     # Node embeddings
        edge_logits = self.edge_mlp(...)     # Edge logits
        B_extra = self.B_head(h)             # B candidates (optional)
        return edge_logits, B_extra
```

### PyAMG Solver Building
```python
# Get model outputs
logits, B_extra = model(x, edge_index, edge_weight)

# 1. Build C from edge logits
selected_edges = sample_topk_per_row(logits, k=3)
C = C_from_selected_edges(A, edge_index, selected_edges)

# 2. Build B from learned vectors
B = build_B_for_pyamg(B_extra)
# Returns: [constant_vector, learned_vector_1, learned_vector_2]

# 3. PyAMG builds the hierarchy
ml = pyamg.smoothed_aggregation_solver(
    A,
    strength=('predefined', {'C': C}),  # Use learned C
    B=B,                                 # Use learned B
)

# 4. Get prolongation operator (computed by PyAMG)
P = ml.levels[0].P  # Built from C and B
```

## The Full Pipeline

```
Input: Matrix A
    ↓
GNN Policy
    ├→ Edge Logits  ──→  Sample  ──→  C matrix
    └→ B candidates ──→  Normalize ──→  B matrix
                ↓
        PyAMG Solver Builder
            ├─ Use C for aggregation
            └─ Use B for interpolation
                ↓
        Multigrid Hierarchy
            ├─ Prolongation P (computed)
            ├─ Restriction R (computed)
            └─ Coarse operators A_c (computed)
                ↓
        Apply V-cycle
                ↓
        Measure convergence → Reward
                ↓
        REINFORCE update → Improve policy
```

## Key Insight

**The model doesn't directly output P (the prolongation operator).**

Instead, it learns:
1. **Which connections are strong** (C matrix) - primary learning target
2. **What smooth modes matter** (B candidates) - helps PyAMG build better P

PyAMG then uses this information to construct P using its standard algorithms (aggregation, smoothing, etc.).

This is more general than directly learning P because:
- P's size depends on coarsening ratio (varies)
- C and B are more fundamental/interpretable
- PyAMG's construction ensures mathematical properties

## Configuration

In `configs/model/gat_default.yaml`:
```yaml
# B-candidate learning (near-nullspace for PyAMG)
learn_B: true      # Enable/disable B learning
B_extra: 2         # Number of additional candidates beyond constant vector

B_head:
  layers: [64]     # MLP architecture for B prediction
  activation: relu
```

Set `learn_B: false` to only learn C (simpler but potentially less effective).

## Summary

| Component | What It Is | Learned or Computed | Purpose |
|-----------|-----------|---------------------|---------|
| **A** | System matrix | Given (input) | The problem to solve |
| **C** | Strength matrix | **Learned** (from edge logits) | Defines aggregation |
| **B** | Nullspace candidates | **Learned** (optional) | Guides interpolation |
| **P** | Prolongation | Computed by PyAMG from C, B | Transfers between levels |
| **R** | Restriction | Computed by PyAMG | Transfers between levels |

The model learns C and B; PyAMG builds P from them!
