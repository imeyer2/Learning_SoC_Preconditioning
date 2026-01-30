# Quick Reference Card

## 🚀 Common Commands

### Training
```bash
# Basic training
python scripts/train.py

# Custom configs
python scripts/train.py \
  --model-config configs/model/gat_default.yaml \
  --train-config configs/training/reinforce_default.yaml \
  --data-config configs/data/anisotropic_default.yaml

# Resume from checkpoint
python scripts/train.py --resume results/experiments/exp_001/checkpoint_latest.pt
```

### Evaluation
```bash
# Evaluate trained model
python scripts/evaluate.py \
  --checkpoint results/experiments/exp_001/checkpoint_final.pt \
  --config configs/evaluation/default.yaml \
  --visualize

# Save results to custom directory
python scripts/evaluate.py \
  --checkpoint path/to/model.pt \
  --output my_results/
```

### Development
```bash
# Install/update dependencies
uv sync

# Run demo (test all modules)
python examples/demo_modules.py

# Launch TensorBoard
tensorboard --logdir results/tensorboard

# Run tests (if you add them later)
pytest tests/
```

---

## 📦 Module Quick Reference

### Import Patterns

```python
# Data generation
from p_sparsity.data import get_problem_generator
generator = get_problem_generator('anisotropic')
problem = generator.generate(grid_size=50, anisotropy_ratio=100)

# Model
from p_sparsity.models import AMGEdgePolicy
model = AMGEdgePolicy(
    input_dim=10,
    hidden_dim=64,
    backbone='gat',
    learn_B=True,
    B_extra=2
)

# PyAMG interface
from p_sparsity.pyamg_interface import (
    sample_topk_without_replacement,
    build_C_from_model,
    build_pyamg_solver
)

# Training
from p_sparsity.rl import ReinforceTrainer
trainer = ReinforceTrainer(model, train_data, config)
trainer.train()

# Evaluation
from p_sparsity.evaluation import (
    run_pcg_analysis,
    run_vcycle_analysis,
    run_eigenvalue_analysis
)

# Visualization
from p_sparsity.visualization import (
    plot_sparsity_pattern,
    plot_convergence_curves,
    plot_training_progress
)
```

---

## ⚙️ Configuration Quick Guide

### Model Config (`configs/model/gat_default.yaml`)
```yaml
backbone: gat          # Options: 'gat', 'gcn', 'graphsage'
hidden_dim: 64         # GNN hidden dimension
num_layers: 3          # Number of GNN layers
learn_B: true          # Enable B-candidate learning
B_extra: 2             # Number of learned B vectors
```

### Training Config (`configs/training/reinforce_default.yaml`)
```yaml
epochs: 100
batch_size: 1          # Currently only supports 1
learning_rate: 0.001
optimizer: adam

# Sampling
edges_per_row: 3       # Top-k edges to select
temperature:
  initial: 1.0
  final: 0.1
  decay: 0.99

# Reward
reward: vcycle_energy_reduction
complexity_penalty: 0.0

# Baseline
baseline: moving_average
baseline_momentum: 0.9
```

### Data Config (`configs/data/anisotropic_default.yaml`)
```yaml
type: anisotropic      # Options: 'anisotropic', 'elasticity', 'helmholtz'

params:
  grid_size: 50
  anisotropy_ratio: 100.0
  theta: 30.0          # Anisotropy angle (degrees)

features:
  use_smooth_errors: true
  num_smoothing_steps: 1
  aggregation: 'mean'  # For graph construction
```

---

## 🔍 Understanding Key Concepts

### What is B?
- **B = near-nullspace candidate vectors**
- Standard: just `[1, 1, 1, ...]` (constant vector)
- Learned: `[ones, learned_vec1, learned_vec2]`
- Purpose: Help PyAMG build better interpolation (P)
- See [docs/UNDERSTANDING_B.md](UNDERSTANDING_B.md) for details

### What is C?
- **C = strength-of-connection matrix**
- Binary matrix indicating "strong" connections
- Determines aggregation (which nodes group together)
- Built from edge logits via sampling

### What is P?
- **P = prolongation operator** (coarse → fine)
- **Computed by PyAMG** (not learned directly!)
- Built from C (aggregation) and B (interpolation)
- The actual operator used in V-cycles

### Relationship
```
Model learns:     C (from edge logits) + B (optional vectors)
                  ↓
PyAMG computes:   Aggregates → P_tent → P (smoothed)
                  ↓
AMG uses:         V-cycle with P and R = P^T
```

---

## 📊 Output Locations

### During Training
```
results/experiments/exp_XXX/
├── checkpoint_latest.pt      # Most recent checkpoint
├── checkpoint_epoch_050.pt   # Periodic checkpoints
├── checkpoint_final.pt       # Final model
├── config.yaml              # Copy of all configs
└── metrics.json             # Training metrics

results/tensorboard/exp_XXX/
└── events.out.tfevents...   # TensorBoard logs
```

### After Evaluation
```
results/evaluation/
├── evaluation_results.yaml   # All metrics
├── comparison_table.md       # Markdown table
└── visualizations/
    ├── problem1_sparsity.png
    ├── problem1_convergence.png
    └── ...
```

---

## 🐛 Troubleshooting

### "Module not found"
```bash
# Make sure you're in the right directory
cd /path/to/p_sparsity

# Install in editable mode
uv pip install -e .
```

### "CUDA out of memory"
```yaml
# In configs/training/reinforce_default.yaml
batch_size: 1  # Already minimum

# Or reduce model size in configs/model/gat_default.yaml
hidden_dim: 32  # Reduce from 64
num_layers: 2   # Reduce from 3
```

### "Training is slow"
```python
# Use CPU for small problems
python scripts/train.py --device cpu

# Or reduce problem size in configs/data/
grid_size: 30  # Reduce from 50
```

### "Reward is not improving"
```yaml
# Try adjusting learning rate
learning_rate: 0.0001  # Reduce if unstable

# Or increase temperature decay
temperature:
  decay: 0.95  # Slower annealing

# Or change reward function
reward: vcycle_energy_reduction  # Default
# See src/p_sparsity/rl/rewards.py for options
```

---

## 🎯 Extension Points

### Add New Problem Type
1. Create `src/p_sparsity/data/generators/my_problem.py`
2. Implement `ProblemGenerator` interface
3. Register with `@register_problem_generator('my_problem')`
4. Create config `configs/data/my_problem.yaml`
5. Use: `python scripts/train.py --data-config configs/data/my_problem.yaml`

### Add New GNN Backbone
1. Add class to `src/p_sparsity/models/gnn_backbones.py`
2. Update `BACKBONE_REGISTRY`
3. Use: Set `backbone: my_gnn` in config

### Add New Reward Function
1. Add function to `src/p_sparsity/rl/rewards.py`
2. Register with `@register_reward('my_reward')`
3. Use: Set `reward: my_reward` in training config

### Add New RL Algorithm
1. Create `src/p_sparsity/rl/algorithms/my_algorithm.py`
2. Implement trainer interface (similar to REINFORCE)
3. Update `scripts/train.py` to support new algorithm
4. Create config `configs/training/my_algorithm.yaml`

---

## 📚 Documentation Index

- [README.md](../README.md) - Project overview and setup
- [QUICKSTART.md](QUICKSTART.md) - Get started in 5 minutes
- [STRUCTURE.md](STRUCTURE.md) - Directory layout
- [UNDERSTANDING_B.md](UNDERSTANDING_B.md) - Deep dive into B vs C vs P
- [ARCHITECTURE.md](ARCHITECTURE.md) - Complete system architecture
- [COMPLETION_SUMMARY.md](../COMPLETION_SUMMARY.md) - What was built
- [MIGRATION.md](MIGRATION.md) - Migrating from old code

---

## 💡 Tips

- Start with small problems (grid_size=20) for fast iteration
- Use TensorBoard to monitor training: `tensorboard --logdir results/tensorboard`
- Save checkpoints frequently (already configured)
- Test on CPU first, then move to GPU if available
- Use `--visualize` flag in evaluation to see plots
- Check `examples/demo_modules.py` for usage examples

---

## 🔗 Key Files to Know

| File | Purpose | When to Edit |
|------|---------|--------------|
| `configs/model/*.yaml` | Model architecture | Change GNN type/size |
| `configs/training/*.yaml` | Training hyperparams | Tune learning |
| `configs/data/*.yaml` | Problem generation | New PDE types |
| `scripts/train.py` | Training entry point | Rarely |
| `scripts/evaluate.py` | Evaluation entry point | Rarely |
| `src/p_sparsity/data/generators/` | Problem types | Add new PDEs |
| `src/p_sparsity/models/gnn_backbones.py` | GNN architectures | Add new GNNs |
| `src/p_sparsity/rl/rewards.py` | Reward functions | New objectives |

---

**Everything is incredibly simple and interchangeable!** 🎉

For questions, see the docs/ folder or examine the code - it's all well-commented!
