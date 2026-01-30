# Quick Start Guide

## 5-Minute Setup

### 1. Install UV (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or: pip install uv
```

### 2. Run Setup Script
```bash
cd /Users/80095022/Desktop/Preconditioning_Thesis_Books/code/p_sparsity
python setup.py
```

### 3. Activate Environment
```bash
source .venv/bin/activate
```

### 4. Try It Out
```bash
# See what the new modules can do
python examples/demo_modules.py

# Try the training script skeleton
python scripts/train.py
```

## Understanding the New Structure

### Old Way (main.py)
```python
# Everything in one 1500-line file
# Hard to modify without breaking things
# No clear separation of concerns
```

### New Way (Modular)
```python
# Data generation
from p_sparsity.data import make_dataset
dataset = make_dataset("anisotropic", num_samples=10, grid_size=32, config=cfg)

# Model creation
from p_sparsity.models import build_policy_from_config  
model = build_policy_from_config(model_cfg)

# Training (coming soon)
# from p_sparsity.rl import ReinforceTrainer
# trainer = ReinforceTrainer(model, dataset, config)
# trainer.train()
```

## Key Files to Know

### For Users
- **README.md** - Full documentation
- **configs/** - All hyperparameters (edit these!)
- **examples/demo_modules.py** - Working examples
- **scripts/train.py** - Main entry point (incomplete)

### For Developers
- **MIGRATION.md** - Detailed refactoring guide
- **SUMMARY.md** - What's done, what's not
- **src/p_sparsity/** - Main package
- **src/p_sparsity/legacy.py** - Code to migrate

## What Works Right Now

✅ **Configuration system** - Edit YAMLs, not code
✅ **Data generation** - Multiple problem types
✅ **Model architecture** - Swappable GNN backbones
✅ **Experiment tracking** - Auto-save configs, checkpoints
✅ **Registry pattern** - Easy to add new generators

## What Needs Work

❌ **Training loop** - Needs RL module
❌ **Reward computation** - Needs PyAMG interface
❌ **Evaluation** - Needs analysis modules
❌ **Visualization** - Needs plotting modules

## Common Tasks

### Change Model Architecture
```bash
# Edit configs/model/gat_default.yaml
backbone: gcn  # Changed from gat to gcn
hidden_dim: 128  # Changed from 64
```

### Add New Problem Type
```python
# In src/p_sparsity/data/generators/my_problem.py
from ..base import ProblemGenerator
from ..registry import register_generator

@register_generator("my_problem")
class MyProblemGenerator(ProblemGenerator):
    def generate(self, grid_size, **kwargs):
        # Your implementation
        return A
```

### Run Experiments
```bash
# Once training is complete:
python scripts/train.py \
    --model configs/model/gat_default.yaml \
    --training configs/training/reinforce_default.yaml \
    --data configs/data/anisotropic_default.yaml \
    --name my_experiment
```

## Next Steps

1. **To Use Now**: Keep using `main.py` - it still works!

2. **To Help Develop**: Read `MIGRATION.md` and help complete modules

3. **To Customize**: Edit YAML configs and use existing modules

## Getting Help

- **Examples not working?** Check dependencies with `uv pip list`
- **Want to contribute?** Start with `MIGRATION.md`  
- **Confused?** Compare with original `main.py`

## TL;DR

```bash
# Setup (one-time)
python setup.py
source .venv/bin/activate

# See what works
python examples/demo_modules.py

# Current status: ~50% complete
# - Data ✓
# - Models ✓  
# - Config ✓
# - Training ✗ (needs work)
# - Evaluation ✗ (needs work)

# For now: use original main.py
# Future: use modular scripts/train.py
```
