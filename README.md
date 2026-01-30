# P-Sparsity: Learning AMG Preconditioner Patterns with RL

Reinforcement Learning for learning optimal sparsity patterns in Algebraic Multigrid (AMG) preconditioners for various PDE types.

## Overview

This project uses Graph Neural Networks (GNNs) trained with Reinforcement Learning to learn strength-of-connection patterns (C matrices) and near-nullspace candidates (B vectors) for PyAMG's Smoothed Aggregation solver. The goal is to improve PCG convergence across different PDE types:

- **Elliptic PDEs**: Anisotropic diffusion, elasticity
- **Hyperbolic PDEs**: Wave equations (planned)
- **Parabolic PDEs**: Heat equations (planned)

> **🔍 Confused about B vs C vs P?** See [docs/UNDERSTANDING_B.md](docs/UNDERSTANDING_B.md) for a detailed explanation of what the model learns and how PyAMG uses it.

## Project Structure

```
p_sparsity/
├── configs/                # YAML configuration files
│   ├── model/             # Model architectures (GAT, GCN, GraphSAGE)
│   ├── training/          # Training hyperparameters
│   ├── data/              # Data generation settings
│   └── evaluation/        # Evaluation configurations
├── src/p_sparsity/        # Main package
│   ├── data/              # Problem generators with registry pattern
│   ├── models/            # GNN models with swappable backbones
│   ├── rl/                # RL algorithms (REINFORCE, etc.)
│   ├── pyamg_interface/   # PyAMG solver building
│   ├── evaluation/        # PCG, V-cycle, eigenvalue analysis
│   ├── visualization/     # Plotting utilities
│   └── utils/             # Config, logging, tracking
├── scripts/               # Entry points
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
│   └── visualize.py      # Visualization script
└── outputs/               # Results (gitignored)
```

## Installation

This project uses [UV](https://github.com/astral-sh/uv) for package management:

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# For development dependencies
uv pip install -e ".[dev]"
```

## Quick Start

### Training

```bash
# Train with default configuration
python scripts/train.py

# Train with custom configs
python scripts/train.py \
    --model configs/model/gat_default.yaml \
    --training configs/training/reinforce_default.yaml \
    --data configs/data/anisotropic_default.yaml
```

### Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py \
    --checkpoint outputs/experiment_name/checkpoints/best_model.pt \
    --config configs/evaluation/default.yaml
```

### Visualization

```bash
# Generate comparison plots
python scripts/visualize.py \
    --checkpoint outputs/experiment_name/checkpoints/best_model.pt \
    --output-dir visualizations/
```

## Key Features

### 🔧 Modular Data Generation
- **Registry Pattern**: Easily add new problem types
- **Supported Problems**: Anisotropic diffusion, elasticity (more coming)
- **Configurable Smooth Errors**: Multiple relaxation schemes

### 🧠 Flexible Model Architecture
- **Swappable GNN Backbones**: GAT, GCN, GraphSAGE
- **Modular Edge Features**: Direction-aware, physical similarity
- **Separate B-Candidate Learning**: Optional near-nullspace candidates

### 🎯 Configurable RL Training
- **RL Algorithms**: REINFORCE (default), extensible to PPO/A2C
- **Pluggable Rewards**: V-cycle efficiency, complexity penalties
- **Baseline Strategies**: Moving average, value networks

### 📊 Comprehensive Evaluation
- **PCG Convergence**: Iteration counts, residual histories
- **V-Cycle Analysis**: Energy reduction ratios
- **Eigenvalue Analysis**: Spectral properties of M^-1 A
- **Sparsity Comparison**: Model vs. standard AMG patterns

### 🎨 Rich Visualizations
- **Training Curves**: Reward evolution, probe metrics
- **Sparsity Patterns**: Heatmaps, spy plots, overlays
- **Convergence Plots**: PCG residuals, iteration bars
- **Eigenvalue Spectra**: Condition numbers, clustering

## Configuration

All aspects of the pipeline are configurable via YAML files:

### Model Configuration (`configs/model/gat_default.yaml`)
- GNN backbone type (GAT/GCN/GraphSAGE)
- Hidden dimensions, attention heads
- Edge feature engineering
- B-candidate learning options

### Training Configuration (`configs/training/reinforce_default.yaml`)
- RL algorithm selection
- Learning rate, epochs, batch size
- Reward function configuration
- Baseline strategy

### Data Configuration (`configs/data/anisotropic_default.yaml`)
- Problem type (anisotropic, elasticity, etc.)
- Grid sizes, parameter ranges
- Smooth error generation settings
- Training/validation splits

### Evaluation Configuration (`configs/evaluation/default.yaml`)
- Test cases to run
- PCG tolerance, maximum iterations
- Visualization options
- Eigenvalue analysis settings

## Experiment Tracking

All experiments are automatically tracked with:
- **TensorBoard**: Real-time training metrics
- **Checkpoints**: Model weights saved at best performance
- **Config Snapshots**: Exact configuration used for reproducibility
- **Artifacts**: Generated plots, statistics, comparisons

Access TensorBoard:
```bash
tensorboard --logdir outputs/
```

## Adding New Components

### New Problem Generator

```python
# src/p_sparsity/data/generators/my_problem.py
from ..base import ProblemGenerator
from ..registry import register_generator

@register_generator("my_problem")
class MyProblemGenerator(ProblemGenerator):
    def generate(self, **kwargs):
        # Your implementation
        return A, metadata
```

### New GNN Backbone

```python
# src/p_sparsity/models/gnn_backbones.py
@register_backbone("my_gnn")
class MyGNNBackbone(nn.Module):
    # Your implementation
```

### New Reward Function

```python
# src/p_sparsity/rl/rewards.py
@register_reward("my_reward")
def my_reward_function(A, ml, **kwargs):
    # Your implementation
    return reward_value
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{p-sparsity-2026,
  title={Learning AMG Preconditioner Sparsity Patterns with Reinforcement Learning},
  author={Your Name},
  year={2026}
}
```

## License

MIT License - see LICENSE file for details
# Learning_SoC_Preconditioning
