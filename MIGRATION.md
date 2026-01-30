# Migration Guide: From main.py to Modular Structure

This guide helps you migrate from the monolithic `main.py` to the new modular structure.

## Overview

The original `main.py` (~1500 lines) has been split into a well-organized package structure:

```
src/p_sparsity/
├── data/           # Problem generators, datasets
├── models/         # GNN architectures, policy networks  
├── rl/             # Training algorithms (TODO)
├── pyamg_interface/  # Solver building (TODO)
├── evaluation/     # Analysis modules (TODO)
├── visualization/  # Plotting utilities (TODO)
└── utils/          # Config, logging, tracking
```

## Current Status

### ✅ Completed
- **Project Structure**: UV-based package with `pyproject.toml`
- **Configuration System**: YAML configs for model, training, data, evaluation
- **Data Module**: 
  - Registry pattern for problem generators
  - Anisotropic diffusion (complete)
  - Elasticity, Helmholtz (placeholders)
  - Configurable smooth error generation
  - Dataset builders
- **Model Module**:
  - Swappable GNN backbones (GAT, GCN, GraphSAGE)
  - Modular edge feature engineering
  - Separate B-candidate learning
- **Utils Module**:
  - YAML config loading
  - TensorBoard integration
  - Experiment tracking (checkpoints, artifacts)

### 🚧 In Progress
The following components from `main.py` are in `src/p_sparsity/legacy.py` and need modularization:

#### RL Module (`src/p_sparsity/rl/`)
- [ ] `algorithms/reinforce.py` - REINFORCE trainer
- [ ] `rewards.py` - Pluggable reward functions
- [ ] `baselines.py` - Baseline strategies

#### PyAMG Interface (`src/p_sparsity/pyamg_interface/`)
- [x] `sampling.py` - Edge sampling strategies (partial in legacy.py)
- [x] `solver_builder.py` - C matrix construction, solver building (partial in legacy.py)

#### Evaluation Module (`src/p_sparsity/evaluation/`)
- [ ] `pcg_analysis.py` - PCG convergence analysis
- [ ] `vcycle_analysis.py` - V-cycle metrics
- [ ] `eigenvalue_analysis.py` - Spectral analysis

#### Visualization Module (`src/p_sparsity/visualization/`)
- [ ] `sparsity.py` - Sparsity pattern plots
- [ ] `convergence.py` - Convergence plots
- [ ] `training_curves.py` - Training progress plots
- [ ] `comparison.py` - Learned vs standard AMG

#### Entry Scripts (`scripts/`)
- [ ] `train.py` - Training entry point
- [ ] `evaluate.py` - Evaluation runner
- [ ] `visualize.py` - Visualization generator

## How to Continue Development

### Phase 1: Complete Remaining Modules

1. **RL Module**
   ```python
   # src/p_sparsity/rl/algorithms/reinforce.py
   # Extract training loop from main.py lines 600-750
   ```

2. **PyAMG Interface**
   ```python
   # Move from legacy.py to proper modules
   # src/p_sparsity/pyamg_interface/solver_builder.py
   # src/p_sparsity/pyamg_interface/sampling.py
   ```

3. **Evaluation**
   ```python
   # src/p_sparsity/evaluation/pcg_analysis.py
   # Extract from main.py lines 900-1000
   ```

4. **Visualization**
   ```python
   # src/p_sparsity/visualization/sparsity.py
   # Extract from main.py lines 1100-1400
   ```

### Phase 2: Create Entry Scripts

Create `scripts/train.py` that:
1. Loads configs from YAML
2. Creates experiment tracker
3. Builds dataset using data module
4. Builds model from config
5. Runs RL training
6. Saves checkpoints and plots

Example skeleton:
```python
import argparse
from pathlib import Path
from p_sparsity.utils import load_config, create_experiment, setup_tensorboard
from p_sparsity.data import make_dataset
from p_sparsity.models import build_policy_from_config
# from p_sparsity.rl import ReinforceTrainer  # TODO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="configs/model/gat_default.yaml")
    parser.add_argument("--training", default="configs/training/reinforce_default.yaml")
    parser.add_argument("--data", default="configs/data/anisotropic_default.yaml")
    args = parser.parse_args()
    
    # Load configs
    model_cfg = load_config(args.model)
    train_cfg = load_config(args.training)
    data_cfg = load_config(args.data)
    
    # Setup experiment
    experiment = create_experiment(train_cfg.experiment.name)
    tb_logger, log_dir = setup_tensorboard(train_cfg.experiment.name)
    
    # Save configs
    experiment.save_config(model_cfg, "model_config.yaml")
    experiment.save_config(train_cfg, "train_config.yaml")
    experiment.save_config(data_cfg, "data_config.yaml")
    
    # Build dataset
    train_data = make_dataset(
        problem_type=data_cfg.problem_type,
        num_samples=data_cfg.train.num_samples,
        grid_size=data_cfg.train.grid_size,
        config=data_cfg,
        seed=data_cfg.train.seed,
    )
    
    # Build model
    model = build_policy_from_config(model_cfg)
    
    # TODO: Create trainer and run
    # trainer = ReinforceTrainer(model, train_data, train_cfg, experiment, tb_logger)
    # trainer.train()
    
    tb_logger.close()

if __name__ == "__main__":
    main()
```

### Phase 3: Testing

1. Create unit tests in `tests/`
2. Test each module independently
3. End-to-end integration test

## Quick Start with Current Code

While the modularization is in progress, you can:

### Option 1: Use Legacy main.py
```bash
# Backup approach - run original code
python main.py
```

### Option 2: Start Using New Modules
```python
from p_sparsity.data import make_dataset, get_generator
from p_sparsity.models import build_policy_from_config
from p_sparsity.utils import load_config

# Load config
cfg = load_config("configs/data/anisotropic_default.yaml")

# Create dataset
data = make_dataset("anisotropic", num_samples=10, grid_size=32, config=cfg)

# Build model
model_cfg = load_config("configs/model/gat_default.yaml")
model = build_policy_from_config(model_cfg)

# Model is ready for training!
```

## Key Improvements in New Structure

### 1. **Configurability**
- All hyperparameters in YAML
- Easy to run parameter sweeps
- Version control friendly

### 2. **Modularity**
- Swap GNN backbones with one config change
- Plugin new problem generators
- Add custom reward functions

### 3. **Experiment Tracking**
- Automatic checkpoint management
- Config snapshots for reproducibility
- Organized output directories

### 4. **Extensibility**
- Registry patterns for easy extension
- Clear separation of concerns
- Reusable components

## Next Steps for Development

1. **Copy remaining functions from main.py to appropriate modules**
   - Use the section comments in `legacy.py` as a guide
   - Preserve all functionality
   
2. **Create proper interfaces**
   - Define abstract base classes where appropriate
   - Use typing for better IDE support

3. **Write entry scripts**
   - `scripts/train.py` - Main training pipeline
   - `scripts/evaluate.py` - Run evaluation suite
   - `scripts/visualize.py` - Generate plots

4. **Add documentation**
   - Docstrings for all public functions
   - Usage examples in module docstrings

5. **Testing**
   - Unit tests for core functionality
   - Integration tests for full pipeline

## Questions?

- Check the module docstrings for usage examples
- Refer to the original `main.py` for algorithmic details
- The `legacy.py` file contains temporary implementations

## Contributing

When adding new features:
1. Follow the existing module structure
2. Add configuration options to YAML files
3. Use the registry pattern for extensible components
4. Update this migration guide
