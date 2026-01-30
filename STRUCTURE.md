# Project Structure Visualization

```
p_sparsity/
│
├── 📋 Configuration Files
│   ├── pyproject.toml           # UV package management, dependencies
│   ├── requirements.txt         # Backup for pip users
│   ├── .gitignore              # Git ignore patterns
│   └── configs/                # All hyperparameters in YAML
│       ├── model/
│       │   └── gat_default.yaml         # Model architecture config
│       ├── training/
│       │   └── reinforce_default.yaml   # Training hyperparameters
│       ├── data/
│       │   └── anisotropic_default.yaml # Data generation settings
│       └── evaluation/
│           └── default.yaml             # Evaluation configuration
│
├── 📚 Documentation
│   ├── README.md               # Main documentation
│   ├── QUICKSTART.md           # 5-minute setup guide  
│   ├── MIGRATION.md            # Detailed refactoring guide
│   └── SUMMARY.md              # Project status & roadmap
│
├── 🔧 Setup & Utilities
│   ├── setup.py                # Automated environment setup
│   └── examples/
│       └── demo_modules.py     # Working examples of all modules
│
├── 🎯 Entry Points
│   └── scripts/
│       ├── train.py            # Training entry point (skeleton)
│       ├── evaluate.py         # Evaluation runner (TODO)
│       └── visualize.py        # Visualization generator (TODO)
│
├── 📦 Main Package
│   └── src/p_sparsity/
│       │
│       ├── __init__.py         # Package initialization
│       ├── legacy.py           # Temporary: code to migrate
│       │
│       ├── 📊 data/            # Data Generation Module ✅ COMPLETE
│       │   ├── __init__.py
│       │   ├── base.py         # Abstract ProblemGenerator
│       │   ├── registry.py     # Registration system
│       │   ├── smooth_errors.py # Relaxation-based error generation
│       │   ├── dataset.py      # Dataset builder
│       │   └── generators/     # Problem generators
│       │       ├── __init__.py
│       │       ├── anisotropic.py  # Anisotropic diffusion ✅
│       │       ├── elasticity.py   # Linear elasticity 🚧
│       │       └── helmholtz.py    # Helmholtz problems 🚧
│       │
│       ├── 🧠 models/          # Model Architecture Module ✅ COMPLETE
│       │   ├── __init__.py
│       │   ├── amg_policy.py      # Main policy network
│       │   ├── gnn_backbones.py   # GAT, GCN, GraphSAGE
│       │   └── edge_features.py   # Edge feature engineering
│       │
│       ├── 🎯 rl/              # RL Training Module ❌ TODO
│       │   ├── __init__.py
│       │   ├── algorithms/
│       │   │   ├── __init__.py
│       │   │   ├── base.py         # Abstract trainer
│       │   │   └── reinforce.py    # REINFORCE algorithm
│       │   ├── rewards.py          # Reward functions
│       │   └── baselines.py        # Variance reduction
│       │
│       ├── 🔗 pyamg_interface/ # PyAMG Integration ❌ TODO
│       │   ├── __init__.py
│       │   ├── solver_builder.py   # C matrix, solver building
│       │   └── sampling.py         # Edge sampling strategies
│       │
│       ├── 📈 evaluation/      # Analysis Module ❌ TODO
│       │   ├── __init__.py
│       │   ├── pcg_analysis.py     # PCG convergence
│       │   ├── vcycle_analysis.py  # V-cycle metrics
│       │   └── eigenvalue_analysis.py # Spectral properties
│       │
│       ├── 🎨 visualization/   # Plotting Module ❌ TODO
│       │   ├── __init__.py
│       │   ├── sparsity.py         # Sparsity patterns
│       │   ├── convergence.py      # Convergence plots
│       │   ├── training_curves.py  # Training progress
│       │   └── comparison.py       # Learned vs standard
│       │
│       └── 🛠️ utils/           # Utilities Module ✅ COMPLETE
│           ├── __init__.py
│           ├── config.py           # YAML config loading
│           ├── tensorboard_logger.py # TensorBoard integration
│           └── experiment.py       # Experiment tracking
│
├── 🧪 Tests (Future)
│   └── tests/
│       ├── __init__.py
│       ├── test_data.py
│       ├── test_models.py
│       └── test_rl.py
│
└── 📁 Output Directory (Generated)
    └── outputs/
        └── {experiment_name}/
            ├── tensorboard/        # TensorBoard logs
            ├── checkpoints/        # Model checkpoints
            │   ├── best_model.pt
            │   └── checkpoint_epoch_*.pt
            ├── configs/            # Saved configurations
            │   ├── model_config.yaml
            │   ├── train_config.yaml
            │   └── data_config.yaml
            ├── plots/              # Generated visualizations
            └── logs/               # Metrics, statistics
                └── metadata.json

```

## Module Status Legend

- ✅ **COMPLETE**: Fully implemented and working
- 🚧 **PARTIAL**: Placeholder or basic implementation
- ❌ **TODO**: Not yet implemented

## Dependency Graph

```
┌─────────────┐
│   Configs   │
│   (YAML)    │
└──────┬──────┘
       │
       ├─────────────────┬────────────────┐
       ▼                 ▼                ▼
┌─────────────┐   ┌────────────┐   ┌───────────┐
│    Data     │   │   Models   │   │  Training │
│   Module    │   │   Module   │   │   Config  │
└──────┬──────┘   └─────┬──────┘   └─────┬─────┘
       │                 │                 │
       └────────┬────────┴────────┬────────┘
                ▼                 ▼
         ┌──────────────┐  ┌─────────────┐
         │  RL Module   │  │  PyAMG      │
         │   (TODO)     │  │  Interface  │
         └──────┬───────┘  └──────┬──────┘
                │                 │
                └────────┬────────┘
                         ▼
                  ┌─────────────┐
                  │ Evaluation  │
                  │   Module    │
                  └──────┬──────┘
                         │
                         ▼
                  ┌─────────────┐
                  │Visualization│
                  │   Module    │
                  └─────────────┘
```

## Data Flow

```
YAML Configs
    ↓
Config Loader (utils)
    ↓
Data Generator (data) → Problem A, coordinates, features
    ↓
Dataset Builder (data) → TrainSample objects
    ↓
GNN Policy (models) → Edge logits, B candidates
    ↓
RL Trainer (rl/TODO) → Sample edges, compute reward
    ↓
PyAMG Solver (pyamg_interface/TODO) → Build solver, apply V-cycle
    ↓
Reward → Backprop → Update policy
    ↓
Evaluation (evaluation/TODO) → PCG, V-cycle, eigenvalue analysis
    ↓
Visualization (visualization/TODO) → Plots, comparisons
    ↓
Experiment Tracker (utils) → Save checkpoints, logs, plots
```

## Key Design Patterns

### 1. Registry Pattern (Data)
```python
@register_generator("anisotropic")
class AnisotropicGenerator(ProblemGenerator):
    ...

generator = get_generator("anisotropic", config)
```

### 2. Config-Driven (Models)
```python
config = load_config("configs/model/gat_default.yaml")
model = build_policy_from_config(config)
# Change backbone: Just edit YAML!
```

### 3. Experiment Tracking (Utils)
```python
experiment = create_experiment("my_exp")
experiment.save_checkpoint(model, metrics={"reward": 0.8})
experiment.save_config(config)
experiment.save_plot(fig, "convergence.png")
```

### 4. Modular Features (Data)
```python
# Features configured in YAML
features:
  use_relaxed_vectors: true
  use_coordinates: true
  use_degree: false
```

## File Count

- **Configuration**: 4 YAML files
- **Documentation**: 4 markdown files
- **Python Modules**: 18 files (14 implemented, 4 TODO placeholders)
- **Scripts**: 2 files (1 skeleton, 1 complete example)
- **Supporting**: 3 files (pyproject.toml, requirements.txt, .gitignore)

**Total**: 31 files organized in professional package structure

## Next Implementation Priority

1. **pyamg_interface/** (2-3 hours) - Critical for training
2. **rl/algorithms/reinforce.py** (4-6 hours) - Core training loop
3. **evaluation/** (2-4 hours) - Metrics and analysis
4. **visualization/** (2-4 hours) - Plots and comparisons
5. **Complete scripts/** (1-2 hours) - Wire everything together

## Migration Path from main.py

```
main.py (1500 lines)
    │
    ├─ Lines 1-100   → Already split
    ├─ Lines 100-300 → src/p_sparsity/data/ ✅
    ├─ Lines 300-450 → src/p_sparsity/models/ ✅
    ├─ Lines 450-600 → src/p_sparsity/legacy.py (temp)
    ├─ Lines 600-900 → src/p_sparsity/rl/ ❌
    ├─ Lines 900-1100 → src/p_sparsity/evaluation/ ❌
    └─ Lines 1100-1500 → src/p_sparsity/visualization/ ❌
```
