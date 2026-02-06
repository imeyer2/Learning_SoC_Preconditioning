# Case Studies Guide

This guide explains how to run case studies, analyze results with EDA, and where to find outputs.

## Directory Structure

```
case_studies/
├── configs/                    # Case study configuration files
│   ├── study_1_anisotropic.yaml
│   ├── study_2_elasticity.yaml
│   └── ...
├── results/                    # Output directory for results
│   └── study_1_anisotropic/
│       └── case_study_1_anisotropic/
│           ├── variation_A/
│           │   ├── results.json        # Raw results
│           │   ├── param_registry.json # Parameter tracking
│           │   ├── plots/              # Auto-generated plots
│           │   └── eda/                # Full EDA analysis
│           └── variation_B/
│               └── ...
└── README.md                   # This file
```

---

## Quick Start

### 1. Run a Case Study

```bash
# Run all variations in a case study
python scripts/run_case_study.py --config case_studies/configs/study_1_anisotropic.yaml

# Run a specific variation
python scripts/run_case_study.py --config case_studies/configs/study_1_anisotropic.yaml --variation B

# Skip training and use existing checkpoint (evaluation only)
python scripts/run_case_study.py \
    --config case_studies/configs/study_1_anisotropic.yaml \
    --variation B \
    --skip-training \
    --init-from outputs/exp_20260124_003342/checkpoints/best_model.pt
```

### 2. Generate Plots from Results

```bash
# Quick aggregate plots
python scripts/plot_results.py path/to/results.json

# Full EDA with individual plots for each test case
python scripts/plot_results.py path/to/results.json --eda

# Specify custom output directory
python scripts/plot_results.py path/to/results.json --eda --output ./my_analysis
```

---

## Configuration Files

Case study configs are YAML files in `case_studies/configs/`. Here's the structure:

```yaml
name: "case_study_1_anisotropic"
description: "Anisotropic diffusion case study"

# Problem definition
problem:
  problem_type: anisotropic
  param_ranges:
    epsilon: [0.001, 0.1]    # Anisotropy strength
    theta: [0.0, 3.14159]    # Anisotropy direction

# Variations to run (each is a separate experiment)
variations:
  A:
    description: "Small training set, single grid size"
    train:
      num_samples: 50
      grid_size: 32
      model_config: configs/model/gat_default.yaml
      train_config: configs/training/reinforce_default.yaml
    test:
      num_samples_per_size: 10
      grid_sizes: [32, 48, 64]
      
  B:
    description: "Larger training set, multi-scale"
    train:
      num_samples: 200
      grid_size: 32
    test:
      num_samples_per_size: 10
      grid_sizes: [32, 48, 64, 96]
```

---

## Output Files

After running a case study, you'll find:

### `results.json`
Raw metrics for all test problems:
```json
{
  "variation_name": "B",
  "train_time": 245.3,
  "eval_time": 89.2,
  "test_problems": [
    {
      "problem_id": "test_0000",
      "grid_size": 32,
      "params": {"epsilon": 0.05, "theta": 1.2},
      "learned": {"pcg": {"iterations": 15, "wall_time": 0.023}},
      "baseline": {"pcg": {"iterations": 28, "wall_time": 0.041}}
    },
    ...
  ]
}
```

### `plots/` (auto-generated)
- `iteration_histogram.png` - Distribution of PCG iterations
- `residual_curves.png` - Convergence curves (sample)
- `energy_decay.png` - V-cycle energy decay
- `scaling_curves.png` - Iterations vs grid size (if scaling study)

### `eda/` (generated with `--eda` flag)
Full exploratory data analysis:
```
eda/
├── eda_statistics.json         # All computed statistics
├── iteration_analysis.png      # 6-panel iteration analysis
├── speedup_analysis.png        # 6-panel speedup analysis
├── parameter_correlations.png  # Parameter vs performance
├── statistical_summary.png     # Summary tables, win/loss chart
└── individual_problems/        # Per-problem plots
    ├── test_0000/
    │   ├── residual_convergence.png
    │   ├── energy_decay.png
    │   └── metadata.json
    └── ...
```

---

## EDA Statistics

The `eda_statistics.json` contains:

```json
{
  "iterations": {
    "learned": {"mean": 24.4, "std": 12.1, "median": 21.5, "min": 8, "max": 60},
    "baseline": {"mean": 35.2, "std": 21.9, "median": 28.0, "min": 9, "max": 101},
    "improvement": {"mean_reduction": 10.8, "win_rate": 0.98}
  },
  "speedup": {
    "mean": 1.45,
    "pct_above_1": 0.82,
    "pct_above_1_5": 0.54
  },
  "correlations": {
    "correlations": {"theta": -0.04, "epsilon": -0.24}
  }
}
```

---

## Common Workflows

### Compare Multiple Variations

```bash
# Run both variations
python scripts/run_case_study.py --config case_studies/configs/study_1_anisotropic.yaml

# Results will be in:
# - results/study_1_anisotropic/.../variation_A/results.json
# - results/study_1_anisotropic/.../variation_B/results.json
# - results/study_1_anisotropic/.../combined_results.json
```

### Re-generate Plots from Existing Results

```bash
# If you already ran a case study and want new plots:
python scripts/plot_results.py \
    case_studies/results/study_1_anisotropic/.../variation_B/results.json \
    --eda
```

### Evaluate Pre-trained Model on New Problems

```bash
python scripts/run_case_study.py \
    --config case_studies/configs/study_1_anisotropic.yaml \
    --variation B \
    --skip-training \
    --init-from path/to/checkpoint.pt
```

---

## Creating a New Case Study

1. Copy an existing config:
   ```bash
   cp case_studies/configs/study_1_anisotropic.yaml case_studies/configs/my_study.yaml
   ```

2. Edit the config:
   - Change `name` and `description`
   - Set `problem.problem_type` (anisotropic, elasticity, helmholtz, poisson)
   - Define `param_ranges` for your problem
   - Configure training and test settings

3. Run:
   ```bash
   python scripts/run_case_study.py --config case_studies/configs/my_study.yaml
   ```

---

## Tips

- **Start small**: Use `--variation A` first to test your config quickly
- **Monitor training**: Check TensorBoard logs in `results/.../tensorboard/`
- **Limit EDA plots**: Use `--max-individual 20` if you have many test problems
- **Reproducibility**: Results include random seeds in `param_registry.json`
