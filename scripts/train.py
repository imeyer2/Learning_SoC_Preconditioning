"""
Training script for P-Sparsity.

This script will be completed once the RL module is fully implemented.
For now, it demonstrates the setup and data loading pipeline.
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from p_sparsity.utils import load_config, merge_configs, ExperimentTracker, setup_tensorboard
from p_sparsity.data import make_dataset
from p_sparsity.models import build_policy_from_config
from p_sparsity.rl import ReinforceTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train AMG policy with RL")
    parser.add_argument(
        "--model",
        type=str,
        default="configs/model/gat_default.yaml",
        help="Path to model config"
    )
    parser.add_argument(
        "--training",
        type=str,
        default="configs/training/reinforce_default.yaml",
        help="Path to training config"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/data/anisotropic_default.yaml",
        help="Path to data config"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (auto-generated if not specified)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cpu/cuda, overrides config)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("P-Sparsity Training")
    print("=" * 80)
    
    # Load configurations
    print("\n[1/6] Loading configurations...")
    model_cfg = load_config(args.model)
    train_cfg = load_config(args.training)
    data_cfg = load_config(args.data)
    
    # Override device if specified
    if args.device:
        train_cfg.device = args.device
    
    print(f"  Model: {args.model}")
    print(f"  Training: {args.training}")
    print(f"  Data: {args.data}")
    print(f"  Device: {train_cfg.device}")
    
    # Setup experiment tracking
    print("\n[2/6] Setting up experiment tracking...")
    from datetime import datetime
    experiment_name = args.name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = Path(train_cfg.experiment.output_dir) / experiment_name
    experiment = ExperimentTracker(str(experiment_dir))
    
    # Save configurations
    experiment.save_config(model_cfg, "model_config.yaml")
    experiment.save_config(train_cfg, "train_config.yaml")
    experiment.save_config(data_cfg, "data_config.yaml")
    
    # Setup TensorBoard
    tb_logger, log_dir = setup_tensorboard(
        experiment_name=experiment_name,
        base_dir=train_cfg.experiment.output_dir
    )
    
    print(f"  Experiment dir: {experiment.experiment_dir}")
    print(f"  TensorBoard: {log_dir}")
    
    # Build training dataset
    print("\n[3/6] Building training dataset...")
    train_data = make_dataset(
        problem_type=data_cfg.problem_type,
        num_samples=data_cfg.train.num_samples,
        grid_size=data_cfg.train.grid_size,
        config=data_cfg,
        seed=data_cfg.train.seed,
    )
    print(f"  Created {len(train_data)} training samples")
    print(f"  Grid size: {data_cfg.train.grid_size}x{data_cfg.train.grid_size}")
    print(f"  Problem type: {data_cfg.problem_type}")
    
    # Build validation dataset (if enabled)
    val_data = None
    if data_cfg.validation.get("enabled", False):
        print("\n[4/6] Building validation dataset...")
        val_data = make_dataset(
            problem_type=data_cfg.problem_type,
            num_samples=data_cfg.validation.num_samples,
            grid_size=data_cfg.validation.grid_size,
            config=data_cfg,
            seed=data_cfg.validation.seed,
        )
        print(f"  Created {len(val_data)} validation samples")
    else:
        print("\n[4/6] Validation dataset disabled")
    
    # Build model
    print("\n[5/6] Building model...")
    model = build_policy_from_config(model_cfg)
    model.to(train_cfg.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {model_cfg.backbone.upper()} backbone")
    print(f"  Parameters: {num_params:,}")
    print(f"  Learn B: {model_cfg.learn_B}")
    
    # Training
    print("\n[6/6] Training...")
    trainer = ReinforceTrainer(
        model=model,
        train_data=train_data,
        config=train_cfg,
        experiment_tracker=experiment,
        tb_logger=tb_logger,
    )
    
    history = trainer.train()
    
    # Save final metrics
    experiment.save_metrics(history, "training_history.json")
    
    print("\n" + "=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"Best reward: {trainer.best_reward:.4f}")
    print(f"Experiment dir: {experiment.experiment_dir}")
    print(f"TensorBoard: tensorboard --logdir {tb_logger.log_dir}")
    print("=" * 80)
    
    # Cleanup
    tb_logger.close()
    
    # Save metadata
    experiment.save_metadata({
        "status": "incomplete",
        "message": "Training module not yet implemented",
        "model_params": num_params,
        "train_samples": len(train_data),
        "val_samples": len(val_data) if val_data else 0,
    })


if __name__ == "__main__":
    main()
