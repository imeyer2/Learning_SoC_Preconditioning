#!/usr/bin/env python
"""
Run a case study experiment.

Usage:
    python scripts/run_case_study.py --config case_studies/configs/study_1_anisotropic.yaml
    python scripts/run_case_study.py --config ... --variation A
    python scripts/run_case_study.py --config ... --checkpoint outputs/exp_xxx/best_model.pt
"""

import argparse
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_colored(msg, color=Colors.END, bold=False):
    """Print colored text to terminal."""
    prefix = Colors.BOLD if bold else ""
    print(f"{prefix}{color}{msg}{Colors.END}")


def print_reward_explanation(reward_type: str):
    """Print explanation of what the reward means."""
    print_colored("\n📖 REWARD INTERPRETATION GUIDE:", Colors.CYAN, bold=True)
    print_colored("─" * 60, Colors.CYAN)
    
    if reward_type == "vcycle_energy_reduction":
        print("""
   Reward = -mean(log(ρ)) - complexity_penalty
   
   where ρ = ||e_after||²_A / ||e_before||²_A  (V-cycle convergence factor)
   
   ┌────────────────────────────────────────────────────────────┐
   │  Reward Value  │           Interpretation                 │
   ├────────────────┼──────────────────────────────────────────┤
   │    > 2.0       │  Excellent: ρ < 0.14 per V-cycle         │
   │   1.0 - 2.0    │  Good: ρ ≈ 0.14 - 0.37                   │
   │   0.5 - 1.0    │  Moderate: ρ ≈ 0.37 - 0.61               │
   │   0.0 - 0.5    │  Poor: ρ ≈ 0.61 - 1.0 (barely converging)│
   │    < 0.0       │  Bad: complexity penalty or divergence    │
   └────────────────┴──────────────────────────────────────────┘
   
   • Higher reward = faster V-cycle convergence
   • ρ < 0.3 is typically considered "good" for multigrid
   • Complexity penalty kicks in if operator complexity > target (default 1.35)
""")
    elif reward_type == "pcg_residual_reduction":
        print("""
   Reward = (1/k) × log(||r₀|| / ||r_k||)  (avg log-reduction per PCG iter)
   
   This measures how fast PCG converges when using AMG as preconditioner.
   
   ┌────────────────────────────────────────────────────────────┐
   │  Reward Value  │           Interpretation                 │
   ├────────────────┼──────────────────────────────────────────┤
   │    > 2.5       │  Excellent: ~10x residual drop per iter  │
   │   1.5 - 2.5    │  Good: ~5-10x residual drop per iter     │
   │   0.7 - 1.5    │  Moderate: 2-5x residual drop per iter   │
   │   0.3 - 0.7    │  Poor: <2x residual drop per iter        │
   │    < 0.3       │  Bad: preconditioner barely helping      │
   └────────────────┴──────────────────────────────────────────┘
   
   • Higher reward = better preconditioner quality
   • Reward ≈ 1.0 means residual drops by factor of e ≈ 2.7 per iteration
   • Typical well-tuned AMG achieves reward 1.5-2.5
   • Compare against baseline (standard AMG) to judge improvement
""")
    elif reward_type == "pcg_iterations":
        print("""
   Reward = -log(iterations / max_iters)
   
   Directly penalizes number of PCG iterations needed.
   
   ┌────────────────────────────────────────────────────────────┐
   │  Reward Value  │           Interpretation                 │
   ├────────────────┼──────────────────────────────────────────┤
   │    > 2.0       │  Excellent: converged in < 14% of max    │
   │   1.0 - 2.0    │  Good: converged in 14-37% of max        │
   │   0.0 - 1.0    │  Moderate: converged in 37-100% of max   │
   │    < 0.0       │  Bad: hit max iterations (didn't converge)│
   └────────────────┴──────────────────────────────────────────┘
   
   • Higher reward = fewer iterations needed
   • Depends on max_iter setting (typically 500)
""")
    else:
        print(f"""
   Unknown reward type: '{reward_type}'
   
   General interpretation: Higher reward = better AMG quality
""")
    
    print_colored("─" * 60, Colors.CYAN)


def _detect_reward_type() -> str:
    """
    Detect reward type from training config file.
    Returns the reward function name used during training.
    """
    import yaml
    
    # Try to read from default training config
    config_path = Path(__file__).parent.parent / "configs" / "training" / "reinforce_default.yaml"
    
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                train_config = yaml.safe_load(f)
            return train_config.get('reward', {}).get('function', 'pcg_residual_reduction')
    except Exception:
        pass
    
    # Default fallback
    return 'pcg_residual_reduction'


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run case study experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all variations of case study 1
  python scripts/run_case_study.py --config case_studies/configs/study_1_anisotropic.yaml

  # Run only variation A
  python scripts/run_case_study.py --config case_studies/configs/study_1_anisotropic.yaml --variation A

  # Run on GPU
  python scripts/run_case_study.py --config ... --device cuda

  # Initialize from existing checkpoint (continues training)
  python scripts/run_case_study.py --config ... --init-from outputs/exp_xxx/best_model.pt

  # Skip training entirely (evaluation only)
  python scripts/run_case_study.py --config ... --checkpoint path/to/model.pt --skip-training
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to case study YAML config"
    )
    parser.add_argument(
        "--variation",
        type=str,
        default=None,
        # Note: choices validated later after loading config
        help="Specific variation to run (A, B, C, D, ... or 'all'). Default: all"
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        dest="init_from",
        help="Initialize model from existing checkpoint (still trains on new data)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: from config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for training/inference (cpu/cuda). Default: auto-detect CUDA"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        dest="skip_training",
        help="Skip training and only run evaluation (requires --init-from)"
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        dest="parallel_workers",
        help="Number of parallel workers for reward computation (1=sequential, 4-16 recommended on HPC)"
    )
    
    args = parser.parse_args()
    
    # Validate: --skip-training requires --init-from
    if args.skip_training and not args.init_from:
        parser.error("--skip-training requires --init-from to specify a checkpoint")
    
    return args


def main():
    args = parse_args()
    
    
    # Auto-detect CUDA if device not specified
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    from p_sparsity.case_studies import (
        CaseStudyConfig,
        CaseStudyRunner,
        run_case_study,
    )
    
    print_colored("=" * 70, Colors.CYAN, bold=True)
    print_colored("  P-Sparsity Case Study Runner", Colors.CYAN, bold=True)
    print_colored("=" * 70, Colors.CYAN, bold=True)
    
    # Load configuration
    print(f"\n📁 Loading config: {args.config}")
    config = CaseStudyConfig.from_yaml(args.config)
    
    print(f"   Case study: {config.name}")
    print(f"   Problem type: {config.problem.problem_type.value}")
    print(f"   Variations available: {list(config.variations.keys())}")
    
    # Determine which variations to run
    if args.variation and args.variation != 'all':
        # Validate variation exists in config
        if args.variation not in config.variations:
            print_colored(f"\n❌ Error: Unknown variation '{args.variation}'", Colors.RED, bold=True)
            print_colored(f"   Available variations: {list(config.variations.keys())}", Colors.YELLOW)
            sys.exit(1)
        variations_to_run = [args.variation]
    else:
        variations_to_run = list(config.variations.keys())
    
    # =========================================================================
    # Print detailed behavior confirmation BEFORE any computation
    # =========================================================================
    print_colored("\n" + "=" * 70, Colors.YELLOW)
    print_colored("  EXECUTION PLAN (please confirm)", Colors.YELLOW, bold=True)
    print_colored("=" * 70, Colors.YELLOW)
    
    for var_name in variations_to_run:
        var_config = config.variations[var_name]
        
        print_colored(f"\n📊 Variation {var_name}:", Colors.GREEN, bold=True)
        print(f"   {var_config.description}")
        
        # Training info
        print_colored("\n   🏋️  TRAINING:", Colors.BLUE, bold=True)
        
        # Check for random init (ablation study)
        if var_config.use_random_init:
            print_colored(f"      🎲 RANDOM BASELINE (no training)", Colors.RED, bold=True)
            print_colored(f"      Model will use random weights for ablation study", Colors.YELLOW)
        else:
            print(f"      Grid size: {var_config.train.grid_size}x{var_config.train.grid_size}")
            print(f"      Samples: {var_config.train.num_samples}")
            print(f"      Epochs: {var_config.train.epochs}")
            print(f"      Batch size: {var_config.train.batch_size}")
            print(f"      Learning rate: {var_config.train.learning_rate}")
            
            if args.skip_training:
                print_colored(f"      ⏭️  SKIPPING TRAINING (evaluation only)", Colors.YELLOW, bold=True)
                print_colored(f"      📦 Loading model from: {args.init_from}", Colors.YELLOW)
            elif args.init_from:
                print_colored(f"      ⚠️  Initializing weights from: {args.init_from}", Colors.YELLOW)
                print_colored(f"      ⚠️  Model will STILL BE TRAINED on new data!", Colors.YELLOW)
            else:
                print_colored(f"      ✨ Training from scratch (random initialization)", Colors.GREEN)
        
        # Testing info
        print_colored("\n   🧪 TESTING:", Colors.BLUE, bold=True)
        print(f"      Grid sizes: {var_config.test.grid_sizes}")
        print(f"      Samples per size: {var_config.test.num_samples_per_size}")
        print(f"      Total test problems: {len(var_config.test.grid_sizes) * var_config.test.num_samples_per_size}")
        if var_config.test.is_scaling_study:
            print_colored(f"      📈 This is a SCALING STUDY", Colors.CYAN)
    
    # Device info
    if args.device == "cuda":
        cuda_device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        print_colored(f"\n💻 Device: {args.device} ({cuda_device})", Colors.CYAN)
    else:
        print_colored(f"\n💻 Device: {args.device}", Colors.CYAN)
    
    # Parallel workers info
    if args.parallel_workers > 1:
        print_colored(f"⚡ Parallel workers: {args.parallel_workers} (faster reward computation)", Colors.CYAN)
    else:
        print_colored(f"⚡ Parallel workers: 1 (sequential, use --parallel-workers N for speedup)", Colors.YELLOW)
    
    # Output info
    output_path = Path(args.output) if args.output else Path(config.output_dir) / config.name
    print_colored(f"📂 Output directory: {output_path}", Colors.CYAN)
    
    # Print reward interpretation guide
    # Detect reward type from training config
    reward_type = _detect_reward_type()
    print_reward_explanation(reward_type)
    
    print_colored("\n" + "=" * 70, Colors.YELLOW)
    print_colored("  Starting computation...", Colors.GREEN, bold=True)
    print_colored("=" * 70 + "\n", Colors.YELLOW)
    
    # Create runner
    runner = CaseStudyRunner(
        config=config,
        output_dir=Path(args.output) if args.output else None,
        device=args.device,
        verbose=not args.quiet,
        parallel_workers=args.parallel_workers,
    )
    
    # Run variations
    results = {}
    for var_name in variations_to_run:
        results[var_name] = runner.run_variation(
            var_name,
            init_checkpoint=args.init_from,
            skip_training=args.skip_training,
        )
    
    # Print summary
    print_colored("\n" + "=" * 70, Colors.GREEN, bold=True)
    print_colored("  ✅ CASE STUDY COMPLETE", Colors.GREEN, bold=True)
    print_colored("=" * 70, Colors.GREEN, bold=True)
    
    for var_name, var_results in results.items():
        summary = var_results.summary()
        print_colored(f"\n📊 Variation {var_name}:", Colors.CYAN, bold=True)
        print(f"   Train samples: {summary['num_train_samples']}")
        print(f"   Test problems: {summary['num_test_problems']}")
        print(f"   Train time: {summary['train_time']:.1f}s")
        print(f"   Eval time: {summary['eval_time']:.1f}s")
        if summary['avg_speedup']:
            print_colored(f"   Avg speedup: {summary['avg_speedup']:.2f}x ± {summary['std_speedup']:.2f}", Colors.GREEN)
            print(f"   Range: [{summary['min_speedup']:.2f}x, {summary['max_speedup']:.2f}x]")
    
    print_colored(f"\n📂 Results saved to: {runner.output_dir}", Colors.CYAN)
    print_colored("=" * 70, Colors.GREEN, bold=True)


if __name__ == "__main__":
    main()
