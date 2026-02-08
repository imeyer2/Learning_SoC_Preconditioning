#!/usr/bin/env python
"""
Visualize training diagnostics from saved JSON files.

Usage:
    python scripts/visualize_diagnostics.py path/to/diagnostics/training_diagnostics.json
    python scripts/visualize_diagnostics.py path/to/experiment_dir
    
References for diagnostic metrics:
    - Policy Entropy: "Trust Region Policy Optimization" (Schulman et al., 2015)
    - Explained Variance: "High-Dimensional Continuous Control Using Generalized 
      Advantage Estimation" (Schulman et al., 2015)
    - Gradient Flow: Common practice in deep learning optimization
    - Over-smoothing: "Measuring and Relieving the Over-smoothing Problem for GNNs"
      (Chen et al., 2020)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.p_sparsity.utils.diagnostics import TrainingDiagnostics


def main():
    parser = argparse.ArgumentParser(
        description="Visualize GNN/RL training diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # From a specific diagnostics file
    python scripts/visualize_diagnostics.py outputs/exp_xxx/diagnostics/training_diagnostics.json
    
    # From an experiment directory (will search for diagnostics file)
    python scripts/visualize_diagnostics.py outputs/exp_xxx
    
    # Open plots interactively
    python scripts/visualize_diagnostics.py outputs/exp_xxx --show
"""
    )
    
    parser.add_argument(
        "path",
        type=str,
        help="Path to diagnostics JSON file or experiment directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for plots (default: same as input)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively (default: just save)"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary of final epoch diagnostics"
    )
    
    args = parser.parse_args()
    
    # Find diagnostics file
    path = Path(args.path)
    
    if path.is_file() and path.suffix == ".json":
        diag_file = path
    elif path.is_dir():
        # Search for diagnostics file
        candidates = [
            path / "diagnostics" / "training_diagnostics.json",
            path / "training_diagnostics.json",
        ]
        diag_file = None
        for c in candidates:
            if c.exists():
                diag_file = c
                break
        
        if diag_file is None:
            print(f"Error: Could not find training_diagnostics.json in {path}")
            sys.exit(1)
    else:
        print(f"Error: {path} is not a valid file or directory")
        sys.exit(1)
    
    print(f"Loading diagnostics from: {diag_file}")
    
    # Load diagnostics
    diagnostics = TrainingDiagnostics.load(str(diag_file))
    
    print(f"Loaded {len(diagnostics.epochs)} epochs of diagnostics")
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = diag_file.parent / "plots"
    
    # Generate plots
    print(f"Generating plots to: {output_dir}")
    diagnostics.plot_all(str(output_dir), show=args.show)
    
    # Print summary if requested
    if args.summary:
        print("\n" + "=" * 60)
        print("FINAL EPOCH DIAGNOSTICS SUMMARY")
        print("=" * 60)
        diagnostics.print_epoch_summary(-1)
        
        # Also print some interpretations
        if diagnostics.epochs:
            e = diagnostics.epochs[-1]
            
            print("\n📋 INTERPRETATION:")
            print("-" * 40)
            
            # Policy entropy interpretation
            print(f"\n🎲 Policy Entropy: {e.policy_entropy:.4f}")
            if e.policy_entropy < 0.5:
                print("   → Very low! Policy is highly deterministic.")
                print("   → Consider: temperature too low, or learning collapsed")
            elif e.policy_entropy > 2.0:
                print("   → High exploration, policy still quite random")
            else:
                print("   → Moderate - policy has learned preferences")
            
            # Explained variance interpretation
            print(f"\n📈 Value Explained Variance: {e.value_explained_var:.4f}")
            if e.value_explained_var > 0.8:
                print("   → Excellent! Baseline predicts rewards well")
            elif e.value_explained_var > 0.5:
                print("   → Good baseline quality")
            elif e.value_explained_var > 0:
                print("   → Baseline helps but could be better")
            else:
                print("   → Poor! Baseline is not helpful")
                print("   → Consider: increasing baseline momentum")
            
            # Gradient health
            print(f"\n🔧 Gradient Norms: Total={e.grad_norm_total:.4f}, GNN={e.grad_norm_gnn:.4f}, Policy={e.grad_norm_policy:.4f}")
            if e.grad_norm_total > 100:
                print("   → Very large gradients! Consider gradient clipping")
            elif e.grad_norm_total < 1e-6:
                print("   → Vanishing gradients! Learning may have stalled")
            else:
                print("   → Gradient magnitudes look healthy")
            
            # Advantage statistics
            print(f"\n📊 Advantage: mean={e.advantage_mean:.4f}, std={e.advantage_std:.4f}")
            if e.advantage_std > abs(e.advantage_mean) * 5:
                print("   → High variance in advantages")
                print("   → Consider: better baseline or reward shaping")


if __name__ == "__main__":
    main()
