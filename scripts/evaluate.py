#!/usr/bin/env python3
"""
Evaluation script for trained AMG policy models.

Usage:
    python scripts/evaluate.py --checkpoint path/to/checkpoint.pt --config configs/evaluation/default.yaml
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import yaml
from omegaconf import OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from p_sparsity.data import get_generator
from p_sparsity.models import build_policy_from_config
from p_sparsity.pyamg_interface import build_pyamg_solver, build_C_from_model
from p_sparsity.evaluation import (
    run_pcg_analysis,
    run_vcycle_analysis,
    run_eigenvalue_analysis,
    compare_pcg_performance,
    compare_vcycle_performance,
    compare_spectral_properties,
)
from p_sparsity.visualization import (
    plot_convergence_curves,
    plot_vcycle_comparison,
    plot_eigenvalue_spectrum,
    plot_C_comparison,
    plot_performance_comparison,
    create_comparison_table,
    # New plot functions
    plot_sparse_density_heatmap,
    plot_C_overlay_heatmap,
    plot_P_comparison,
    compute_C_overlap_stats,
    plot_iteration_bar,
    plot_eigenvalue_spectra_comparison,
    plot_pcg_convergence_comparison,
    plot_suite_summary,
)

import pyamg


def load_checkpoint(checkpoint_path: str, device: str = 'cpu'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model config and state dict
    model_config = checkpoint.get('model_config', {})
    model_state = checkpoint.get('model_state_dict', checkpoint.get('state_dict', {}))
    
    # Rebuild model using the config
    from omegaconf import OmegaConf
    if isinstance(model_config, dict):
        model_config = OmegaConf.create(model_config)
    model = build_policy_from_config(model_config)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    return model, checkpoint


def build_baseline_solver(A, B=None, max_levels=None, max_coarse=10):
    """Build standard PyAMG solver for comparison."""
    if B is None:
        B = np.ones((A.shape[0], 1))
    
    kwargs = {"B": B, "max_coarse": max_coarse}
    if max_levels is not None:
        kwargs["max_levels"] = max_levels
    
    ml = pyamg.smoothed_aggregation_solver(A, **kwargs)
    return ml


def build_tuned_solver(A, theta=0.25, B=None, max_levels=None, max_coarse=10):
    """Build tuned PyAMG solver with specified theta."""
    if B is None:
        B = np.ones((A.shape[0], 1))
    
    kwargs = {
        "strength": ('symmetric', {'theta': theta}),
        "B": B,
        "max_coarse": max_coarse,
    }
    if max_levels is not None:
        kwargs["max_levels"] = max_levels
    
    # Use symmetric strength with specified theta for ALL levels
    ml = pyamg.smoothed_aggregation_solver(A, **kwargs)
    
    # Also compute C for visualization
    from pyamg.strength import symmetric_strength_of_connection
    C_tuned = symmetric_strength_of_connection(A, theta=theta)
    
    return ml, C_tuned


def evaluate_single_problem(
    model,
    problem_data,
    eval_config,
    device: str = 'cpu',
    verbose: bool = True
):
    """
    Evaluate learned AMG on a single problem.
    
    Returns:
        dict with evaluation results
    """
    A = problem_data.A
    n = A.shape[0]
    
    # Build learned solver
    if verbose:
        print(f"Building learned AMG solver (n={n})...")
    
    # Get k_per_row from model config
    k_per_row = model.config.get('edges_per_row', 3)
    
    C_learned, B_learned_extra = build_C_from_model(
        A, problem_data.grid_n, model, k_per_row, device=device
    )
    
    # Prepare B for PyAMG
    B_learned = None
    if B_learned_extra is not None:
        # Prepend constant vector
        B_learned = np.column_stack([np.ones(n), B_learned_extra])
    
    # Use consistent max_levels for fair comparison (2 levels = fine + coarse)
    max_levels = 2
    max_coarse = 10
    
    ml_learned = build_pyamg_solver(A, C_learned, B_learned, max_levels=max_levels, max_coarse=max_coarse)
    
    # Build baseline solver with same hierarchy depth
    if verbose:
        print("Building baseline AMG solver...")
    ml_baseline = build_baseline_solver(A, B_learned, max_levels=max_levels, max_coarse=max_coarse)
    
    # Build tuned solver (theta=0.25) with same hierarchy depth
    if verbose:
        print("Building tuned AMG solver (θ=0.25)...")
    ml_tuned, C_tuned = build_tuned_solver(A, theta=0.25, B=B_learned, max_levels=max_levels, max_coarse=max_coarse)
    
    # Debug: print solver hierarchy info
    if verbose:
        print(f"\nSolver hierarchy info:")
        print(f"  Learned: {len(ml_learned.levels)} levels, coarse size = {ml_learned.levels[-1].A.shape[0]}")
        print(f"  Baseline: {len(ml_baseline.levels)} levels, coarse size = {ml_baseline.levels[-1].A.shape[0]}")
        print(f"  Tuned: {len(ml_tuned.levels)} levels, coarse size = {ml_tuned.levels[-1].A.shape[0]}")
    
    # For baseline, compute standard strength of connection
    # (PyAMG uses symmetric measure with theta=0.0 by default)
    from pyamg.strength import symmetric_strength_of_connection
    C_baseline = symmetric_strength_of_connection(A, theta=0.0)
    
    results = {
        'A': A,
        'C_learned': C_learned,
        'C_baseline': C_baseline,
        'C_tuned': C_tuned,
        'ml_learned': ml_learned,
        'ml_baseline': ml_baseline,
        'ml_tuned': ml_tuned,
    }
    
    # PCG analysis
    pcg_config = eval_config.get('pcg', {})
    if pcg_config.get('enabled', True):
        if verbose:
            print("\nRunning PCG analysis...")
        
        pcg_comparison = compare_pcg_performance(
            A, ml_learned, ml_baseline,
            ml_tuned=ml_tuned,
            num_trials=5,
            max_iter=pcg_config.get('max_iterations', 100),
            tol=pcg_config.get('tolerance', 1e-8),
            verbose=verbose,
            show_progress=True,
        )
        results['pcg'] = pcg_comparison
        
        if verbose:
            print("\nPCG Results:")
            print(f"  Learned:  {pcg_comparison['learned']['avg_iterations']:.1f} iterations")
            print(f"  Baseline: {pcg_comparison['baseline']['avg_iterations']:.1f} iterations")
            if 'tuned' in pcg_comparison:
                print(f"  Tuned:    {pcg_comparison['tuned']['avg_iterations']:.1f} iterations")
            print(f"  Speedup:  {pcg_comparison['speedup']:.2f}×")
    
    # V-cycle analysis
    vcycle_config = eval_config.get('vcycle', {})
    if vcycle_config.get('enabled', True):
        if verbose:
            print("\nRunning V-cycle analysis...")
        
        vcycle_comparison = compare_vcycle_performance(
            A, ml_learned, ml_baseline,
            num_trials=5,
            num_cycles=10,
            verbose=False
        )
        results['vcycle'] = vcycle_comparison
        
        if verbose:
            print("\nV-cycle Results:")
            print(f"  Learned:  {vcycle_comparison['learned']['avg_reduction_factor']:.4f} avg reduction")
            print(f"  Baseline: {vcycle_comparison['baseline']['avg_reduction_factor']:.4f} avg reduction")
            print(f"  Improvement: {vcycle_comparison['improvement']:.2f}×")
    
    # Eigenvalue analysis
    eigenvalue_config = eval_config.get('eigenvalue', {})
    if eigenvalue_config.get('enabled', True):
        method = eigenvalue_config.get('method', 'fast')  # 'fast' or 'eigenvalues'
        if verbose:
            method_name = "convergence-based" if method == 'fast' else "full eigenvalue"
            print(f"\nRunning spectral analysis ({method_name})...")
        
        spectral_comparison = compare_spectral_properties(
            A, ml_learned, ml_baseline,
            k=eigenvalue_config.get('num_eigenvalues', 20),
            method=method
        )
        results['spectral'] = spectral_comparison
        
        if verbose:
            metric_type = spectral_comparison['learned'].get('metric_type', 'condition_number')
            if metric_type == 'convergence_rate':
                print("\nSpectral Properties:")
                print(f"  Learned convergence rate:  {spectral_comparison['learned']['condition_number']:.3f}")
                print(f"  Baseline convergence rate: {spectral_comparison['baseline']['condition_number']:.3f}")
                print(f"  Improvement: {spectral_comparison['condition_improvement']:.2f}×")
            else:
                print("\nSpectral Properties:")
                print(f"  Learned condition number:  {spectral_comparison['learned']['condition_number']:.2e}")
                print(f"  Baseline condition number: {spectral_comparison['baseline']['condition_number']:.2e}")
                print(f"  Improvement: {spectral_comparison['condition_improvement']:.2f}×")
    
    return results


def generate_visualizations(results, output_dir: Path, problem_name: str):
    """Generate and save visualization plots like the reference script."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Generating visualizations for {problem_name}...")
    
    # 1. Sparsity comparison (existing)
    if 'C_learned' in results and 'C_baseline' in results:
        fig = plot_C_comparison(
            results['A'],
            results['C_learned'],
            results['C_baseline']
        )
        fig.savefig(output_dir / f"{problem_name}_sparsity.png", dpi=150)
        plt.close(fig)
        print(f"    Saved: {problem_name}_sparsity.png")
    
    # 2. C overlay heatmap (Red=Baseline, Blue=Learned, Purple=Both)
    if 'C_learned' in results and 'C_baseline' in results:
        fig = plot_C_overlay_heatmap(
            results['C_baseline'],
            results['C_learned'],
            title=f"Strength Matrix Overlay ({problem_name})"
        )
        fig.savefig(output_dir / f"{problem_name}_C_overlay.png", dpi=150)
        plt.close(fig)
        print(f"    Saved: {problem_name}_C_overlay.png")
        
        # Print overlap stats
        overlap_stats = compute_C_overlap_stats(results['C_baseline'], results['C_learned'])
        print(f"    C Overlap Stats: IoU={overlap_stats['iou']:.4f}, "
              f"Baseline edges={overlap_stats['n_baseline']}, "
              f"Learned edges={overlap_stats['n_learned']}")
    
    # 3. P (Prolongation) comparison heatmaps
    if 'ml_learned' in results and 'ml_baseline' in results:
        try:
            P_baseline = results['ml_baseline'].levels[0].P
            P_learned = results['ml_learned'].levels[0].P
            
            fig = plot_P_comparison(
                P_baseline, P_learned,
                title=f"Prolongation Operator Comparison ({problem_name})",
                crop=(2000, 2000) if P_baseline.shape[0] > 2000 else None
            )
            fig.savefig(output_dir / f"{problem_name}_P_comparison.png", dpi=150)
            plt.close(fig)
            print(f"    Saved: {problem_name}_P_comparison.png")
        except Exception as e:
            print(f"    Warning: Could not plot P comparison: {e}")
    
    # 4. PCG convergence curves
    if 'pcg' in results and 'residuals' in results['pcg'].get('learned', {}):
        residuals_baseline = results['pcg']['baseline'].get('residuals', [])
        residuals_learned = results['pcg']['learned'].get('residuals', [])
        residuals_tuned = results['pcg'].get('tuned', {}).get('residuals', None)
        
        if residuals_baseline and residuals_learned:
            fig = plot_pcg_convergence_comparison(
                residuals_baseline,
                residuals_learned,
                residuals_tuned=residuals_tuned,
                title=f"PCG Convergence ({problem_name})"
            )
            fig.savefig(output_dir / f"{problem_name}_pcg_convergence.png", dpi=150)
            plt.close(fig)
            print(f"    Saved: {problem_name}_pcg_convergence.png")
    
    # 5. Iteration bar chart
    if 'pcg' in results:
        iterations = {
            'Baseline': results['pcg']['baseline']['avg_iterations'],
            'Learned': results['pcg']['learned']['avg_iterations'],
        }
        if 'tuned' in results['pcg']:
            iterations['Tuned (θ=0.25)'] = results['pcg']['tuned']['avg_iterations']
        
        fig = plot_iteration_bar(
            {k: int(v) for k, v in iterations.items()},
            title=f"PCG Iterations ({problem_name})"
        )
        fig.savefig(output_dir / f"{problem_name}_iterations.png", dpi=150)
        plt.close(fig)
        print(f"    Saved: {problem_name}_iterations.png")
    
    # 6. Eigenvalue spectra (if available and small enough)
    if 'spectral' in results:
        spectral = results['spectral']
        if 'eigenvalues' in spectral.get('learned', {}) and 'eigenvalues' in spectral.get('baseline', {}):
            fig = plot_eigenvalue_spectra_comparison(
                spectral['baseline']['eigenvalues'],
                spectral['learned']['eigenvalues'],
                title=f"Eigenvalue Spectra ({problem_name})"
            )
            fig.savefig(output_dir / f"{problem_name}_spectra.png", dpi=150)
            plt.close(fig)
            print(f"    Saved: {problem_name}_spectra.png")


import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained AMG policy")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str,
                       default='configs/evaluation/default.yaml',
                       help='Path to evaluation config')
    parser.add_argument('--output', type=str, default='results/evaluation',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu/cuda)')
    parser.add_argument('--visualize', default=True, action='store_true',
                       help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # Load configs
    eval_config = OmegaConf.load(args.config)
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, checkpoint = load_checkpoint(args.checkpoint, device=args.device)
    print(f"  Trained for {checkpoint.get('epoch', 'unknown')} epochs")
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get test cases
    print("\nGenerating test problems...")
    test_cases = eval_config.test_cases
    all_results = {}
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"Evaluating: {test_case.name}")
        print(f"{'='*60}")
        
        # Create generator for this test case
        # Create a minimal config for the generator
        gen_config = OmegaConf.create({
            'type': test_case.problem_type,
            'params': test_case.params,
            'features': {
                'use_smooth_errors': True,
                'num_smoothing_steps': 1,
                'aggregation': 'mean',
            }
        })
        generator = get_generator(test_case.problem_type, gen_config)
        
        # Generate problem with test parameters
        # Create TrainSample format
        from p_sparsity.data.dataset import make_train_sample
        feature_config = {
            'use_smooth_errors': True,
            'num_smoothing_steps': 1,
            'aggregation': 'mean',
            'normalize_weights': True,
        }
        problem_data = make_train_sample(
            generator,
            test_case.grid_size,
            test_case.params,
            feature_config
        )
        
        # Evaluate
        results = evaluate_single_problem(
            model, problem_data, eval_config,
            device=args.device, verbose=False
        )
        all_results[test_case.name] = results
        
        # Generate visualizations
        if args.visualize:
            print("\nGenerating visualizations...")
            viz_dir = output_dir / "visualizations"
            generate_visualizations(results, viz_dir, test_case.name)
    
    # Aggregate results
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    # Create comparison table
    comparison_data = {}
    for name, results in all_results.items():
        comparison_data[name] = {}
        
        if 'pcg' in results:
            comparison_data[name]['PCG Iterations (Learned)'] = \
                results['pcg']['learned']['avg_iterations']
            comparison_data[name]['PCG Iterations (Baseline)'] = \
                results['pcg']['baseline']['avg_iterations']
            comparison_data[name]['PCG Speedup'] = results['pcg']['speedup']
        
        if 'vcycle' in results:
            comparison_data[name]['V-cycle Reduction (Learned)'] = \
                results['vcycle']['learned']['avg_reduction_factor']
            comparison_data[name]['V-cycle Reduction (Baseline)'] = \
                results['vcycle']['baseline']['avg_reduction_factor']
        
        if 'spectral' in results:
            # Use appropriate column name based on metric type
            metric_type = results['spectral']['learned'].get('metric_type', 'condition_number')
            if metric_type == 'convergence_rate':
                col_name = 'Convergence Rate'
                comparison_data[name][f'{col_name} (Learned)'] = \
                    results['spectral']['learned']['condition_number']
                comparison_data[name][f'{col_name} (Baseline)'] = \
                    results['spectral']['baseline']['condition_number']
            else:
                col_name = 'Condition Number'
                comparison_data[name][f'{col_name} (Learned)'] = \
                    results['spectral']['learned']['condition_number']
                comparison_data[name][f'{col_name} (Baseline)'] = \
                    results['spectral']['baseline']['condition_number']
    
    # Print table
    table = create_comparison_table(comparison_data, output_format='markdown')
    print(table)
    
    # Save results
    results_file = output_dir / "evaluation_results.yaml"
    with open(results_file, 'w') as f:
        yaml.dump(comparison_data, f)
    print(f"\nResults saved to: {results_file}")
    
    table_file = output_dir / "comparison_table.md"
    with open(table_file, 'w') as f:
        f.write(table)
    print(f"Table saved to: {table_file}")
    
    # Generate suite summary plot (like the reference script)
    if args.visualize and len(all_results) > 1:
        print("\nGenerating suite summary plot...")
        suite_results = []
        for name, results in all_results.items():
            if 'pcg' in results:
                result_dict = {
                    'tag': name,
                    'it_baseline': int(results['pcg']['baseline']['avg_iterations']),
                    'it_learned': int(results['pcg']['learned']['avg_iterations']),
                }
                if 'tuned' in results['pcg']:
                    result_dict['it_tuned'] = int(results['pcg']['tuned']['avg_iterations'])
                suite_results.append(result_dict)
        
        if suite_results:
            viz_dir = output_dir / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            fig = plot_suite_summary(suite_results, title="PCG Iterations Across Evaluation Suite")
            fig.savefig(viz_dir / "suite_pcg_iterations.png", dpi=150)
            plt.close(fig)
            print(f"  Saved: visualizations/suite_pcg_iterations.png")


if __name__ == '__main__':
    main()
