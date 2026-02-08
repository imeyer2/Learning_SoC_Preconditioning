"""
Comprehensive Exploratory Data Analysis (EDA) for case study results.

Generates:
1. Individual plots for each test problem
2. Aggregate statistical analysis
3. Parameter correlation analysis
4. Performance distribution analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


@dataclass
class EDAConfig:
    """Configuration for EDA generation."""
    # Individual problem plots
    plot_individual_residuals: bool = True
    plot_individual_energy: bool = True
    
    # Aggregate plots
    plot_iteration_distributions: bool = True
    plot_speedup_analysis: bool = True
    plot_parameter_correlations: bool = True
    plot_convergence_comparison: bool = True
    plot_statistical_summary: bool = True
    
    # Output settings
    dpi: int = 150
    figsize_individual: Tuple[int, int] = (12, 5)
    figsize_aggregate: Tuple[int, int] = (14, 10)
    
    # Analysis settings
    max_individual_plots: int = 100  # Cap on individual plots


def run_full_eda(
    results_path: Path,
    output_dir: Optional[Path] = None,
    config: Optional[EDAConfig] = None,
) -> Dict[str, Any]:
    """
    Run full EDA on results.json file.
    
    Args:
        results_path: Path to results.json
        output_dir: Output directory (default: eda/ in same dir)
        config: EDA configuration
        
    Returns:
        Dictionary with computed statistics
    """
    results_path = Path(results_path)
    config = config or EDAConfig()
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    if output_dir is None:
        output_dir = results_path.parent / "eda"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    variation_name = results.get('variation_name', results_path.parent.name)
    test_problems = results.get('test_problems', [])
    
    print(f"=" * 60)
    print(f"FULL EDA for {variation_name}")
    print(f"=" * 60)
    print(f"Test problems: {len(test_problems)}")
    print(f"Output: {output_dir}")
    print()
    
    stats = {}
    
    # 1. Individual problem plots
    if config.plot_individual_residuals or config.plot_individual_energy:
        print("[1/5] Generating individual problem plots...")
        individual_dir = output_dir / "individual_problems"
        individual_dir.mkdir(exist_ok=True)
        _generate_individual_plots(test_problems, individual_dir, config, variation_name)
    
    # 2. Iteration distribution analysis
    if config.plot_iteration_distributions:
        print("[2/5] Analyzing iteration distributions...")
        stats['iterations'] = _analyze_iterations(test_problems, output_dir, variation_name)
    
    # 3. Speedup analysis
    if config.plot_speedup_analysis:
        print("[3/5] Analyzing speedups...")
        stats['speedup'] = _analyze_speedups(test_problems, output_dir, variation_name)
    
    # 4. Parameter correlation analysis
    if config.plot_parameter_correlations:
        print("[4/5] Analyzing parameter correlations...")
        stats['correlations'] = _analyze_parameter_correlations(test_problems, output_dir, variation_name)
    
    # 5. Statistical summary
    if config.plot_statistical_summary:
        print("[5/5] Generating statistical summary...")
        stats['summary'] = _generate_statistical_summary(test_problems, output_dir, variation_name)
    
    # Save stats to JSON
    stats_path = output_dir / "eda_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nSaved statistics to {stats_path}")
    
    print(f"\n{'=' * 60}")
    print(f"EDA COMPLETE - All outputs in {output_dir}")
    print(f"{'=' * 60}")
    
    return stats


def _generate_individual_plots(
    test_problems: List[Dict],
    output_dir: Path,
    config: EDAConfig,
    variation_name: str,
) -> None:
    """Generate individual plots for each test problem."""
    n_problems = min(len(test_problems), config.max_individual_plots)
    
    for i, pm in enumerate(test_problems[:n_problems]):
        problem_id = pm.get('problem_id', f'problem_{i:04d}')
        grid_size = pm.get('grid_size', '?')
        params = pm.get('params', {})
        
        # Create problem-specific directory
        problem_dir = output_dir / problem_id
        problem_dir.mkdir(exist_ok=True)
        
        # Get solver data
        learned = pm.get('learned', {})
        baseline = pm.get('baseline', {})
        tuned = pm.get('tuned', {})
        
        # Plot residual convergence
        if config.plot_individual_residuals:
            _plot_problem_residuals(
                problem_id, grid_size, params,
                learned, baseline, tuned,
                problem_dir / "residual_convergence.png",
                config,
            )
        
        # Plot energy decay
        if config.plot_individual_energy:
            _plot_problem_energy(
                problem_id, grid_size, params,
                learned, baseline, tuned,
                problem_dir / "energy_decay.png",
                config,
            )
        
        # Save problem metadata
        meta = {
            'problem_id': problem_id,
            'grid_size': grid_size,
            'n': pm.get('n', grid_size**2 if isinstance(grid_size, int) else None),
            'params': params,
            'learned_iterations': learned.get('pcg', {}).get('iterations'),
            'baseline_iterations': baseline.get('pcg', {}).get('iterations'),
            'tuned_iterations': tuned.get('pcg', {}).get('iterations') if tuned else None,
            'learned_time': learned.get('pcg', {}).get('wall_time'),
            'baseline_time': baseline.get('pcg', {}).get('wall_time'),
        }
        with open(problem_dir / "metadata.json", 'w') as f:
            json.dump(meta, f, indent=2)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated plots for {i+1}/{n_problems} problems")
    
    print(f"  Saved individual plots to {output_dir}")


def _plot_problem_residuals(
    problem_id: str,
    grid_size: int,
    params: Dict,
    learned: Dict,
    baseline: Dict,
    tuned: Dict,
    save_path: Path,
    config: EDAConfig,
) -> None:
    """Plot residual convergence for a single problem."""
    fig, axes = plt.subplots(1, 2, figsize=config.figsize_individual)
    
    # Left: Residual history
    ax1 = axes[0]
    
    learned_res = learned.get('pcg', {}).get('residual_history', [])
    baseline_res = baseline.get('pcg', {}).get('residual_history', [])
    tuned_res = tuned.get('pcg', {}).get('residual_history', []) if tuned else []
    
    if learned_res:
        learned_norm = np.array(learned_res) / learned_res[0] if learned_res[0] > 0 else learned_res
        ax1.semilogy(learned_norm, 'b-', linewidth=2, label=f'Learned ({len(learned_res)-1} iters)')
    
    if baseline_res:
        baseline_norm = np.array(baseline_res) / baseline_res[0] if baseline_res[0] > 0 else baseline_res
        ax1.semilogy(baseline_norm, 'r--', linewidth=2, label=f'Baseline ({len(baseline_res)-1} iters)')
    
    if tuned_res:
        tuned_norm = np.array(tuned_res) / tuned_res[0] if tuned_res[0] > 0 else tuned_res
        ax1.semilogy(tuned_norm, 'g:', linewidth=2, label=f'Tuned ({len(tuned_res)-1} iters)')
    
    ax1.axhline(y=1e-8, color='gray', linestyle=':', alpha=0.5, label='Tolerance (1e-8)')
    ax1.set_xlabel('PCG Iteration', fontsize=11)
    ax1.set_ylabel('Relative Residual', fontsize=11)
    ax1.set_title('PCG Residual Convergence', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=1e-10)
    
    # Right: Convergence rate comparison
    ax2 = axes[1]
    
    # Compute convergence rates
    rates = []
    labels = []
    colors = []
    
    if len(learned_res) > 2:
        learned_rate = _compute_convergence_rate(learned_res)
        rates.append(learned_rate)
        labels.append('Learned')
        colors.append('blue')
    
    if len(baseline_res) > 2:
        baseline_rate = _compute_convergence_rate(baseline_res)
        rates.append(baseline_rate)
        labels.append('Baseline')
        colors.append('red')
    
    if len(tuned_res) > 2:
        tuned_rate = _compute_convergence_rate(tuned_res)
        rates.append(tuned_rate)
        labels.append('Tuned')
        colors.append('green')
    
    if rates:
        bars = ax2.bar(labels, rates, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Avg Convergence Rate', fontsize=11)
        ax2.set_title('Convergence Rate (lower = faster)', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Add value labels
        for bar, rate in zip(bars, rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{rate:.3f}', ha='center', fontsize=10)
    
    # Title with problem info
    param_str = ", ".join([f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" 
                          for k, v in params.items()])
    fig.suptitle(f'{problem_id} | Grid: {grid_size}×{grid_size} | {param_str}', 
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
    plt.close(fig)


def _plot_problem_energy(
    problem_id: str,
    grid_size: int,
    params: Dict,
    learned: Dict,
    baseline: Dict,
    tuned: Dict,
    save_path: Path,
    config: EDAConfig,
) -> None:
    """Plot energy decay for a single problem."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    learned_energy = learned.get('vcycle', {}).get('energy_history', [])
    baseline_energy = baseline.get('vcycle', {}).get('energy_history', [])
    tuned_energy = tuned.get('vcycle', {}).get('energy_history', []) if tuned else []
    
    has_data = False
    
    if learned_energy and len(learned_energy) > 1:
        learned_norm = np.array(learned_energy) / learned_energy[0]
        ax.semilogy(learned_norm, 'b-', linewidth=2, marker='o', markersize=4,
                   label=f'Learned')
        has_data = True
    
    if baseline_energy and len(baseline_energy) > 1:
        baseline_norm = np.array(baseline_energy) / baseline_energy[0]
        ax.semilogy(baseline_norm, 'r--', linewidth=2, marker='s', markersize=4,
                   label=f'Baseline')
        has_data = True
    
    if tuned_energy and len(tuned_energy) > 1:
        tuned_norm = np.array(tuned_energy) / tuned_energy[0]
        ax.semilogy(tuned_norm, 'g:', linewidth=2, marker='^', markersize=4,
                   label=f'Tuned')
        has_data = True
    
    if not has_data:
        ax.text(0.5, 0.5, 'No energy history data available',
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
    
    ax.set_xlabel('V-cycle Iteration', fontsize=11)
    ax.set_ylabel('Normalized Energy Norm', fontsize=11)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    param_str = ", ".join([f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" 
                          for k, v in params.items()])
    ax.set_title(f'{problem_id} | Grid: {grid_size}×{grid_size} | {param_str}', 
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
    plt.close(fig)


def _analyze_iterations(
    test_problems: List[Dict],
    output_dir: Path,
    variation_name: str,
) -> Dict[str, Any]:
    """Analyze iteration counts across all problems."""
    
    learned_iters = []
    baseline_iters = []
    tuned_iters = []
    grid_sizes = []
    
    for pm in test_problems:
        l_iter = pm.get('learned', {}).get('pcg', {}).get('iterations')
        b_iter = pm.get('baseline', {}).get('pcg', {}).get('iterations')
        t_iter = pm.get('tuned', {}).get('pcg', {}).get('iterations')
        gs = pm.get('grid_size')
        
        if l_iter is not None:
            learned_iters.append(l_iter)
        if b_iter is not None:
            baseline_iters.append(b_iter)
        if t_iter is not None:
            tuned_iters.append(t_iter)
        if gs is not None:
            grid_sizes.append(gs)
    
    # Create comprehensive iteration analysis figure
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Histogram comparison
    ax1 = fig.add_subplot(2, 3, 1)
    if learned_iters and baseline_iters:
        all_iters = learned_iters + baseline_iters
        bins = np.linspace(min(all_iters), max(all_iters), 25)
        ax1.hist(baseline_iters, bins=bins, alpha=0.6, label=f'Baseline (μ={np.mean(baseline_iters):.1f})', 
                color='red', edgecolor='darkred')
        ax1.hist(learned_iters, bins=bins, alpha=0.6, label=f'Learned (μ={np.mean(learned_iters):.1f})', 
                color='blue', edgecolor='darkblue')
        if tuned_iters:
            ax1.hist(tuned_iters, bins=bins, alpha=0.4, label=f'Tuned (μ={np.mean(tuned_iters):.1f})',
                    color='green', edgecolor='darkgreen')
    ax1.set_xlabel('PCG Iterations')
    ax1.set_ylabel('Count')
    ax1.set_title('Iteration Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    ax2 = fig.add_subplot(2, 3, 2)
    box_data = []
    box_labels = []
    if baseline_iters:
        box_data.append(baseline_iters)
        box_labels.append('Baseline')
    if learned_iters:
        box_data.append(learned_iters)
        box_labels.append('Learned')
    if tuned_iters:
        box_data.append(tuned_iters)
        box_labels.append('Tuned')
    
    if box_data:
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
        colors = ['red', 'blue', 'green'][:len(box_data)]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax2.set_ylabel('PCG Iterations')
    ax2.set_title('Iteration Box Plot')
    ax2.grid(True, alpha=0.3)
    
    # 3. Scatter: Learned vs Baseline iterations
    ax3 = fig.add_subplot(2, 3, 3)
    if learned_iters and baseline_iters and len(learned_iters) == len(baseline_iters):
        ax3.scatter(baseline_iters, learned_iters, alpha=0.6, c='purple', edgecolor='black', s=50)
        max_iter = max(max(baseline_iters), max(learned_iters))
        ax3.plot([0, max_iter], [0, max_iter], 'k--', alpha=0.5, label='y=x (equal)')
        ax3.set_xlabel('Baseline Iterations')
        ax3.set_ylabel('Learned Iterations')
        ax3.set_title('Learned vs Baseline')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add win/loss counts
        wins = sum(1 for l, b in zip(learned_iters, baseline_iters) if l < b)
        losses = sum(1 for l, b in zip(learned_iters, baseline_iters) if l > b)
        ties = len(learned_iters) - wins - losses
        ax3.text(0.05, 0.95, f'Wins: {wins}\nLosses: {losses}\nTies: {ties}',
                transform=ax3.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Iteration reduction histogram
    ax4 = fig.add_subplot(2, 3, 4)
    if learned_iters and baseline_iters and len(learned_iters) == len(baseline_iters):
        reductions = [(b - l) for l, b in zip(learned_iters, baseline_iters)]
        ax4.hist(reductions, bins=25, alpha=0.7, color='teal', edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No improvement')
        ax4.axvline(x=np.mean(reductions), color='blue', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(reductions):.1f}')
        ax4.set_xlabel('Iteration Reduction (Baseline - Learned)')
        ax4.set_ylabel('Count')
        ax4.set_title('Iteration Reduction Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Iterations by grid size
    ax5 = fig.add_subplot(2, 3, 5)
    if grid_sizes and learned_iters:
        unique_gs = sorted(set(grid_sizes))
        learned_by_gs = {gs: [] for gs in unique_gs}
        baseline_by_gs = {gs: [] for gs in unique_gs}
        
        for i, pm in enumerate(test_problems):
            gs = pm.get('grid_size')
            l_iter = pm.get('learned', {}).get('pcg', {}).get('iterations')
            b_iter = pm.get('baseline', {}).get('pcg', {}).get('iterations')
            if gs and l_iter:
                learned_by_gs[gs].append(l_iter)
            if gs and b_iter:
                baseline_by_gs[gs].append(b_iter)
        
        x = np.arange(len(unique_gs))
        width = 0.35
        
        learned_means = [np.mean(learned_by_gs[gs]) if learned_by_gs[gs] else 0 for gs in unique_gs]
        baseline_means = [np.mean(baseline_by_gs[gs]) if baseline_by_gs[gs] else 0 for gs in unique_gs]
        learned_stds = [np.std(learned_by_gs[gs]) if learned_by_gs[gs] else 0 for gs in unique_gs]
        baseline_stds = [np.std(baseline_by_gs[gs]) if baseline_by_gs[gs] else 0 for gs in unique_gs]
        
        ax5.bar(x - width/2, baseline_means, width, yerr=baseline_stds, label='Baseline',
               color='red', alpha=0.7, capsize=3)
        ax5.bar(x + width/2, learned_means, width, yerr=learned_stds, label='Learned',
               color='blue', alpha=0.7, capsize=3)
        ax5.set_xticks(x)
        ax5.set_xticklabels([f'{gs}×{gs}' for gs in unique_gs])
        ax5.set_xlabel('Grid Size')
        ax5.set_ylabel('Mean Iterations')
        ax5.set_title('Iterations by Grid Size')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Cumulative distribution
    ax6 = fig.add_subplot(2, 3, 6)
    if learned_iters and baseline_iters:
        learned_sorted = np.sort(learned_iters)
        baseline_sorted = np.sort(baseline_iters)
        learned_cdf = np.arange(1, len(learned_sorted) + 1) / len(learned_sorted)
        baseline_cdf = np.arange(1, len(baseline_sorted) + 1) / len(baseline_sorted)
        
        ax6.plot(baseline_sorted, baseline_cdf, 'r-', linewidth=2, label='Baseline')
        ax6.plot(learned_sorted, learned_cdf, 'b-', linewidth=2, label='Learned')
        ax6.set_xlabel('PCG Iterations')
        ax6.set_ylabel('Cumulative Probability')
        ax6.set_title('CDF of Iterations')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    fig.suptitle(f'{variation_name}: Iteration Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / "iteration_analysis.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Compute statistics
    stats = {
        'learned': {
            'mean': float(np.mean(learned_iters)) if learned_iters else None,
            'std': float(np.std(learned_iters)) if learned_iters else None,
            'median': float(np.median(learned_iters)) if learned_iters else None,
            'min': int(min(learned_iters)) if learned_iters else None,
            'max': int(max(learned_iters)) if learned_iters else None,
        },
        'baseline': {
            'mean': float(np.mean(baseline_iters)) if baseline_iters else None,
            'std': float(np.std(baseline_iters)) if baseline_iters else None,
            'median': float(np.median(baseline_iters)) if baseline_iters else None,
            'min': int(min(baseline_iters)) if baseline_iters else None,
            'max': int(max(baseline_iters)) if baseline_iters else None,
        },
        'improvement': {
            'mean_reduction': float(np.mean(baseline_iters) - np.mean(learned_iters)) if learned_iters and baseline_iters else None,
            'win_rate': float(sum(1 for l, b in zip(learned_iters, baseline_iters) if l < b) / len(learned_iters)) if learned_iters and baseline_iters else None,
        }
    }
    
    print(f"  Saved: iteration_analysis.png")
    return stats


def _analyze_speedups(
    test_problems: List[Dict],
    output_dir: Path,
    variation_name: str,
) -> Dict[str, Any]:
    """Analyze speedups across all problems."""
    
    speedups = []
    iter_ratios = []
    params_list = []
    grid_sizes = []
    
    for pm in test_problems:
        learned = pm.get('learned', {}).get('pcg', {})
        baseline = pm.get('baseline', {}).get('pcg', {})
        
        l_time = learned.get('wall_time')
        b_time = baseline.get('wall_time')
        l_iter = learned.get('iterations')
        b_iter = baseline.get('iterations')
        
        if l_time and b_time and l_time > 0:
            speedups.append(b_time / l_time)
        if l_iter and b_iter and l_iter > 0:
            iter_ratios.append(b_iter / l_iter)
        
        params_list.append(pm.get('params', {}))
        grid_sizes.append(pm.get('grid_size'))
    
    if not speedups:
        print("  Warning: No speedup data available")
        return {}
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Speedup histogram
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.hist(speedups, bins=25, alpha=0.7, color='green', edgecolor='darkgreen')
    ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='No speedup')
    ax1.axvline(x=np.mean(speedups), color='blue', linestyle='-', linewidth=2,
               label=f'Mean: {np.mean(speedups):.2f}x')
    ax1.set_xlabel('Speedup (Baseline Time / Learned Time)')
    ax1.set_ylabel('Count')
    ax1.set_title('Speedup Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Iteration ratio histogram
    ax2 = fig.add_subplot(2, 3, 2)
    if iter_ratios:
        ax2.hist(iter_ratios, bins=25, alpha=0.7, color='purple', edgecolor='black')
        ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='No improvement')
        ax2.axvline(x=np.mean(iter_ratios), color='blue', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(iter_ratios):.2f}x')
        ax2.set_xlabel('Iteration Ratio (Baseline / Learned)')
        ax2.set_ylabel('Count')
        ax2.set_title('Iteration Ratio Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Speedup vs iteration ratio
    ax3 = fig.add_subplot(2, 3, 3)
    if iter_ratios and len(speedups) == len(iter_ratios):
        ax3.scatter(iter_ratios, speedups, alpha=0.6, c='teal', edgecolor='black', s=50)
        ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        ax3.axvline(x=1.0, color='red', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Iteration Ratio')
        ax3.set_ylabel('Wall Time Speedup')
        ax3.set_title('Speedup vs Iteration Ratio')
        ax3.grid(True, alpha=0.3)
        
        # Fit line
        if len(iter_ratios) > 2:
            z = np.polyfit(iter_ratios, speedups, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(iter_ratios), max(iter_ratios), 100)
            ax3.plot(x_line, p(x_line), 'b--', alpha=0.5, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
            ax3.legend()
    
    # 4. Speedup by grid size
    ax4 = fig.add_subplot(2, 3, 4)
    if grid_sizes:
        unique_gs = sorted(set(gs for gs in grid_sizes if gs is not None))
        speedup_by_gs = {gs: [] for gs in unique_gs}
        
        for gs, sp in zip(grid_sizes, speedups):
            if gs is not None:
                speedup_by_gs[gs].append(sp)
        
        x = np.arange(len(unique_gs))
        means = [np.mean(speedup_by_gs[gs]) if speedup_by_gs[gs] else 0 for gs in unique_gs]
        stds = [np.std(speedup_by_gs[gs]) if speedup_by_gs[gs] else 0 for gs in unique_gs]
        
        bars = ax4.bar(x, means, yerr=stds, alpha=0.7, color='green', capsize=3, edgecolor='black')
        ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=2)
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'{gs}×{gs}' for gs in unique_gs])
        ax4.set_xlabel('Grid Size')
        ax4.set_ylabel('Mean Speedup')
        ax4.set_title('Speedup by Grid Size')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, mean in zip(bars, means):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{mean:.2f}x', ha='center', fontsize=9)
    
    # 5. Speedup percentiles
    ax5 = fig.add_subplot(2, 3, 5)
    percentiles = [5, 25, 50, 75, 95]
    pct_values = [np.percentile(speedups, p) for p in percentiles]
    ax5.bar(percentiles, pct_values, width=10, alpha=0.7, color='orange', edgecolor='black')
    ax5.axhline(y=1.0, color='red', linestyle='--', linewidth=2)
    ax5.set_xlabel('Percentile')
    ax5.set_ylabel('Speedup')
    ax5.set_title('Speedup Percentiles')
    ax5.grid(True, alpha=0.3, axis='y')
    
    for p, v in zip(percentiles, pct_values):
        ax5.text(p, v + 0.05, f'{v:.2f}x', ha='center', fontsize=9)
    
    # 6. Summary stats text
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""
    SPEEDUP STATISTICS
    ══════════════════════════════════
    
    Sample Size:     {len(speedups)}
    
    Mean Speedup:    {np.mean(speedups):.3f}x
    Std Dev:         {np.std(speedups):.3f}
    Median:          {np.median(speedups):.3f}x
    
    Min:             {min(speedups):.3f}x
    Max:             {max(speedups):.3f}x
    
    % with speedup > 1.0:  {100 * sum(1 for s in speedups if s > 1.0) / len(speedups):.1f}%
    % with speedup > 1.5:  {100 * sum(1 for s in speedups if s > 1.5) / len(speedups):.1f}%
    % with speedup > 2.0:  {100 * sum(1 for s in speedups if s > 2.0) / len(speedups):.1f}%
    """
    
    ax6.text(0.1, 0.5, stats_text, transform=ax6.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.suptitle(f'{variation_name}: Speedup Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / "speedup_analysis.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    stats = {
        'mean': float(np.mean(speedups)),
        'std': float(np.std(speedups)),
        'median': float(np.median(speedups)),
        'min': float(min(speedups)),
        'max': float(max(speedups)),
        'pct_above_1': float(sum(1 for s in speedups if s > 1.0) / len(speedups)),
        'pct_above_1_5': float(sum(1 for s in speedups if s > 1.5) / len(speedups)),
        'pct_above_2': float(sum(1 for s in speedups if s > 2.0) / len(speedups)),
    }
    
    print(f"  Saved: speedup_analysis.png")
    return stats


def _analyze_parameter_correlations(
    test_problems: List[Dict],
    output_dir: Path,
    variation_name: str,
) -> Dict[str, Any]:
    """Analyze correlations between parameters and performance."""
    
    # Extract parameters and metrics
    data = []
    for pm in test_problems:
        params = pm.get('params', {})
        learned = pm.get('learned', {}).get('pcg', {})
        baseline = pm.get('baseline', {}).get('pcg', {})
        
        row = {**params}
        row['learned_iters'] = learned.get('iterations')
        row['baseline_iters'] = baseline.get('iterations')
        row['grid_size'] = pm.get('grid_size')
        
        if learned.get('wall_time') and baseline.get('wall_time'):
            row['speedup'] = baseline['wall_time'] / learned['wall_time']
        if learned.get('iterations') and baseline.get('iterations'):
            row['iter_ratio'] = baseline['iterations'] / learned['iterations']
        
        data.append(row)
    
    if not data:
        return {}
    
    # Get parameter names
    param_names = [k for k in data[0].keys() 
                   if k not in ['learned_iters', 'baseline_iters', 'speedup', 'iter_ratio', 'grid_size']]
    
    if not param_names:
        print("  Warning: No parameters found for correlation analysis")
        return {}
    
    n_params = len(param_names)
    fig = plt.figure(figsize=(6 * min(n_params, 3), 5 * ((n_params + 2) // 3 + 1)))
    
    correlations = {}
    
    for i, param in enumerate(param_names):
        param_values = [d.get(param) for d in data if d.get(param) is not None]
        speedups = [d.get('speedup') for d in data if d.get(param) is not None and d.get('speedup') is not None]
        learned_iters = [d.get('learned_iters') for d in data if d.get(param) is not None and d.get('learned_iters') is not None]
        
        if len(param_values) < 3 or len(speedups) < 3:
            continue
        
        # Align arrays
        valid_data = [(d.get(param), d.get('speedup'), d.get('learned_iters')) 
                      for d in data 
                      if d.get(param) is not None and d.get('speedup') is not None]
        
        if len(valid_data) < 3:
            continue
        
        p_vals, sp_vals, li_vals = zip(*valid_data)
        
        ax = fig.add_subplot((n_params + 2) // 3 + 1, min(n_params, 3), i + 1)
        
        scatter = ax.scatter(p_vals, sp_vals, alpha=0.6, c=li_vals, cmap='viridis',
                            edgecolor='black', s=50)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel(param)
        ax.set_ylabel('Speedup')
        ax.set_title(f'Speedup vs {param}')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for learned iterations
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Learned Iters', fontsize=8)
        
        # Compute correlation
        if len(p_vals) > 2:
            corr = np.corrcoef(p_vals, sp_vals)[0, 1]
            correlations[param] = float(corr)
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle(f'{variation_name}: Parameter Correlations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / "parameter_correlations.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: parameter_correlations.png")
    return {'correlations': correlations}


def _generate_statistical_summary(
    test_problems: List[Dict],
    output_dir: Path,
    variation_name: str,
) -> Dict[str, Any]:
    """Generate comprehensive statistical summary."""
    
    # Collect all metrics
    learned_iters = []
    baseline_iters = []
    learned_times = []
    baseline_times = []
    convergence_rates_learned = []
    convergence_rates_baseline = []
    
    for pm in test_problems:
        learned = pm.get('learned', {}).get('pcg', {})
        baseline = pm.get('baseline', {}).get('pcg', {})
        
        if learned.get('iterations'):
            learned_iters.append(learned['iterations'])
        if baseline.get('iterations'):
            baseline_iters.append(baseline['iterations'])
        if learned.get('wall_time'):
            learned_times.append(learned['wall_time'])
        if baseline.get('wall_time'):
            baseline_times.append(baseline['wall_time'])
        
        # Compute convergence rates
        l_res = learned.get('residual_history', [])
        b_res = baseline.get('residual_history', [])
        
        if len(l_res) > 2:
            convergence_rates_learned.append(_compute_convergence_rate(l_res))
        if len(b_res) > 2:
            convergence_rates_baseline.append(_compute_convergence_rate(b_res))
    
    # Create summary figure
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Convergence rate comparison
    ax1 = fig.add_subplot(2, 2, 1)
    if convergence_rates_learned and convergence_rates_baseline:
        data = [convergence_rates_baseline, convergence_rates_learned]
        bp = ax1.boxplot(data, labels=['Baseline', 'Learned'], patch_artist=True)
        bp['boxes'][0].set_facecolor('red')
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor('blue')
        bp['boxes'][1].set_alpha(0.6)
        ax1.set_ylabel('Convergence Rate')
        ax1.set_title('Convergence Rate Comparison (lower = faster)')
        ax1.grid(True, alpha=0.3)
    
    # 2. Time comparison
    ax2 = fig.add_subplot(2, 2, 2)
    if learned_times and baseline_times:
        data = [baseline_times, learned_times]
        bp = ax2.boxplot(data, labels=['Baseline', 'Learned'], patch_artist=True)
        bp['boxes'][0].set_facecolor('red')
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor('blue')
        bp['boxes'][1].set_alpha(0.6)
        ax2.set_ylabel('Wall Time (s)')
        ax2.set_title('Solve Time Comparison')
        ax2.grid(True, alpha=0.3)
    
    # 3. Summary table
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis('off')
    
    # Build summary table
    table_data = []
    headers = ['Metric', 'Learned', 'Baseline', 'Improvement']
    
    if learned_iters and baseline_iters:
        l_mean = np.mean(learned_iters)
        b_mean = np.mean(baseline_iters)
        table_data.append(['Iterations (mean)', f'{l_mean:.1f}', f'{b_mean:.1f}', 
                          f'{100*(b_mean-l_mean)/b_mean:.1f}%'])
        table_data.append(['Iterations (std)', f'{np.std(learned_iters):.1f}', 
                          f'{np.std(baseline_iters):.1f}', '-'])
    
    if learned_times and baseline_times:
        l_mean = np.mean(learned_times)
        b_mean = np.mean(baseline_times)
        table_data.append(['Time (mean)', f'{l_mean:.4f}s', f'{b_mean:.4f}s',
                          f'{b_mean/l_mean:.2f}x'])
    
    if convergence_rates_learned and convergence_rates_baseline:
        l_mean = np.mean(convergence_rates_learned)
        b_mean = np.mean(convergence_rates_baseline)
        table_data.append(['Conv. Rate (mean)', f'{l_mean:.4f}', f'{b_mean:.4f}',
                          f'{100*(b_mean-l_mean)/b_mean:.1f}%'])
    
    if table_data:
        table = ax3.table(cellText=table_data, colLabels=headers, loc='center',
                         cellLoc='center', colColours=['lightgray']*4)
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
    ax3.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    # 4. Win/Loss analysis
    ax4 = fig.add_subplot(2, 2, 4)
    if learned_iters and baseline_iters and len(learned_iters) == len(baseline_iters):
        wins = sum(1 for l, b in zip(learned_iters, baseline_iters) if l < b)
        losses = sum(1 for l, b in zip(learned_iters, baseline_iters) if l > b)
        ties = len(learned_iters) - wins - losses
        
        sizes = [wins, losses, ties]
        labels = [f'Learned Wins\n({wins})', f'Baseline Wins\n({losses})', f'Ties\n({ties})']
        colors = ['green', 'red', 'gray']
        explode = (0.05, 0.05, 0)
        
        ax4.pie(sizes, labels=labels, colors=colors, explode=explode,
               autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
        ax4.set_title('Win/Loss Analysis (by iterations)')
    
    fig.suptitle(f'{variation_name}: Statistical Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / "statistical_summary.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Build stats dict
    stats = {
        'n_problems': len(test_problems),
        'learned': {
            'iterations_mean': float(np.mean(learned_iters)) if learned_iters else None,
            'iterations_std': float(np.std(learned_iters)) if learned_iters else None,
            'time_mean': float(np.mean(learned_times)) if learned_times else None,
            'convergence_rate_mean': float(np.mean(convergence_rates_learned)) if convergence_rates_learned else None,
        },
        'baseline': {
            'iterations_mean': float(np.mean(baseline_iters)) if baseline_iters else None,
            'iterations_std': float(np.std(baseline_iters)) if baseline_iters else None,
            'time_mean': float(np.mean(baseline_times)) if baseline_times else None,
            'convergence_rate_mean': float(np.mean(convergence_rates_baseline)) if convergence_rates_baseline else None,
        },
    }
    
    if learned_iters and baseline_iters:
        stats['win_rate'] = float(sum(1 for l, b in zip(learned_iters, baseline_iters) if l < b) / len(learned_iters))
    
    print(f"  Saved: statistical_summary.png")
    return stats


def _compute_convergence_rate(residuals: List[float]) -> float:
    """Compute average convergence rate from residual history."""
    if len(residuals) < 2:
        return 1.0
    
    rates = []
    for i in range(1, len(residuals)):
        if residuals[i-1] > 0:
            rate = residuals[i] / residuals[i-1]
            if 0 < rate < 1:
                rates.append(rate)
    
    return float(np.mean(rates)) if rates else 1.0


# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run full EDA on results.json")
    parser.add_argument("results_path", type=str, help="Path to results.json")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory")
    parser.add_argument("--max-individual", type=int, default=100, 
                       help="Max individual problem plots")
    
    args = parser.parse_args()
    
    config = EDAConfig(max_individual_plots=args.max_individual)
    run_full_eda(Path(args.results_path), Path(args.output) if args.output else None, config)
