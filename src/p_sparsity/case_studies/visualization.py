"""
Results visualization for case studies.

Generates plots for:
- Scaling curves (iterations vs grid size)
- Energy decay curves
- Iteration histograms
- Comparison tables
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Any
import json


def plot_scaling_curves(
    scaling_data: Dict[str, Any],
    title: str = "Scaling Study",
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Plot iterations and wall time vs grid size.
    
    Args:
        scaling_data: Scaling metrics from case study
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    summary = scaling_data.get('summary', {})
    
    grid_sizes = sorted([int(gs) for gs in summary.keys()])
    n_values = [summary[str(gs)]['n'] for gs in grid_sizes]
    
    # Extract data
    learned_iters = [summary[str(gs)].get('learned_iters_mean', np.nan) for gs in grid_sizes]
    baseline_iters = [summary[str(gs)].get('baseline_iters_mean', np.nan) for gs in grid_sizes]
    learned_times = [summary[str(gs)].get('learned_time_mean', np.nan) for gs in grid_sizes]
    baseline_times = [summary[str(gs)].get('baseline_time_mean', np.nan) for gs in grid_sizes]
    speedups = [summary[str(gs)].get('speedup_mean', np.nan) for gs in grid_sizes]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Iterations vs grid size
    ax1 = axes[0]
    ax1.loglog(n_values, learned_iters, 'b-o', label='Learned', linewidth=2, markersize=8)
    ax1.loglog(n_values, baseline_iters, 'r--s', label='Baseline', linewidth=2, markersize=8)
    ax1.set_xlabel('DOFs (n)', fontsize=12)
    ax1.set_ylabel('PCG Iterations', fontsize=12)
    ax1.set_title('Iterations vs Problem Size', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Wall time vs grid size
    ax2 = axes[1]
    ax2.loglog(n_values, learned_times, 'b-o', label='Learned', linewidth=2, markersize=8)
    ax2.loglog(n_values, baseline_times, 'r--s', label='Baseline', linewidth=2, markersize=8)
    ax2.set_xlabel('DOFs (n)', fontsize=12)
    ax2.set_ylabel('Wall Time (s)', fontsize=12)
    ax2.set_title('Solve Time vs Problem Size', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Speedup vs grid size
    ax3 = axes[2]
    ax3.semilogx(n_values, speedups, 'g-^', linewidth=2, markersize=8)
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No speedup')
    ax3.set_xlabel('DOFs (n)', fontsize=12)
    ax3.set_ylabel('Speedup (baseline/learned)', fontsize=12)
    ax3.set_title('Speedup vs Problem Size', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_energy_decay(
    problem_metrics: List[Dict[str, Any]],
    title: str = "Energy Norm Decay",
    save_path: Optional[Path] = None,
    max_problems: int = 5,
) -> plt.Figure:
    """
    Plot energy norm decay curves for selected problems.
    
    Args:
        problem_metrics: List of problem metric dictionaries
        title: Plot title
        save_path: Path to save figure
        max_problems: Maximum number of problems to plot
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(problem_metrics), max_problems)))
    
    for i, pm in enumerate(problem_metrics[:max_problems]):
        grid_size = pm.get('grid_size', '?')
        
        # Get energy histories
        learned_energy = pm.get('learned', {}).get('vcycle', {}).get('energy_history', [])
        baseline_energy = pm.get('baseline', {}).get('vcycle', {}).get('energy_history', [])
        
        if learned_energy:
            # Normalize
            learned_norm = np.array(learned_energy) / learned_energy[0]
            ax.semilogy(learned_norm, '-', color=colors[i], 
                       label=f'Learned (n={grid_size}²)', linewidth=2)
        
        if baseline_energy:
            baseline_norm = np.array(baseline_energy) / baseline_energy[0]
            ax.semilogy(baseline_norm, '--', color=colors[i], alpha=0.6,
                       label=f'Baseline (n={grid_size}²)', linewidth=1.5)
    
    ax.set_xlabel('V-cycle Iteration', fontsize=12)
    ax.set_ylabel('Normalized Energy Norm', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_iteration_histogram(
    problem_metrics: List[Dict[str, Any]],
    title: str = "PCG Iteration Distribution",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot histogram of PCG iterations.
    
    Args:
        problem_metrics: List of problem metric dictionaries
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    learned_iters = []
    baseline_iters = []
    
    for pm in problem_metrics:
        l_iters = pm.get('learned', {}).get('pcg', {}).get('iterations')
        b_iters = pm.get('baseline', {}).get('pcg', {}).get('iterations')
        
        if l_iters is not None:
            learned_iters.append(l_iters)
        if b_iters is not None:
            baseline_iters.append(b_iters)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.linspace(
        min(min(learned_iters, default=0), min(baseline_iters, default=0)),
        max(max(learned_iters, default=1), max(baseline_iters, default=1)) * 1.1,
        20
    )
    
    ax.hist(baseline_iters, bins=bins, alpha=0.7, label='Baseline', color='red', edgecolor='darkred')
    ax.hist(learned_iters, bins=bins, alpha=0.7, label='Learned', color='blue', edgecolor='darkblue')
    
    ax.axvline(np.mean(learned_iters), color='blue', linestyle='--', linewidth=2,
               label=f'Learned mean: {np.mean(learned_iters):.1f}')
    ax.axvline(np.mean(baseline_iters), color='red', linestyle='--', linewidth=2,
               label=f'Baseline mean: {np.mean(baseline_iters):.1f}')
    
    ax.set_xlabel('PCG Iterations', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_residual_curves(
    problem_metrics: List[Dict[str, Any]],
    title: str = "PCG Residual History",
    save_path: Optional[Path] = None,
    max_problems: int = 3,
) -> plt.Figure:
    """
    Plot PCG residual convergence curves.
    
    Args:
        problem_metrics: List of problem metric dictionaries
        title: Plot title
        save_path: Path to save figure
        max_problems: Maximum number of problems to plot
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set1(np.linspace(0, 1, min(len(problem_metrics), max_problems)))
    
    for i, pm in enumerate(problem_metrics[:max_problems]):
        problem_id = pm.get('problem_id', f'Problem {i}')
        
        learned_res = pm.get('learned', {}).get('pcg', {}).get('residual_history', [])
        baseline_res = pm.get('baseline', {}).get('pcg', {}).get('residual_history', [])
        
        if learned_res:
            # Normalize by initial residual
            learned_norm = np.array(learned_res) / learned_res[0]
            ax.semilogy(learned_norm, '-', color=colors[i], 
                       label=f'Learned ({problem_id})', linewidth=2)
        
        if baseline_res:
            baseline_norm = np.array(baseline_res) / baseline_res[0]
            ax.semilogy(baseline_norm, '--', color=colors[i], alpha=0.6,
                       label=f'Baseline ({problem_id})', linewidth=1.5)
    
    ax.axhline(y=1e-8, color='gray', linestyle=':', alpha=0.5, label='Tolerance')
    ax.set_xlabel('PCG Iteration', fontsize=12)
    ax.set_ylabel('Relative Residual', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def generate_comparison_table(
    results: Dict[str, Any],
    save_path: Optional[Path] = None,
) -> str:
    """
    Generate markdown comparison table from results.
    
    Args:
        results: Combined results dictionary
        save_path: Path to save markdown file
        
    Returns:
        Markdown table string
    """
    lines = []
    lines.append("# Case Study Results Comparison\n")
    lines.append(f"**Study:** {results.get('config', {}).get('name', 'Unknown')}\n")
    lines.append(f"**Problem Type:** {results.get('config', {}).get('problem_type', 'Unknown')}\n")
    
    for var_name, var_data in results.get('variations', {}).items():
        summary = var_data.get('summary', {})
        
        lines.append(f"\n## Variation {var_name}\n")
        lines.append(f"- Train samples: {summary.get('num_train_samples', 'N/A')}")
        lines.append(f"- Test problems: {summary.get('num_test_problems', 'N/A')}")
        lines.append(f"- Train time: {summary.get('train_time', 'N/A'):.1f}s")
        lines.append(f"- Eval time: {summary.get('eval_time', 'N/A'):.1f}s")
        
        if summary.get('avg_speedup'):
            lines.append(f"\n### Performance\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Avg Speedup | {summary['avg_speedup']:.2f}x |")
            lines.append(f"| Std Speedup | ±{summary.get('std_speedup', 0):.2f} |")
            lines.append(f"| Min Speedup | {summary.get('min_speedup', 'N/A'):.2f}x |")
            lines.append(f"| Max Speedup | {summary.get('max_speedup', 'N/A'):.2f}x |")
        
        # If scaling study, add scaling table
        if 'scaling' in var_data:
            scaling_summary = var_data['scaling'].get('summary', {})
            if scaling_summary:
                lines.append(f"\n### Scaling Results\n")
                lines.append("| Grid Size | DOFs | Learned Iters | Baseline Iters | Speedup |")
                lines.append("|-----------|------|---------------|----------------|---------|")
                
                for gs in sorted([int(k) for k in scaling_summary.keys()]):
                    gs_data = scaling_summary[str(gs)]
                    n = gs_data.get('n', gs*gs)
                    l_iters = gs_data.get('learned_iters_mean')
                    b_iters = gs_data.get('baseline_iters_mean')
                    speedup = gs_data.get('speedup_mean')
                    
                    l_str = f"{l_iters:.1f}" if l_iters else "N/A"
                    b_str = f"{b_iters:.1f}" if b_iters else "N/A"
                    s_str = f"{speedup:.2f}x" if speedup else "N/A"
                    
                    lines.append(f"| {gs}x{gs} | {n:,} | {l_str} | {b_str} | {s_str} |")
    
    markdown = "\n".join(lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(markdown)
        print(f"Saved: {save_path}")
    
    return markdown


def plot_from_results_json(
    results_path: Path,
    output_dir: Optional[Path] = None,
    variation_name: Optional[str] = None,
) -> None:
    """
    Generate plots from a single variation's results.json file.
    
    This is useful for regenerating plots after a run, or for 
    generating plots from previously saved results.
    
    Args:
        results_path: Path to results.json (single variation)
        output_dir: Directory to save plots (default: plots/ in same dir)
        variation_name: Name for plot titles (default: inferred from path)
    """
    results_path = Path(results_path)
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    if output_dir is None:
        output_dir = results_path.parent / "plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Infer variation name if not provided
    if variation_name is None:
        variation_name = results.get('variation_name', results_path.parent.name)
    
    test_problems = results.get('test_problems', [])
    
    print(f"Generating plots for {variation_name} in {output_dir}...")
    
    # Iteration histogram
    if test_problems:
        try:
            plot_iteration_histogram(
                test_problems,
                title=f"{variation_name}: PCG Iteration Distribution",
                save_path=output_dir / "iteration_histogram.png"
            )
            plt.close()
        except Exception as e:
            print(f"  Warning: Could not generate iteration histogram: {e}")
    
    # Residual curves
    if test_problems:
        try:
            plot_residual_curves(
                test_problems,
                title=f"{variation_name}: Residual Convergence",
                save_path=output_dir / "residual_curves.png",
                max_problems=5,
            )
            plt.close()
        except Exception as e:
            print(f"  Warning: Could not generate residual curves: {e}")
    
    # Energy decay
    if test_problems:
        try:
            plot_energy_decay(
                test_problems,
                title=f"{variation_name}: V-cycle Energy Decay",
                save_path=output_dir / "energy_decay.png",
                max_problems=5,
            )
            plt.close()
        except Exception as e:
            print(f"  Warning: Could not generate energy decay plot: {e}")
    
    # Scaling curves (if scaling study)
    scaling_data = results.get('scaling_metrics')
    if scaling_data and scaling_data.get('summary'):
        try:
            plot_scaling_curves(
                scaling_data,
                title=f"{variation_name}: Scaling Study",
                save_path=output_dir / "scaling_curves.png"
            )
            plt.close()
        except Exception as e:
            print(f"  Warning: Could not generate scaling curves: {e}")
    
    print(f"Done! Plots saved to {output_dir}")


def generate_all_plots(
    results_path: Path,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Generate all visualization plots from combined_results.json file.
    
    Args:
        results_path: Path to combined_results.json
        output_dir: Directory to save plots (default: same as results)
    """
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    if output_dir is None:
        output_dir = results_path.parent / "plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating plots in {output_dir}...")
    
    for var_name, var_data in results.get('variations', {}).items():
        var_dir = output_dir / f"variation_{var_name}"
        var_dir.mkdir(exist_ok=True)
        
        test_problems = var_data.get('test_problems', [])
        
        # Iteration histogram
        if test_problems:
            plot_iteration_histogram(
                test_problems,
                title=f"Variation {var_name}: PCG Iterations",
                save_path=var_dir / "iteration_histogram.png"
            )
            plt.close()
        
        # Residual curves
        if test_problems:
            plot_residual_curves(
                test_problems,
                title=f"Variation {var_name}: Residual Convergence",
                save_path=var_dir / "residual_curves.png"
            )
            plt.close()
        
        # Energy decay
        if test_problems:
            plot_energy_decay(
                test_problems,
                title=f"Variation {var_name}: V-cycle Energy Decay",
                save_path=var_dir / "energy_decay.png"
            )
            plt.close()
        
        # Scaling curves (if scaling study)
        if 'scaling' in var_data:
            plot_scaling_curves(
                var_data['scaling'],
                title=f"Variation {var_name}: Scaling Study",
                save_path=var_dir / "scaling_curves.png"
            )
            plt.close()
    
    # Comparison table
    generate_comparison_table(
        results,
        save_path=output_dir / "comparison_table.md"
    )
    
    print(f"Generated plots in {output_dir}")
