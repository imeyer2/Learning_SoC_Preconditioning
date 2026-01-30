"""Performance comparison visualization."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple


def plot_performance_comparison(
    comparison_data: Dict[str, Dict[str, float]],
    metrics: List[str],
    figsize: Tuple[int, int] = (12, 6)
) -> Figure:
    """
    Create bar plots comparing multiple solvers across metrics.
    
    Args:
        comparison_data: Nested dict {solver_name: {metric_name: value}}
        metrics: List of metric names to plot
        figsize: Figure size
        
    Returns:
        matplotlib Figure with subplots
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)
    
    if num_metrics == 1:
        axes = [axes]
    
    solver_names = list(comparison_data.keys())
    
    for ax, metric in zip(axes, metrics):
        values = [comparison_data[solver].get(metric, np.nan) 
                 for solver in solver_names]
        
        bars = ax.bar(solver_names, values, alpha=0.7, edgecolor='black')
        
        # Color by performance (lower is usually better)
        if 'time' in metric.lower() or 'iter' in metric.lower() or 'cond' in metric.lower():
            # Lower is better: green for lowest
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(values)))
            sorted_idx = np.argsort(values)
        else:
            # Higher is better: green for highest
            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(values)))
            sorted_idx = np.argsort(values)[::-1]
        
        for i, bar in enumerate(bars):
            if not np.isnan(values[i]):
                bar.set_color(colors[np.where(sorted_idx == i)[0][0]])
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def create_comparison_table(
    comparison_data: Dict[str, Dict[str, float]],
    output_format: str = 'markdown'
) -> str:
    """
    Create a formatted comparison table.
    
    Args:
        comparison_data: Nested dict {solver_name: {metric_name: value}}
        output_format: 'markdown', 'latex', or 'csv'
        
    Returns:
        Formatted table string
    """
    df = pd.DataFrame(comparison_data).T
    
    if output_format == 'markdown':
        return df.to_markdown()
    elif output_format == 'latex':
        return df.to_latex()
    elif output_format == 'csv':
        return df.to_csv()
    else:
        return str(df)


def plot_speedup_chart(
    baseline_times: Dict[str, float],
    learned_times: Dict[str, float],
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Plot speedup relative to baseline.
    
    Args:
        baseline_times: dict mapping problem names to baseline times
        learned_times: dict mapping problem names to learned solver times
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    problems = list(baseline_times.keys())
    speedups = [baseline_times[p] / learned_times[p] for p in problems]
    
    bars = ax.bar(problems, speedups, alpha=0.7, edgecolor='black')
    
    # Color: green if speedup > 1, red if < 1
    for bar, speedup in zip(bars, speedups):
        if speedup >= 1.0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2,
              label='No speedup (baseline)')
    
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Learned AMG Speedup vs Baseline")
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        label = f'{speedup:.2f}×'
        va = 'bottom' if speedup >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., height,
               label, ha='center', va=va, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_radar_comparison(
    comparison_data: Dict[str, Dict[str, float]],
    metrics: List[str],
    figsize: Tuple[int, int] = (8, 8)
) -> Figure:
    """
    Create radar chart comparing solvers across multiple metrics.
    
    Args:
        comparison_data: Nested dict {solver_name: {metric_name: value}}
        metrics: List of metric names to include
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    from math import pi
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    # Number of variables
    num_vars = len(metrics)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the circle
    
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], metrics)
    
    # Plot data for each solver
    for solver_name, values_dict in comparison_data.items():
        values = [values_dict.get(m, 0) for m in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=solver_name)
        ax.fill(angles, values, alpha=0.15)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Multi-Metric Comparison", y=1.08)
    
    return fig


def plot_convergence_comparison_grid(
    results_dict: Dict[str, Dict[str, List[float]]],
    problem_names: List[str],
    figsize: Tuple[int, int] = (15, 10)
) -> Figure:
    """
    Grid of convergence plots for multiple problems and solvers.
    
    Args:
        results_dict: Nested dict {problem_name: {solver_name: residuals}}
        problem_names: List of problem names to plot
        figsize: Figure size
        
    Returns:
        matplotlib Figure with grid of subplots
    """
    num_problems = len(problem_names)
    ncols = min(3, num_problems)
    nrows = (num_problems + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if num_problems > 1 else [axes]
    
    for idx, problem in enumerate(problem_names):
        ax = axes[idx]
        
        for solver_name, residuals in results_dict[problem].items():
            iterations = range(len(residuals))
            ax.semilogy(iterations, residuals, marker='o', 
                       label=solver_name, linewidth=2)
        
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual")
        ax.set_title(f"Problem: {problem}")
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
    
    # Hide unused subplots
    for idx in range(num_problems, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig
