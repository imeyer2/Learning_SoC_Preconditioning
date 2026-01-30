"""Training progress visualization."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from typing import List, Optional, Tuple, Dict


def plot_training_progress(
    epochs: List[int],
    metrics: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 8),
    smoothing_window: int = 10
) -> Figure:
    """
    Plot training metrics over epochs.
    
    Args:
        epochs: List of epoch numbers
        metrics: dict mapping metric names to values (e.g., 'loss', 'reward', 'avg_edges')
        figsize: Figure size
        smoothing_window: Window size for moving average smoothing
        
    Returns:
        matplotlib Figure with subplots for each metric
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=figsize, sharex=True)
    
    if num_metrics == 1:
        axes = [axes]
    
    for ax, (name, values) in zip(axes, metrics.items()):
        # Raw values
        ax.plot(epochs, values, alpha=0.3, color='blue', label='Raw')
        
        # Smoothed values
        if len(values) >= smoothing_window:
            smoothed = np.convolve(values, 
                                  np.ones(smoothing_window)/smoothing_window,
                                  mode='valid')
            smooth_epochs = epochs[smoothing_window-1:]
            ax.plot(smooth_epochs, smoothed, color='red', linewidth=2,
                   label=f'Smoothed (window={smoothing_window})')
        
        ax.set_ylabel(name.replace('_', ' ').title())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Epoch")
    fig.suptitle("Training Progress", fontsize=14, y=0.995)
    plt.tight_layout()
    return fig


def plot_reward_distribution(
    rewards: List[float],
    figsize: Tuple[int, int] = (10, 6),
    num_bins: int = 50
) -> Figure:
    """
    Plot distribution of rewards during training.
    
    Args:
        rewards: List of reward values
        figsize: Figure size
        num_bins: Number of histogram bins
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    axes[0].hist(rewards, bins=num_bins, alpha=0.7, edgecolor='black', density=True)
    axes[0].axvline(np.mean(rewards), color='red', linestyle='--',
                   label=f'Mean: {np.mean(rewards):.3f}')
    axes[0].axvline(np.median(rewards), color='green', linestyle='--',
                   label=f'Median: {np.median(rewards):.3f}')
    axes[0].set_xlabel("Reward")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Reward Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(rewards, vert=True)
    axes[1].set_ylabel("Reward")
    axes[1].set_title("Reward Statistics")
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_loss_components(
    epochs: List[int],
    policy_loss: List[float],
    entropy: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Plot RL loss components.
    
    Args:
        epochs: List of epoch numbers
        policy_loss: Policy gradient loss values
        entropy: Entropy bonus values (optional)
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(epochs, policy_loss, label='Policy Loss', linewidth=2, color='blue')
    
    if entropy is not None:
        ax2 = ax.twinx()
        ax2.plot(epochs, entropy, label='Entropy', linewidth=2, 
                color='orange', linestyle='--')
        ax2.set_ylabel("Entropy")
        ax2.legend(loc='upper right')
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Policy Loss")
    ax.set_title("Training Loss Components")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_temperature_schedule(
    epochs: List[int],
    temperatures: List[float],
    figsize: Tuple[int, int] = (8, 5)
) -> Figure:
    """
    Plot temperature annealing schedule.
    
    Args:
        epochs: List of epoch numbers
        temperatures: Temperature values
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(epochs, temperatures, linewidth=2, color='purple')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Temperature")
    ax.set_title("Gumbel-Softmax Temperature Annealing")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_baseline_tracking(
    epochs: List[int],
    raw_rewards: List[float],
    baseline_values: List[float],
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Plot reward baseline tracking.
    
    Args:
        epochs: List of epoch numbers
        raw_rewards: Raw reward values
        baseline_values: Baseline (moving average) values
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(epochs, raw_rewards, alpha=0.3, label='Raw Reward', color='blue')
    ax.plot(epochs, baseline_values, linewidth=2, label='Baseline', color='red')
    
    # Advantage (difference)
    advantages = np.array(raw_rewards) - np.array(baseline_values)
    ax.fill_between(epochs, baseline_values, raw_rewards,
                    where=(np.array(raw_rewards) >= np.array(baseline_values)),
                    alpha=0.3, color='green', label='Positive Advantage')
    ax.fill_between(epochs, baseline_values, raw_rewards,
                    where=(np.array(raw_rewards) < np.array(baseline_values)),
                    alpha=0.3, color='red', label='Negative Advantage')
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    ax.set_title("Baseline Tracking (Moving Average)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
