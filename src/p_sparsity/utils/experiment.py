"""
Experiment tracking and artifact management.
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import torch

from .config import save_config


class ExperimentTracker:
    """
    Tracks experiment artifacts: checkpoints, configs, logs, etc.
    """
    
    def __init__(self, experiment_dir: str):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_dir: Root directory for experiment
        """
        self.experiment_dir = Path(experiment_dir)
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.config_dir = self.experiment_dir / "configs"
        self.plot_dir = self.experiment_dir / "plots"
        self.log_dir = self.experiment_dir / "logs"
        
        # Create directories
        for d in [self.checkpoint_dir, self.config_dir, self.plot_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Tracking state
        self.best_metric = float('-inf')
        self.best_checkpoint = None
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "experiment_dir": str(self.experiment_dir),
        }
    
    def save_config(self, config: Dict[str, Any], name: str = "config.yaml"):
        """Save configuration."""
        save_config(config, self.config_dir / name)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        name: str = "checkpoint.pt",
    ):
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state (optional)
            epoch: Current epoch
            metrics: Metrics dict (optional)
            name: Checkpoint filename
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics or {},
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        path = self.checkpoint_dir / name
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
        return str(path)
    
    def save_best_checkpoint(
        self,
        model: torch.nn.Module,
        metric_value: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        higher_is_better: bool = True,
    ):
        """
        Save checkpoint if metric is best so far.
        
        Args:
            model: Model to save
            metric_value: Current metric value
            optimizer: Optimizer (optional)
            epoch: Current epoch
            metrics: All metrics (optional)
            higher_is_better: Whether higher metric is better
        """
        is_best = (
            (higher_is_better and metric_value > self.best_metric) or
            (not higher_is_better and metric_value < self.best_metric)
        )
        
        if is_best:
            self.best_metric = metric_value
            path = self.save_checkpoint(
                model, optimizer, epoch, metrics, name="best_model.pt"
            )
            self.best_checkpoint = path
            print(f"New best model! Metric: {metric_value:.6f}")
    
    def load_checkpoint(self, name: str = "best_model.pt") -> Dict:
        """Load checkpoint."""
        path = self.checkpoint_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return torch.load(path)
    
    def save_metrics(self, metrics: Dict[str, Any], name: str = "metrics.json"):
        """Save metrics to JSON."""
        path = self.log_dir / name
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def save_plot(self, fig, name: str):
        """
        Save matplotlib figure.
        
        Args:
            fig: Matplotlib figure
            name: Filename (with extension)
        """
        path = self.plot_dir / name
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"Saved plot: {path}")
    
    def save_metadata(self, additional_metadata: Optional[Dict] = None):
        """Save experiment metadata."""
        if additional_metadata:
            self.metadata.update(additional_metadata)
        
        self.metadata["updated_at"] = datetime.now().isoformat()
        
        path = self.experiment_dir / "metadata.json"
        with open(path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_plot_path(self, name: str) -> str:
        """Get path for saving plot."""
        return str(self.plot_dir / name)
    
    def get_log_path(self, name: str) -> str:
        """Get path for saving log file."""
        return str(self.log_dir / name)


def create_experiment(
    name: Optional[str] = None,
    base_dir: str = "outputs",
) -> ExperimentTracker:
    """
    Create a new experiment.
    
    Args:
        name: Experiment name (auto-generated if None)
        base_dir: Base output directory
        
    Returns:
        tracker: ExperimentTracker instance
    """
    if name is None:
        name = f"exp_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    experiment_dir = Path(base_dir) / name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    tracker = ExperimentTracker(str(experiment_dir))
    print(f"Created experiment: {experiment_dir}")
    
    return tracker
