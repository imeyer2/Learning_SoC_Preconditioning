"""
TensorBoard logging utilities.
"""

import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """Wrapper for TensorBoard SummaryWriter with convenience methods."""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalars under a main tag."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log a histogram."""
        self.writer.add_histogram(tag, values, step)
    
    def log_text(self, tag: str, text: str, step: int):
        """Log text."""
        self.writer.add_text(tag, text, step)
    
    def log_hparams(self, hparam_dict: dict, metric_dict: dict):
        """Log hyperparameters."""
        self.writer.add_hparams(hparam_dict, metric_dict)
    
    def close(self):
        """Close the writer."""
        self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def setup_tensorboard(experiment_name: str = None, base_dir: str = "outputs") -> tuple:
    """
    Setup TensorBoard logging directory and logger.
    
    Args:
        experiment_name: Name of experiment (auto-generated if None)
        base_dir: Base directory for outputs
        
    Returns:
        logger: TensorBoardLogger instance
        log_dir: Log directory path
    """
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = f"exp_{timestamp}"
    
    log_dir = Path(base_dir) / experiment_name / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = TensorBoardLogger(str(log_dir))
    print(f"TensorBoard logging to: {log_dir}")
    
    return logger, str(log_dir)


def launch_tensorboard(log_dir: str, port: int = 6006):
    """
    Launch TensorBoard server.
    
    Args:
        log_dir: TensorBoard log directory
        port: Port number
        
    Returns:
        process: TensorBoard process
    """
    print(f"\n[System] Launching TensorBoard...")
    tb_process = subprocess.Popen(
        [sys.executable, "-m", "tensorboard.main", "--logdir", log_dir, "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(2)
    print(f"[System] TensorBoard running at http://localhost:{port}")
    return tb_process
