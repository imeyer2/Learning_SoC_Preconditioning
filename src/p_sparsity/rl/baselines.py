"""
Baseline strategies for variance reduction in policy gradients.
"""

import torch


class MovingAverageBaseline:
    """Moving average baseline for REINFORCE."""
    
    def __init__(self, momentum: float = 0.95):
        """
        Initialize moving average baseline.
        
        Args:
            momentum: Momentum for exponential moving average
        """
        self.momentum = momentum
        self.baseline = 0.0
    
    def update(self, reward: float) -> float:
        """
        Update baseline with new reward.
        
        Args:
            reward: Current reward
            
        Returns:
            advantage: reward - baseline
        """
        self.baseline = self.momentum * self.baseline + (1.0 - self.momentum) * reward
        advantage = reward - self.baseline
        return advantage
    
    def get_baseline(self) -> float:
        """Get current baseline value."""
        return self.baseline
