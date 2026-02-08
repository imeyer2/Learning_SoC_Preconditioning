"""
Baseline strategies for variance reduction in policy gradients.

References:
- Schulman et al. "High-Dimensional Continuous Control Using GAE" (2015)
- Greensmith et al. "Variance Reduction Techniques for Gradient Estimates" (2004)
"""

import numpy as np
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


class RunningMeanStdBaseline:
    """
    Baseline with running mean AND standard deviation for reward normalization.
    
    This provides both:
    1. Baseline subtraction (reduces mean of advantage)
    2. Reward normalization (reduces variance of advantage)
    
    Reference: Schulman et al. recommend normalizing advantages.
    """
    
    def __init__(self, momentum: float = 0.99, normalize: bool = True, clip: float = 10.0):
        """
        Args:
            momentum: Momentum for exponential moving average
            normalize: Whether to normalize by std
            clip: Clip normalized advantages to [-clip, clip]
        """
        self.momentum = momentum
        self.normalize = normalize
        self.clip = clip
        
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
    
    def update(self, reward: float) -> float:
        """
        Update statistics and return normalized advantage.
        """
        self.count += 1
        
        # Welford's online algorithm for mean and variance
        delta = reward - self.mean
        self.mean = self.mean + delta / self.count
        delta2 = reward - self.mean
        self.var = self.var + (delta * delta2 - self.var) / self.count
        
        # Compute advantage
        advantage = reward - self.mean
        
        # Optionally normalize by std
        if self.normalize and self.var > 1e-8:
            std = np.sqrt(self.var)
            advantage = advantage / std
            advantage = np.clip(advantage, -self.clip, self.clip)
        
        return float(advantage)
    
    def get_baseline(self) -> float:
        """Get current baseline (mean)."""
        return self.mean
    
    def get_std(self) -> float:
        """Get current std estimate."""
        return np.sqrt(max(self.var, 1e-8))


class ExponentialMovingBaseline:
    """
    Exponential moving average baseline with optional normalization.
    
    More responsive than RunningMeanStdBaseline for non-stationary rewards.
    """
    
    def __init__(
        self, 
        momentum: float = 0.95, 
        var_momentum: float = 0.99,
        normalize: bool = True,
        clip: float = 10.0,
        warmup_steps: int = 100,
    ):
        """
        Args:
            momentum: Momentum for mean EMA
            var_momentum: Momentum for variance EMA (typically higher)
            normalize: Whether to normalize by std
            clip: Clip normalized advantages
            warmup_steps: Use simple average during warmup
        """
        self.momentum = momentum
        self.var_momentum = var_momentum
        self.normalize = normalize
        self.clip = clip
        self.warmup_steps = warmup_steps
        
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        
        # Track recent rewards for warmup
        self._warmup_rewards = []
    
    def update(self, reward: float) -> float:
        """Update and return normalized advantage."""
        self.count += 1
        
        if self.count <= self.warmup_steps:
            # During warmup, use simple statistics
            self._warmup_rewards.append(reward)
            self.mean = np.mean(self._warmup_rewards)
            if len(self._warmup_rewards) > 1:
                self.var = np.var(self._warmup_rewards)
        else:
            # EMA update
            delta = reward - self.mean
            self.mean = self.momentum * self.mean + (1 - self.momentum) * reward
            self.var = self.var_momentum * self.var + (1 - self.var_momentum) * (delta ** 2)
        
        # Compute advantage
        advantage = reward - self.mean
        
        if self.normalize and self.var > 1e-8:
            std = np.sqrt(self.var)
            advantage = advantage / std
            advantage = np.clip(advantage, -self.clip, self.clip)
        
        return float(advantage)
    
    def get_baseline(self) -> float:
        return self.mean
    
    def get_std(self) -> float:
        return np.sqrt(max(self.var, 1e-8))
