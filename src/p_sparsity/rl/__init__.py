"""
Reinforcement Learning Module.

Implements training algorithms, reward functions, and baseline strategies.
"""

from .algorithms import ReinforceTrainer
from .rewards import (
    vcycle_energy_reduction_reward,
    register_reward,
    get_reward_function,
    compute_rewards_parallel,
)
from .baselines import MovingAverageBaseline

__all__ = [
    "ReinforceTrainer",
    "vcycle_energy_reduction_reward",
    "register_reward",
    "get_reward_function",
    "compute_rewards_parallel",
    "MovingAverageBaseline",
]
