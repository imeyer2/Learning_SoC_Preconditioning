"""
REINFORCE training algorithm for AMG policy learning.
"""

from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ...data import TrainSample
from ...pyamg_interface import (
    sample_topk_without_replacement,
    C_from_selected_edges,
    build_B_for_pyamg,
    build_pyamg_solver,
)
from ..rewards import get_reward_function
from ..baselines import MovingAverageBaseline


class ReinforceTrainer:
    """
    REINFORCE (policy gradient) trainer for AMG edge policy.
    
    Samples edge selections from policy, evaluates reward via PyAMG,
    and updates policy to maximize expected reward.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_data: List[TrainSample],
        config: Any,
        experiment_tracker: Any,
        tb_logger: Any,
    ):
        """
        Initialize REINFORCE trainer.
        
        Args:
            model: AMG edge policy model
            train_data: List of training samples
            config: Training configuration
            experiment_tracker: Experiment tracking object
            tb_logger: TensorBoard logger
        """
        self.model = model
        self.train_data = train_data
        self.config = config
        self.experiment = experiment_tracker
        self.tb_logger = tb_logger
        
        # Setup optimizer
        self.optimizer = self._build_optimizer()
        
        # Setup baseline
        baseline_config = config.baseline
        if baseline_config.type == "moving_average":
            self.baseline = MovingAverageBaseline(
                momentum=baseline_config.momentum
            )
        else:
            self.baseline = MovingAverageBaseline(momentum=0.95)
        
        # Get reward function
        reward_config = config.reward
        self.reward_fn = get_reward_function(reward_config.function)
        
        # Training state
        self.step = 0
        self.best_reward = float('-inf')
        
        # History
        self.history = {
            "train_reward": [],
            "baseline": [],
            "advantage": [],
            "loss": [],
        }
    
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer from config."""
        opt_config = self.config.optimizer
        
        if opt_config.type == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=opt_config.lr,
                weight_decay=opt_config.get("weight_decay", 0.0),
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config.type}")
    
    def train(self) -> Dict[str, List]:
        """
        Run training loop.
        
        Returns:
            history: Training metrics history
        """
        self.model.to(self.config.device)
        self.model.train()
        
        # Temperature schedule
        temperature = self.config.temperature.initial
        temp_anneal = self.config.temperature.get("anneal", True)
        temp_factor = self.config.temperature.get("anneal_factor", 0.97)
        min_temp = self.config.temperature.get("min_temperature", 0.5)
        
        print("\n" + "=" * 80)
        print("Training Started")
        print("=" * 80)
        
        for epoch in range(self.config.epochs):
            # Shuffle training data
            np.random.shuffle(self.train_data)
            
            epoch_rewards = []
            
            # Training loop
            pbar = tqdm(self.train_data, desc=f"Epoch {epoch+1}/{self.config.epochs}")
            for sample in pbar:
                reward, loss = self._train_step(sample, temperature)
                epoch_rewards.append(reward)
                
                # Update progress bar
                pbar.set_postfix({
                    "reward": f"{reward:.4f}",
                    "baseline": f"{self.baseline.get_baseline():.4f}",
                    "temp": f"{temperature:.3f}",
                })
            
            # Epoch summary
            mean_reward = np.mean(epoch_rewards)
            print(f"Epoch {epoch+1:03d} | "
                  f"Mean Reward: {mean_reward:.4f} | "
                  f"Baseline: {self.baseline.get_baseline():.4f} | "
                  f"Temp: {temperature:.3f}")
            
            # Log to TensorBoard
            self.tb_logger.log_scalar("train/epoch_mean_reward", mean_reward, epoch)
            self.tb_logger.log_scalar("train/temperature", temperature, epoch)
            
            # Save checkpoint if best
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.experiment.save_best_checkpoint(
                    self.model,
                    metric_value=mean_reward,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics={"mean_reward": mean_reward},
                    higher_is_better=True,
                )
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.logging.save_interval == 0:
                self.experiment.save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch=epoch,
                    metrics={"mean_reward": mean_reward},
                    name=f"checkpoint_epoch_{epoch+1}.pt",
                )
            
            # Temperature annealing
            if temp_anneal:
                temperature = max(min_temp, temperature * temp_factor)
        
        print("\n" + "=" * 80)
        print("Training Complete!")
        print(f"Best Reward: {self.best_reward:.4f}")
        print("=" * 80)
        
        return self.history
    
    def _train_step(self, sample: TrainSample, temperature: float) -> tuple:
        """
        Single training step on one sample.
        
        Args:
            sample: Training sample
            temperature: Current temperature for sampling
            
        Returns:
            reward: Reward value
            loss: Loss value
        """
        self.step += 1
        
        # Move data to device
        x = sample.x.to(self.config.device)
        ei = sample.edge_index.to(self.config.device)
        ew = sample.edge_weight.to(self.config.device)
        
        # Forward pass: get logits and B candidates
        logits, B_extra = self.model(x, ei, ew)
        
        # Sample edges from policy
        selected_mask_t, logp_sum = sample_topk_without_replacement(
            logits=logits,
            row_groups=sample.row_groups,
            k=self.config.sampling.k_per_row,
            temperature=temperature,
            gumbel=self.config.sampling.get("gumbel", True),
        )
        
        # Build C matrix and PyAMG solver
        selected_mask = selected_mask_t.detach().cpu().numpy()
        C = C_from_selected_edges(sample.A, sample.edge_index, selected_mask)
        B = build_B_for_pyamg(B_extra)
        
        try:
            ml = build_pyamg_solver(
                sample.A,
                C,
                B,
                coarse_solver=self.config.pyamg.get("coarse_solver", "splu"),
            )
            # Compute reward
            R = self.reward_fn(sample.A, ml, self.config.reward)
        except Exception as e:
            # If PyAMG fails, give negative reward
            print(f"Warning: PyAMG failed with {e}, using penalty reward")
            R = -5.0
        
        # Update baseline and compute advantage
        advantage = self.baseline.update(R)
        
        # REINFORCE loss: -advantage * logprob
        loss = -(
            torch.tensor(advantage, device=self.config.device, dtype=torch.float32)
            * logp_sum
        )
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.optimizer.get("grad_clip", 0.0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.optimizer.grad_clip
            )
        
        self.optimizer.step()
        
        # Logging
        if self.step % self.config.logging.log_interval == 0:
            self.tb_logger.log_scalar("train/reward", R, self.step)
            self.tb_logger.log_scalar("train/baseline", self.baseline.get_baseline(), self.step)
            self.tb_logger.log_scalar("train/advantage", advantage, self.step)
            self.tb_logger.log_scalar("train/loss", float(loss.detach().cpu().item()), self.step)
        
        # Store in history
        self.history["train_reward"].append(R)
        self.history["baseline"].append(self.baseline.get_baseline())
        self.history["advantage"].append(advantage)
        self.history["loss"].append(float(loss.detach().cpu().item()))
        
        return R, float(loss.detach().cpu().item())
