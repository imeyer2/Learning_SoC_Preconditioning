"""
REINFORCE training algorithm for AMG policy learning.
"""

from typing import List, Dict, Any, Optional
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ...data import TrainSample
from ...pyamg_interface import (
    sample_topk_without_replacement,
    sample_topk_fully_vectorized,
    build_row_csr,
    C_from_selected_edges,
    build_B_for_pyamg,
    build_pyamg_solver,
    get_hierarchy_summary,
    print_hierarchy_info,
)
from ..rewards import get_reward_function
from ..baselines import MovingAverageBaseline, ExponentialMovingBaseline
from ...utils.diagnostics import TrainingDiagnostics, compute_policy_entropy


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
        
        # Setup baseline with normalization for better variance reduction
        baseline_config = config.baseline
        baseline_type = getattr(baseline_config, 'type', 'moving_average')
        
        if baseline_type == "exponential_moving":
            # Better baseline with normalization
            self.baseline = ExponentialMovingBaseline(
                momentum=getattr(baseline_config, 'momentum', 0.95),
                var_momentum=getattr(baseline_config, 'var_momentum', 0.99),
                normalize=getattr(baseline_config, 'normalize', True),
                clip=getattr(baseline_config, 'clip', 10.0),
                warmup_steps=getattr(baseline_config, 'warmup_steps', 100),
            )
        else:
            # Simple moving average (original)
            self.baseline = MovingAverageBaseline(
                momentum=getattr(baseline_config, 'momentum', 0.95)
            )
        
        # Entropy coefficient for exploration bonus
        self.entropy_coef = getattr(config, 'entropy_coef', 0.01)
        
        # Parallel reward computation settings
        self.parallel_workers = getattr(config, 'parallel_workers', 1)
        self.parallel_batch_size = getattr(config, 'parallel_batch_size', 8)
        
        # Get reward function
        reward_config = config.reward
        self.reward_fn = get_reward_function(reward_config.function)
        
        # Training state
        self.step = 0
        self.best_reward = float('-inf')
        
        # Diagnostics (detailed training metrics)
        self.diagnostics = TrainingDiagnostics()
        self.enable_diagnostics = getattr(config, 'enable_diagnostics', True)
        
        # Timing instrumentation
        self.timing_enabled = getattr(config, 'timing_enabled', True)
        self.timing = {
            'data_transfer': 0.0,
            'gnn_forward': 0.0,
            'sampling': 0.0,
            'c_matrix_build': 0.0,
            'amg_setup': 0.0,
            'reward_compute': 0.0,
            'backprop': 0.0,
            'optimizer_step': 0.0,
            'total': 0.0,
        }
        self.timing_counts = 0
        
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
        if self.parallel_workers > 1:
            print(f"Using {self.parallel_workers} parallel workers for reward computation")
            print(f"Batch size for parallel: {self.parallel_batch_size}")
        print("=" * 80)
        
        for epoch in range(self.config.epochs):
            # Shuffle training data
            np.random.shuffle(self.train_data)
            
            epoch_rewards = []
            
            # Start diagnostic collection for this epoch
            if self.enable_diagnostics:
                self.diagnostics.start_epoch(epoch + 1)
            
            # Training loop - use parallel or sequential depending on config
            if self.parallel_workers > 1:
                # Parallel batch training
                batches = [
                    self.train_data[i:i + self.parallel_batch_size]
                    for i in range(0, len(self.train_data), self.parallel_batch_size)
                ]
                pbar = tqdm(batches, desc=f"Epoch {epoch+1}/{self.config.epochs} (parallel)")
                for batch in pbar:
                    rewards = self._train_batch_parallel(batch, temperature)
                    epoch_rewards.extend(rewards)
                    
                    # Update progress bar
                    pbar.set_postfix({
                        "batch_reward": f"{np.mean(rewards):.4f}",
                        "baseline": f"{self.baseline.get_baseline():.4f}",
                        "temp": f"{temperature:.3f}",
                    })
            else:
                # Sequential training (original)
                pbar = tqdm(self.train_data, desc=f"Epoch {epoch+1}/{self.config.epochs}")
                for sample in pbar:
                    reward, loss, step_info = self._train_step(sample, temperature)
                    epoch_rewards.append(reward)
                    
                    # Record diagnostics for this step
                    if self.enable_diagnostics and step_info is not None:
                        self.diagnostics.record_step(
                            logits=step_info['logits'],
                            actions=step_info['selected_edges'],
                            reward=reward,
                            baseline=step_info['baseline'],
                            num_total_edges=step_info['num_total_edges'],
                            attention_weights=step_info.get('attention_weights'),
                            node_embeddings=step_info.get('node_embeddings'),
                        )
                    
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
            
            # Print timing at end of each epoch
            if self.timing_enabled:
                self._print_timing_summary()
                self.reset_timing()
            
            # Log to TensorBoard
            self.tb_logger.log_scalar("train/epoch_mean_reward", mean_reward, epoch)
            self.tb_logger.log_scalar("train/temperature", temperature, epoch)
            
            # End epoch diagnostics and log detailed metrics
            if self.enable_diagnostics:
                self.diagnostics.end_epoch(self.model, compute_expensive=True)
                self._log_diagnostics_to_tensorboard(epoch)
                
                # Print epoch diagnostic summary every 10 epochs
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    self.diagnostics.print_epoch_summary()
            
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
        
        # Save and plot diagnostics
        if self.enable_diagnostics and hasattr(self.experiment, 'experiment_dir'):
            self._save_diagnostics()
        
        return self.history
    
    def _log_diagnostics_to_tensorboard(self, epoch: int):
        """Log detailed diagnostics to TensorBoard."""
        if not self.diagnostics.epochs:
            return
        
        e = self.diagnostics.epochs[-1]
        
        # RL Metrics
        self.tb_logger.log_scalar("diagnostics/policy_entropy", e.policy_entropy, epoch)
        self.tb_logger.log_scalar("diagnostics/advantage_mean", e.advantage_mean, epoch)
        self.tb_logger.log_scalar("diagnostics/advantage_std", e.advantage_std, epoch)
        self.tb_logger.log_scalar("diagnostics/value_explained_var", e.value_explained_var, epoch)
        
        # Gradient Metrics
        self.tb_logger.log_scalar("diagnostics/grad_norm_total", e.grad_norm_total, epoch)
        self.tb_logger.log_scalar("diagnostics/grad_norm_gnn", e.grad_norm_gnn, epoch)
        self.tb_logger.log_scalar("diagnostics/grad_norm_policy", e.grad_norm_policy, epoch)
        self.tb_logger.log_scalar("diagnostics/grad_max", e.grad_max, epoch)
        
        # GNN/Problem Metrics
        self.tb_logger.log_scalar("diagnostics/sparsity_ratio", e.sparsity_ratio, epoch)
        self.tb_logger.log_scalar("diagnostics/reward_variance", e.reward_variance, epoch)
        
        # Gradient health indicator (ratio of GNN to policy gradients)
        if e.grad_norm_policy > 1e-10:
            grad_ratio = e.grad_norm_gnn / e.grad_norm_policy
            self.tb_logger.log_scalar("diagnostics/grad_ratio_gnn_policy", grad_ratio, epoch)
    
    def _save_diagnostics(self):
        """Save diagnostics JSON and generate plots."""
        from pathlib import Path
        
        exp_dir = Path(self.experiment.experiment_dir)
        diagnostics_dir = exp_dir / "diagnostics"
        diagnostics_dir.mkdir(exist_ok=True)
        
        # Save JSON
        self.diagnostics.save(str(diagnostics_dir / "training_diagnostics.json"))
        
        # Generate plots
        try:
            self.diagnostics.plot_all(str(diagnostics_dir / "plots"))
            print(f"\n📊 Diagnostic plots saved to: {diagnostics_dir / 'plots'}")
        except Exception as e:
            print(f"Warning: Could not generate diagnostic plots: {e}")
    
    def _print_timing_summary(self):
        """Print timing breakdown for profiling."""
        if self.timing_counts == 0:
            return
            
        total = self.timing['total']
        if total <= 0:
            return
        
        print(f"\n⏱️  TIMING BREAKDOWN (steps={self.timing_counts}, total={total:.2f}s):")
        print("  " + "-" * 55)
        
        # Sort by time descending
        phases = [
            ('data_transfer', 'Data Transfer (CPU→GPU)'),
            ('gnn_forward', 'GNN Forward Pass'),
            ('sampling', 'Edge Sampling (Gumbel)'),
            ('c_matrix_build', 'C Matrix Build'),
            ('amg_setup', 'AMG Hierarchy Setup'),
            ('reward_compute', 'Reward (V-cycle/PCG)'),
            ('backprop', 'Backpropagation'),
            ('optimizer_step', 'Optimizer Step'),
        ]
        
        for key, name in sorted(phases, key=lambda x: self.timing[x[0]], reverse=True):
            t = self.timing[key]
            pct = 100.0 * t / total
            avg_ms = 1000.0 * t / self.timing_counts
            bar_len = int(pct / 2)
            bar = "█" * bar_len + "░" * (50 - bar_len)
            print(f"  {name:28s} {t:7.2f}s ({pct:5.1f}%) | {avg_ms:6.1f}ms/step | {bar[:20]}")
        
        print("  " + "-" * 55)
        print(f"  {'TOTAL':28s} {total:7.2f}s         | {1000.0*total/self.timing_counts:6.1f}ms/step")
        print()
    
    def reset_timing(self):
        """Reset timing counters."""
        for key in self.timing:
            self.timing[key] = 0.0
        self.timing_counts = 0
    
    def _train_step(self, sample: TrainSample, temperature: float) -> tuple:
        """
        Single training step on one sample.
        
        Args:
            sample: Training sample
            temperature: Current temperature for sampling
            
        Returns:
            reward: Reward value
            loss: Loss value
            step_info: Dict with diagnostic info (logits, selected edges, etc.)
        """
        step_start = time.perf_counter()
        self.step += 1
        
        # ========== DATA TRANSFER ==========
        t0 = time.perf_counter()
        x = sample.x.to(self.config.device)
        ei = sample.edge_index.to(self.config.device)
        ew = sample.edge_weight.to(self.config.device)
        # Build CSR format for vectorized sampling (much faster than row_groups list)
        num_nodes = sample.A.shape[0]
        row_ptr, perm = build_row_csr(ei, num_nodes)
        if self.config.device == 'cuda':
            torch.cuda.synchronize()
        self.timing['data_transfer'] += time.perf_counter() - t0
        
        # ========== GNN FORWARD ==========
        t0 = time.perf_counter()
        # Optionally collect internals for diagnostics (every N steps to reduce overhead)
        collect_internals = self.enable_diagnostics and (self.step % 10 == 0)
        
        # Check if model uses new dict-based forward
        output = self.model(x, ei, ew, return_internals=collect_internals)
        
        # Handle both dict-based and legacy tuple-based returns
        if isinstance(output, dict):
            logits = output['edge_logits']
            B_extra = output['B_extra']
            k_per_node = output.get('k_per_node')  # None if learn_k=False
            k_continuous = output.get('k_continuous')  # For loss if needed
            internals = output.get('internals', {})
        else:
            # Legacy tuple return
            if collect_internals and len(output) == 3:
                logits, B_extra, internals = output
            else:
                logits, B_extra = output[:2]
                internals = {}
            k_per_node = None
            k_continuous = None
        
        # Store logits for diagnostics (before sampling modifies them)
        logits_for_diag = logits.detach().clone()
        if self.config.device == 'cuda':
            torch.cuda.synchronize()
        self.timing['gnn_forward'] += time.perf_counter() - t0
        
        # ========== SAMPLING ==========
        t0 = time.perf_counter()
        # Determine k values to use for sampling
        learnable_k = self.config.sampling.get("learnable_k", False)
        if learnable_k and k_per_node is not None:
            k_to_use = k_per_node
        else:
            k_to_use = self.config.sampling.k_per_row
        
        # Sample edges using vectorized Gumbel-top-k (much faster than sequential)
        selected_mask_t, logp_sum = sample_topk_fully_vectorized(
            logits=logits,
            row_ptr=row_ptr,
            perm=perm,
            k=k_to_use,
            temperature=temperature,
            gumbel=self.config.sampling.get("gumbel", True),
        )
        if self.config.device == 'cuda':
            torch.cuda.synchronize()
        self.timing['sampling'] += time.perf_counter() - t0
        
        # ========== C MATRIX BUILD ==========
        t0 = time.perf_counter()
        selected_mask = selected_mask_t.detach().cpu().numpy()
        C = C_from_selected_edges(sample.A, sample.edge_index, selected_mask)
        B = build_B_for_pyamg(B_extra)
        self.timing['c_matrix_build'] += time.perf_counter() - t0
        
        # Track selected edges for diagnostics
        selected_edges = np.where(selected_mask)[0]
        
        # ========== AMG SETUP ==========
        t0 = time.perf_counter()
        try:
            ml = build_pyamg_solver(
                sample.A,
                C,
                B,
                coarse_solver=self.config.pyamg.get("coarse_solver", "splu"),
                max_coarse=self.config.pyamg.get("max_coarse", 50),
            )
            self.timing['amg_setup'] += time.perf_counter() - t0
            
            # Log hierarchy info periodically (every 50 steps or first step)
            if self.step == 1 or self.step % 50 == 0:
                hierarchy_summary = get_hierarchy_summary(ml)
                print(f"  [Step {self.step}] Hierarchy: {hierarchy_summary}")
                # Log coarsest DOFs for debugging
                coarsest_dofs = ml.levels[-1].A.shape[0]
                if coarsest_dofs > 500:
                    print(f"  ⚠️  Coarsest level has {coarsest_dofs} DOFs - expensive direct solve!")
                    print_hierarchy_info(ml, prefix="    ")
            
            # ========== REWARD COMPUTE (V-cycle / PCG) ==========
            t0 = time.perf_counter()
            R = self.reward_fn(sample.A, ml, self.config.reward)
            self.timing['reward_compute'] += time.perf_counter() - t0
        except Exception as e:
            # If PyAMG fails, give negative reward
            print(f"Warning: PyAMG failed with {e}, using penalty reward")
            R = -5.0
            self.timing['amg_setup'] += time.perf_counter() - t0
        
        # Update baseline and compute advantage
        advantage = self.baseline.update(R)
        
        # ========== BACKPROPAGATION ==========
        t0 = time.perf_counter()
        # Compute policy entropy for exploration bonus
        # H(π) = -Σ p(a) log p(a)
        with torch.no_grad():
            probs = torch.softmax(logits_for_diag / temperature, dim=-1)
            log_probs = torch.log_softmax(logits_for_diag / temperature, dim=-1)
            entropy = -torch.sum(probs * log_probs, dim=-1).mean()
        
        # REINFORCE loss: -advantage * logprob - entropy_coef * entropy
        # Entropy bonus encourages exploration (negative because we minimize loss)
        policy_loss = -(
            torch.tensor(advantage, device=self.config.device, dtype=torch.float32)
            * logp_sum
        )
        entropy_bonus = -self.entropy_coef * entropy
        loss = policy_loss + entropy_bonus
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.optimizer.get("grad_clip", 0.0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.optimizer.grad_clip
            )
        if self.config.device == 'cuda':
            torch.cuda.synchronize()
        self.timing['backprop'] += time.perf_counter() - t0
        
        # ========== OPTIMIZER STEP ==========
        t0 = time.perf_counter()
        self.optimizer.step()
        if self.config.device == 'cuda':
            torch.cuda.synchronize()
        self.timing['optimizer_step'] += time.perf_counter() - t0
        
        # Track total time
        self.timing['total'] += time.perf_counter() - step_start
        self.timing_counts += 1
        
        # Logging
        if self.step % self.config.logging.log_interval == 0:
            self.tb_logger.log_scalar("train/reward", R, self.step)
            self.tb_logger.log_scalar("train/baseline", self.baseline.get_baseline(), self.step)
            self.tb_logger.log_scalar("train/advantage", advantage, self.step)
            self.tb_logger.log_scalar("train/loss", float(loss.detach().cpu().item()), self.step)
            self.tb_logger.log_scalar("train/entropy", float(entropy.item()), self.step)
            
            # Log baseline std if available
            if hasattr(self.baseline, 'get_std'):
                self.tb_logger.log_scalar("train/baseline_std", self.baseline.get_std(), self.step)
            
            # Log k statistics if learnable_k is enabled
            if learnable_k and k_per_node is not None:
                k_mean = float(k_per_node.float().mean().item())
                k_std = float(k_per_node.float().std().item())
                k_min_val = float(k_per_node.min().item())
                k_max_val = float(k_per_node.max().item())
                self.tb_logger.log_scalar("train/k_mean", k_mean, self.step)
                self.tb_logger.log_scalar("train/k_std", k_std, self.step)
                self.tb_logger.log_scalar("train/k_min", k_min_val, self.step)
                self.tb_logger.log_scalar("train/k_max", k_max_val, self.step)
        
        # Store in history
        self.history["train_reward"].append(R)
        self.history["baseline"].append(self.baseline.get_baseline())
        self.history["advantage"].append(advantage)
        self.history["loss"].append(float(loss.detach().cpu().item()))
        
        # Prepare diagnostic info
        step_info = {
            'logits': logits_for_diag,
            'selected_edges': selected_edges,
            'baseline': self.baseline.get_baseline(),
            'num_total_edges': sample.edge_index.shape[1] if sample.edge_index.dim() > 1 else len(sample.edge_index),
            'attention_weights': internals.get('attention_weights'),
            'node_embeddings': internals.get('node_embeddings'),
        }
        
        return R, float(loss.detach().cpu().item()), step_info

    def _train_batch_parallel(self, batch: List[TrainSample], temperature: float) -> List[float]:
        """
        Train on a batch of samples with parallel reward computation.
        
        This is more efficient than _train_step when using multiple CPU cores
        because PyAMG reward computation is the bottleneck.
        
        Args:
            batch: List of training samples
            temperature: Current temperature for sampling
            
        Returns:
            List of rewards for each sample
        """
        from ..rewards import compute_rewards_parallel
        
        # Phase 1: Forward passes and sampling (GPU)
        batch_data = []
        for sample in batch:
            self.step += 1
            
            # Move data to device
            x = sample.x.to(self.config.device)
            ei = sample.edge_index.to(self.config.device)
            ew = sample.edge_weight.to(self.config.device)
            # Build CSR format for vectorized sampling
            num_nodes = sample.A.shape[0]
            row_ptr, perm = build_row_csr(ei, num_nodes)
            
            # Forward pass
            output = self.model(x, ei, ew, return_internals=False)
            
            if isinstance(output, dict):
                logits = output['edge_logits']
                B_extra = output['B_extra']
                k_per_node = output.get('k_per_node')
            else:
                logits, B_extra = output[:2]
                k_per_node = None
            
            # Sample edges using vectorized Gumbel-top-k
            learnable_k = self.config.sampling.get("learnable_k", False)
            if learnable_k and k_per_node is not None:
                k_to_use = k_per_node
            else:
                k_to_use = self.config.sampling.k_per_row
            
            selected_mask_t, logp_sum = sample_topk_fully_vectorized(
                logits=logits,
                row_ptr=row_ptr,
                perm=perm,
                k=k_to_use,
                temperature=temperature,
                gumbel=self.config.sampling.get("gumbel", True),
            )
            
            # Build C matrix (CPU)
            selected_mask = selected_mask_t.detach().cpu().numpy()
            C = C_from_selected_edges(sample.A, sample.edge_index, selected_mask)
            B = build_B_for_pyamg(B_extra)
            
            # Store for later
            batch_data.append({
                'sample': sample,
                'logits': logits,
                'logp_sum': logp_sum,
                'A': sample.A,
                'C': C,
                'B': B,
                'temperature': temperature,
            })
        
        # Phase 2: Parallel reward computation (CPU multiprocessing)
        tasks = [(d['A'], d['C'], d['B']) for d in batch_data]
        
        reward_config = {
            k: dict(v) if hasattr(v, 'items') else v 
            for k, v in self.config.reward.items()
        } if hasattr(self.config.reward, 'items') else dict(self.config.reward)
        
        pyamg_config = {
            k: dict(v) if hasattr(v, 'items') else v 
            for k, v in self.config.pyamg.items()
        } if hasattr(self.config.pyamg, 'items') else dict(self.config.pyamg)
        
        # Log hierarchy info periodically
        verbose = (self.step <= self.parallel_batch_size) or (self.step % 100 < self.parallel_batch_size)
        
        rewards, hierarchy_infos = compute_rewards_parallel(
            tasks,
            reward_name=self.config.reward.function,
            reward_config=reward_config,
            pyamg_config=pyamg_config,
            n_workers=self.parallel_workers,
            verbose=verbose,
        )
        
        # Phase 3: Backward passes (GPU)
        total_loss = 0.0
        for data, R in zip(batch_data, rewards):
            # Update baseline and compute advantage
            advantage = self.baseline.update(R)
            
            # Compute entropy
            with torch.no_grad():
                probs = torch.softmax(data['logits'] / temperature, dim=-1)
                log_probs = torch.log_softmax(data['logits'] / temperature, dim=-1)
                entropy = -torch.sum(probs * log_probs, dim=-1).mean()
            
            # REINFORCE loss
            policy_loss = -(
                torch.tensor(advantage, device=self.config.device, dtype=torch.float32)
                * data['logp_sum']
            )
            entropy_bonus = -self.entropy_coef * entropy
            loss = policy_loss + entropy_bonus
            total_loss += loss
            
            # Store history
            self.history["train_reward"].append(R)
            self.history["baseline"].append(self.baseline.get_baseline())
            self.history["advantage"].append(advantage)
            self.history["loss"].append(float(loss.detach().cpu().item()))
        
        # Single backward pass for accumulated loss
        self.optimizer.zero_grad()
        total_loss.backward()
        
        if self.config.optimizer.get("grad_clip", 0.0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.optimizer.grad_clip
            )
        
        self.optimizer.step()
        
        # Log average
        if self.step % self.config.logging.log_interval == 0:
            avg_reward = np.mean(rewards)
            self.tb_logger.log_scalar("train/reward", avg_reward, self.step)
            self.tb_logger.log_scalar("train/baseline", self.baseline.get_baseline(), self.step)
        
        return rewards
