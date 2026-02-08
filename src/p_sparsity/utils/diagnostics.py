"""
Training Diagnostics for GNN/RL AMG Learning.

Implements detailed metrics for monitoring training dynamics:
- Policy gradient diagnostics (entropy, advantages, gradients)
- GNN diagnostics (attention entropy, over-smoothing, gradient flow)
- Problem-specific (sparsity, selection patterns)

References:
- Schulman et al. "PPO" - policy entropy, KL divergence
- Schulman et al. "GAE" - advantage estimation diagnostics
- Xu et al. "How Powerful are GNNs" - GNN expressiveness
- Chen et al. "DropEdge" - over-smoothing in deep GNNs
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import matplotlib.pyplot as plt


@dataclass
class EpochDiagnostics:
    """Diagnostics collected for a single epoch."""
    epoch: int
    
    # RL / Policy Gradient Metrics
    policy_entropy: float = 0.0           # H(π) - exploration measure
    policy_entropy_per_node: float = 0.0  # Normalized by nodes
    advantage_mean: float = 0.0           # E[A]
    advantage_std: float = 0.0            # Std[A]
    advantage_min: float = 0.0
    advantage_max: float = 0.0
    value_explained_var: float = 0.0      # 1 - Var[R-V]/Var[R]
    
    # Gradient Metrics
    grad_norm_total: float = 0.0          # ||∇θ||
    grad_norm_policy: float = 0.0         # Policy head gradients
    grad_norm_gnn: float = 0.0            # GNN backbone gradients
    grad_max: float = 0.0                 # Max gradient component
    
    # GNN-specific Metrics
    attention_entropy: float = 0.0        # H(attention weights) for GAT
    node_embedding_mad: float = 0.0       # Over-smoothing indicator
    layer_grad_norms: List[float] = field(default_factory=list)
    
    # Problem-specific Metrics
    sparsity_ratio: float = 0.0           # Edges kept / total edges
    selection_entropy: float = 0.0        # Entropy of edge selection
    reward_variance: float = 0.0          # Var[R]
    
    # Per-sample statistics (for distributions)
    sample_rewards: List[float] = field(default_factory=list)
    sample_entropies: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'epoch': self.epoch,
            'policy_entropy': self.policy_entropy,
            'policy_entropy_per_node': self.policy_entropy_per_node,
            'advantage_mean': self.advantage_mean,
            'advantage_std': self.advantage_std,
            'advantage_min': self.advantage_min,
            'advantage_max': self.advantage_max,
            'value_explained_var': self.value_explained_var,
            'grad_norm_total': self.grad_norm_total,
            'grad_norm_policy': self.grad_norm_policy,
            'grad_norm_gnn': self.grad_norm_gnn,
            'grad_max': self.grad_max,
            'attention_entropy': self.attention_entropy,
            'node_embedding_mad': self.node_embedding_mad,
            'layer_grad_norms': self.layer_grad_norms,
            'sparsity_ratio': self.sparsity_ratio,
            'selection_entropy': self.selection_entropy,
            'reward_variance': self.reward_variance,
        }


class TrainingDiagnostics:
    """
    Collect and analyze training diagnostics for GNN/RL.
    
    Usage:
        diagnostics = TrainingDiagnostics()
        
        for epoch in range(epochs):
            diagnostics.start_epoch(epoch)
            
            for sample in data:
                # ... training step ...
                diagnostics.record_step(
                    logits=logits,
                    actions=actions,
                    reward=reward,
                    baseline=baseline,
                )
            
            diagnostics.end_epoch(model)
        
        diagnostics.save("diagnostics.json")
        diagnostics.plot_all("plots/")
    """
    
    def __init__(self):
        self.epochs: List[EpochDiagnostics] = []
        self.current_epoch: Optional[EpochDiagnostics] = None
        
        # Accumulators for current epoch
        self._rewards: List[float] = []
        self._baselines: List[float] = []
        self._entropies: List[float] = []
        self._sparsities: List[float] = []
        self._advantages: List[float] = []
        self._attention_weights: List[torch.Tensor] = []
        self._embeddings: List[torch.Tensor] = []
    
    def start_epoch(self, epoch: int):
        """Start collecting diagnostics for a new epoch."""
        self.current_epoch = EpochDiagnostics(epoch=epoch)
        self._rewards = []
        self._baselines = []
        self._entropies = []
        self._sparsities = []
        self._advantages = []
        self._attention_weights = []
        self._embeddings = []
    
    def record_step(
        self,
        logits: torch.Tensor,
        actions: torch.Tensor,
        reward: float,
        baseline: float,
        attention_weights: Optional[torch.Tensor] = None,
        node_embeddings: Optional[torch.Tensor] = None,
        num_total_edges: Optional[int] = None,
    ):
        """
        Record metrics from a single training step.
        
        Args:
            logits: Raw policy outputs before softmax (N, E_per_node) or (E,)
            actions: Selected edge indices
            reward: Reward for this sample
            baseline: Baseline value
            attention_weights: Optional GAT attention weights
            node_embeddings: Optional node embeddings for over-smoothing
            num_total_edges: Total edges in graph (for sparsity)
        """
        self._rewards.append(reward)
        self._baselines.append(baseline)
        self._advantages.append(reward - baseline)
        
        # Policy entropy: H(π) = -Σ p(a) log p(a)
        if logits.dim() == 1:
            probs = torch.softmax(logits, dim=0)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        else:
            # Per-node entropy, then average
            probs = torch.softmax(logits, dim=-1)
            per_node_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            entropy = per_node_entropy.mean().item()
        self._entropies.append(entropy)
        
        # Sparsity
        if num_total_edges is not None:
            sparsity = len(actions) / num_total_edges if num_total_edges > 0 else 0
            self._sparsities.append(sparsity)
        
        # Store attention weights and embeddings for later analysis
        if attention_weights is not None:
            self._attention_weights.append(attention_weights.detach().cpu())
        if node_embeddings is not None:
            self._embeddings.append(node_embeddings.detach().cpu())
    
    def end_epoch(
        self,
        model: nn.Module,
        compute_expensive: bool = True,
    ):
        """
        Finalize epoch diagnostics.
        
        Args:
            model: The model (for gradient analysis)
            compute_expensive: Whether to compute expensive metrics
        """
        if self.current_epoch is None:
            return
        
        diag = self.current_epoch
        
        # Advantage statistics
        advantages = np.array(self._advantages)
        diag.advantage_mean = float(np.mean(advantages))
        diag.advantage_std = float(np.std(advantages))
        diag.advantage_min = float(np.min(advantages))
        diag.advantage_max = float(np.max(advantages))
        
        # Policy entropy
        diag.policy_entropy = float(np.mean(self._entropies))
        
        # Value explained variance: 1 - Var[R-V] / Var[R]
        rewards = np.array(self._rewards)
        baselines = np.array(self._baselines)
        reward_var = np.var(rewards)
        residual_var = np.var(rewards - baselines)
        if reward_var > 1e-8:
            diag.value_explained_var = float(1 - residual_var / reward_var)
        else:
            diag.value_explained_var = 0.0
        
        diag.reward_variance = float(reward_var)
        
        # Sparsity
        if self._sparsities:
            diag.sparsity_ratio = float(np.mean(self._sparsities))
        
        # Selection entropy (entropy of the selection probability distribution)
        diag.selection_entropy = float(np.mean(self._entropies))
        
        # Gradient analysis
        self._compute_gradient_metrics(model, diag)
        
        # Attention entropy (for GAT)
        if self._attention_weights and compute_expensive:
            diag.attention_entropy = self._compute_attention_entropy()
        
        # Over-smoothing (MAD of node embeddings)
        if self._embeddings and compute_expensive:
            diag.node_embedding_mad = self._compute_embedding_mad()
        
        # Store sample data for distributions
        diag.sample_rewards = self._rewards.copy()
        diag.sample_entropies = self._entropies.copy()
        
        self.epochs.append(diag)
        self.current_epoch = None
    
    def _compute_gradient_metrics(self, model: nn.Module, diag: EpochDiagnostics):
        """Compute gradient norms from model parameters."""
        total_norm = 0.0
        gnn_norm = 0.0
        policy_norm = 0.0
        max_grad = 0.0
        layer_norms = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_norm += grad_norm ** 2
                max_grad = max(max_grad, param.grad.data.abs().max().item())
                
                name_lower = name.lower()
                
                # Categorize by layer type
                # GNN layers: backbone, gnn, conv, gat, message passing layers
                if any(k in name_lower for k in ['backbone', 'gnn', 'conv', 'gat', 'sage', 'gcn']):
                    gnn_norm += grad_norm ** 2
                # Policy/output layers: edge_mlp, B_head, policy, head, output, mlp
                elif any(k in name_lower for k in ['edge_mlp', 'b_head', 'policy', 'head', 'output', 'mlp', 'edge_encoder']):
                    policy_norm += grad_norm ** 2
                
                # Track per-layer norms
                layer_norms.append((name, grad_norm))
        
        diag.grad_norm_total = float(np.sqrt(total_norm))
        diag.grad_norm_gnn = float(np.sqrt(gnn_norm))
        diag.grad_norm_policy = float(np.sqrt(policy_norm))
        diag.grad_max = float(max_grad)
        
        # Keep top layers by gradient norm
        layer_norms.sort(key=lambda x: x[1], reverse=True)
        diag.layer_grad_norms = [norm for _, norm in layer_norms[:10]]
    
    def _compute_attention_entropy(self) -> float:
        """Compute average entropy of attention weights."""
        if not self._attention_weights:
            return 0.0
        
        entropies = []
        for attn in self._attention_weights:
            # Attention weights should sum to 1 per node
            # H = -Σ α log(α)
            attn_clamp = torch.clamp(attn, min=1e-10)
            entropy = -torch.sum(attn_clamp * torch.log(attn_clamp), dim=-1)
            entropies.append(entropy.mean().item())
        
        return float(np.mean(entropies))
    
    def _compute_embedding_mad(self) -> float:
        """
        Compute Mean Absolute Deviation of node embeddings.
        
        Low MAD indicates over-smoothing (all nodes have similar embeddings).
        Reference: Chen et al. "Measuring and Relieving Over-smoothing in GNNs"
        """
        if not self._embeddings:
            return 0.0
        
        mads = []
        for emb in self._embeddings:
            # MAD = mean(|h_i - mean(h)|)
            mean_emb = emb.mean(dim=0, keepdim=True)
            mad = torch.mean(torch.abs(emb - mean_emb))
            mads.append(mad.item())
        
        return float(np.mean(mads))
    
    def save(self, path: str):
        """Save diagnostics to JSON file."""
        data = {
            'epochs': [e.to_dict() for e in self.epochs]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingDiagnostics':
        """Load diagnostics from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        diag = cls()
        for e_data in data['epochs']:
            epoch_diag = EpochDiagnostics(epoch=e_data['epoch'])
            for key, value in e_data.items():
                if hasattr(epoch_diag, key):
                    setattr(epoch_diag, key, value)
            diag.epochs.append(epoch_diag)
        
        return diag
    
    def plot_all(self, output_dir: str, show: bool = False):
        """Generate all diagnostic plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        epochs = [e.epoch for e in self.epochs]
        
        # 1. Policy/RL Diagnostics
        self._plot_rl_diagnostics(epochs, output_dir)
        
        # 2. Gradient Diagnostics
        self._plot_gradient_diagnostics(epochs, output_dir)
        
        # 3. GNN Diagnostics
        self._plot_gnn_diagnostics(epochs, output_dir)
        
        # 4. Combined Overview
        self._plot_overview(epochs, output_dir)
        
        if show:
            plt.show()
        
        print(f"Diagnostic plots saved to {output_dir}")
    
    def _plot_rl_diagnostics(self, epochs: List[int], output_dir: Path):
        """Plot RL/policy gradient diagnostics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Policy Entropy
        ax = axes[0, 0]
        entropies = [e.policy_entropy for e in self.epochs]
        ax.plot(epochs, entropies, 'b-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Policy Entropy H(π)')
        ax.set_title('Policy Entropy (Exploration)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Advantage Statistics
        ax = axes[0, 1]
        adv_mean = [e.advantage_mean for e in self.epochs]
        adv_std = [e.advantage_std for e in self.epochs]
        ax.plot(epochs, adv_mean, 'g-', linewidth=2, label='Mean')
        ax.fill_between(epochs, 
                       [m - s for m, s in zip(adv_mean, adv_std)],
                       [m + s for m, s in zip(adv_mean, adv_std)],
                       alpha=0.3, color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Advantage')
        ax.set_title('Advantage Statistics (Mean ± Std)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Value Explained Variance
        ax = axes[1, 0]
        vev = [e.value_explained_var for e in self.epochs]
        ax.plot(epochs, vev, 'purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Explained Variance')
        ax.set_title('Baseline Quality (1 = perfect)')
        ax.set_ylim(-0.1, 1.1)
        ax.axhline(y=1, color='g', linestyle='--', alpha=0.5, label='Perfect')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Useless')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Reward Variance
        ax = axes[1, 1]
        reward_var = [e.reward_variance for e in self.epochs]
        ax.plot(epochs, reward_var, 'orange', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Variance')
        ax.set_title('Reward Variance')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(output_dir / 'rl_diagnostics.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_gradient_diagnostics(self, epochs: List[int], output_dir: Path):
        """Plot gradient flow diagnostics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Total Gradient Norm
        ax = axes[0, 0]
        total_norms = [e.grad_norm_total for e in self.epochs]
        ax.semilogy(epochs, total_norms, 'b-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('||∇θ||₂ (log scale)')
        ax.set_title('Total Gradient Norm')
        ax.grid(True, alpha=0.3)
        
        # GNN vs Policy Gradients
        ax = axes[0, 1]
        gnn_norms = [e.grad_norm_gnn for e in self.epochs]
        policy_norms = [e.grad_norm_policy for e in self.epochs]
        ax.semilogy(epochs, gnn_norms, 'g-', linewidth=2, label='GNN')
        ax.semilogy(epochs, policy_norms, 'r-', linewidth=2, label='Policy Head')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Norm (log)')
        ax.set_title('Component Gradient Norms')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Max Gradient
        ax = axes[1, 0]
        max_grads = [e.grad_max for e in self.epochs]
        ax.semilogy(epochs, max_grads, 'm-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Max |∇|')
        ax.set_title('Maximum Gradient Component')
        ax.grid(True, alpha=0.3)
        
        # Gradient Ratio (GNN / Policy)
        ax = axes[1, 1]
        ratios = [g / (p + 1e-10) for g, p in zip(gnn_norms, policy_norms)]
        ax.plot(epochs, ratios, 'c-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('GNN / Policy Ratio')
        ax.set_title('Gradient Balance (GNN vs Policy)')
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(output_dir / 'gradient_diagnostics.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_gnn_diagnostics(self, epochs: List[int], output_dir: Path):
        """Plot GNN-specific diagnostics."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Attention Entropy
        ax = axes[0]
        attn_entropy = [e.attention_entropy for e in self.epochs]
        if any(e > 0 for e in attn_entropy):
            ax.plot(epochs, attn_entropy, 'b-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('H(attention)')
            ax.set_title('Attention Entropy\n(higher = more distributed)')
        else:
            ax.text(0.5, 0.5, 'No attention data', ha='center', va='center', transform=ax.transAxes)
        ax.grid(True, alpha=0.3)
        
        # Node Embedding MAD (Over-smoothing)
        ax = axes[1]
        mad = [e.node_embedding_mad for e in self.epochs]
        if any(m > 0 for m in mad):
            ax.plot(epochs, mad, 'r-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MAD')
            ax.set_title('Node Embedding MAD\n(low = over-smoothing)')
        else:
            ax.text(0.5, 0.5, 'No embedding data', ha='center', va='center', transform=ax.transAxes)
        ax.grid(True, alpha=0.3)
        
        # Sparsity Ratio
        ax = axes[2]
        sparsity = [e.sparsity_ratio for e in self.epochs]
        ax.plot(epochs, sparsity, 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Edges Kept / Total')
        ax.set_title('Sparsity Ratio')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(output_dir / 'gnn_diagnostics.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_overview(self, epochs: List[int], output_dir: Path):
        """Plot combined overview dashboard."""
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Row 1: Main metrics
        ax1 = fig.add_subplot(gs[0, :2])
        rewards = [np.mean(e.sample_rewards) if e.sample_rewards else 0 for e in self.epochs]
        ax1.plot(epochs, rewards, 'b-', linewidth=2, label='Mean Reward')
        ax1.set_ylabel('Reward', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_xlabel('Epoch')
        ax1.set_title('Training Progress', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax1b = ax1.twinx()
        entropy = [e.policy_entropy for e in self.epochs]
        ax1b.plot(epochs, entropy, 'r--', linewidth=2, label='Entropy')
        ax1b.set_ylabel('Entropy', color='red')
        ax1b.tick_params(axis='y', labelcolor='red')
        
        ax2 = fig.add_subplot(gs[0, 2:])
        vev = [e.value_explained_var for e in self.epochs]
        ax2.plot(epochs, vev, 'purple', linewidth=2)
        ax2.fill_between(epochs, 0, vev, alpha=0.3, color='purple')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Explained Var')
        ax2.set_title('Baseline Quality', fontsize=12, fontweight='bold')
        ax2.set_ylim(-0.1, 1.1)
        ax2.grid(True, alpha=0.3)
        
        # Row 2: Gradient health
        ax3 = fig.add_subplot(gs[1, :2])
        gnn_norms = [e.grad_norm_gnn for e in self.epochs]
        policy_norms = [e.grad_norm_policy for e in self.epochs]
        ax3.semilogy(epochs, gnn_norms, 'g-', linewidth=2, label='GNN')
        ax3.semilogy(epochs, policy_norms, 'orange', linewidth=2, label='Policy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Gradient Norm')
        ax3.set_title('Gradient Flow', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(gs[1, 2:])
        adv_mean = [e.advantage_mean for e in self.epochs]
        adv_std = [e.advantage_std for e in self.epochs]
        ax4.plot(epochs, adv_mean, 'b-', linewidth=2)
        ax4.fill_between(epochs,
                        [m - s for m, s in zip(adv_mean, adv_std)],
                        [m + s for m, s in zip(adv_mean, adv_std)],
                        alpha=0.3, color='blue')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Advantage')
        ax4.set_title('Advantage Distribution', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Row 3: Distributions for last epoch
        if self.epochs and self.epochs[-1].sample_rewards:
            last = self.epochs[-1]
            
            ax5 = fig.add_subplot(gs[2, :2])
            ax5.hist(last.sample_rewards, bins=30, color='blue', alpha=0.7, edgecolor='black')
            ax5.axvline(np.mean(last.sample_rewards), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(last.sample_rewards):.2f}')
            ax5.set_xlabel('Reward')
            ax5.set_ylabel('Count')
            ax5.set_title(f'Reward Distribution (Epoch {last.epoch})', fontsize=12, fontweight='bold')
            ax5.legend()
            
            ax6 = fig.add_subplot(gs[2, 2:])
            ax6.hist(last.sample_entropies, bins=30, color='green', alpha=0.7, edgecolor='black')
            ax6.axvline(np.mean(last.sample_entropies), color='red', linestyle='--',
                       label=f'Mean: {np.mean(last.sample_entropies):.2f}')
            ax6.set_xlabel('Entropy')
            ax6.set_ylabel('Count')
            ax6.set_title(f'Policy Entropy Distribution (Epoch {last.epoch})', fontsize=12, fontweight='bold')
            ax6.legend()
        
        fig.suptitle('Training Diagnostics Overview', fontsize=14, fontweight='bold', y=1.02)
        fig.savefig(output_dir / 'overview_dashboard.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def print_epoch_summary(self, epoch_idx: int = -1):
        """Print a summary of diagnostics for an epoch."""
        if not self.epochs:
            print("No epochs recorded")
            return
        
        e = self.epochs[epoch_idx]
        
        print(f"\n{'='*60}")
        print(f"  Epoch {e.epoch} Diagnostics")
        print(f"{'='*60}")
        
        print(f"\n📊 Policy/RL Metrics:")
        print(f"   Policy Entropy:      {e.policy_entropy:.4f}")
        print(f"   Advantage Mean:      {e.advantage_mean:.4f}")
        print(f"   Advantage Std:       {e.advantage_std:.4f}")
        print(f"   Value Explained Var: {e.value_explained_var:.4f}")
        
        print(f"\n📈 Gradient Metrics:")
        print(f"   Total Grad Norm:     {e.grad_norm_total:.4f}")
        print(f"   GNN Grad Norm:       {e.grad_norm_gnn:.4f}")
        print(f"   Policy Grad Norm:    {e.grad_norm_policy:.4f}")
        print(f"   Max Gradient:        {e.grad_max:.6f}")
        
        print(f"\n🔗 GNN Metrics:")
        print(f"   Attention Entropy:   {e.attention_entropy:.4f}")
        print(f"   Node Embedding MAD:  {e.node_embedding_mad:.4f}")
        print(f"   Sparsity Ratio:      {e.sparsity_ratio:.4f}")
        
        print(f"{'='*60}\n")


def compute_policy_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of policy distribution.
    
    H(π) = -Σ π(a|s) log π(a|s)
    
    Higher entropy = more exploration
    Lower entropy = more deterministic
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def compute_explained_variance(returns: np.ndarray, baselines: np.ndarray) -> float:
    """
    Compute explained variance of baseline predictions.
    
    EV = 1 - Var[R - V] / Var[R]
    
    EV = 1: Perfect baseline
    EV = 0: Baseline no better than predicting mean
    EV < 0: Baseline worse than mean
    """
    if np.var(returns) < 1e-8:
        return 0.0
    return 1.0 - np.var(returns - baselines) / np.var(returns)
