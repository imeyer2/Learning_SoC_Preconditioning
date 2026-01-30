"""
amg_blackbox_train_C.py

Black-box (policy-gradient) training of a GNN that outputs a strength-of-connection pattern C
for PyAMG Smoothed Aggregation, with the reward defined by *measured two-grid / V-cycle error
reduction* on a batch of smooth error vectors.

Key properties:
- PyAMG is used in-the-loop (non-differentiable).
- The GNN defines a stochastic policy over edges of A.
- We sample a top-k set of strong edges per row (degree-controlled).
- We construct a symmetric binary C (CSR) and pass it to PyAMG via strength=('predefined', {'C': C}).
- Reward = average A-energy reduction after ONE V-cycle applied to Ax=0 starting from smooth errors.
- Optional: learn near-nullspace candidates B (in addition to constant vector) from node embeddings.

Dependencies:
  pip install torch torch-geometric pyamg scipy numpy matplotlib tqdm tensorboard
  (optional for faster per-row grouping: torch-scatter is commonly installed with torch-geometric)

Notes:
- This script targets iteration reduction (convergence factor), not setup time.
- For tractable training, use modest grid sizes (e.g., 24–48). After training, test on larger grids.
"""

from __future__ import annotations

import os
import math
import time
import sys
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import from_scipy_sparse_matrix

import pyamg
from pyamg.strength import symmetric_strength_of_connection
from tqdm import tqdm


# ============================================================
# 0) TensorBoard
# ============================================================
def setup_tensorboard():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"runs/amg_blackbox_{timestamp}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logging to: {log_dir}")
    return writer, log_dir

def launch_tensorboard(log_dir):
    print(f"\n[System] Launching TensorBoard...")
    tb_process = subprocess.Popen(
        [sys.executable, "-m", "tensorboard.main", "--logdir", log_dir, "--port", "6006"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(2)
    print(f"[System] TensorBoard running at http://localhost:6006")
    return tb_process


# ============================================================
# 1) Problem generator + smooth errors
# ============================================================
def get_anisotropic_problem(n: int, epsilon: float = 0.001, theta: float = 0.0) -> sp.csr_matrix:
    stencil = pyamg.gallery.diffusion_stencil_2d(type="FD", epsilon=epsilon, theta=theta)
    return pyamg.gallery.stencil_grid(stencil, (n, n), format="csr")

def build_node_coords(grid_n: int) -> np.ndarray:
    N = grid_n * grid_n
    coords = np.zeros((N, 2), dtype=np.float64)
    coords[:, 0] = np.tile(np.linspace(0, 1, grid_n), grid_n)
    coords[:, 1] = np.repeat(np.linspace(0, 1, grid_n), grid_n)
    return coords

def relaxed_smooth_vectors(A: sp.csr_matrix, num_vecs: int, iters: int, omega: float = 2.0/3.0,
                          seed: Optional[int] = None) -> np.ndarray:
    """
    Generate smooth error samples by relaxing Ax = 0 from random initial errors.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((A.shape[0], num_vecs))
    else:
        X = np.random.randn(A.shape[0], num_vecs)

    B = np.zeros((A.shape[0], num_vecs))
    for k in range(num_vecs):
        x = np.ascontiguousarray(X[:, k])
        b = np.ascontiguousarray(B[:, k])
        pyamg.relaxation.relaxation.jacobi(A, x, b, iterations=iters, omega=omega)
        X[:, k] = x

    X = X / (np.abs(X).max() + 1e-6)
    return X

def node_features_for_policy(A: sp.csr_matrix, coords: np.ndarray,
                             num_vecs_in: int = 4, iters_in: int = 5) -> torch.Tensor:
    """
    Policy input features: a few relaxed vectors + geometric coordinates.
    """
    X_in = relaxed_smooth_vectors(A, num_vecs=num_vecs_in, iters=iters_in)
    feats = np.hstack([X_in, coords])
    return torch.tensor(feats, dtype=torch.float32)


# ============================================================
# 2) Dataset
# ============================================================
@dataclass
class TrainSample:
    A: sp.csr_matrix
    grid_n: int
    coords: np.ndarray
    edge_index: torch.Tensor       # (2,E) CPU
    edge_weight: torch.Tensor      # (E,) CPU
    x: torch.Tensor               # (N,F) CPU
    row_groups: List[torch.Tensor]  # list of edge-id tensors per row (CPU)


def make_train_set(num_samples: int = 30, grid_n: int = 32, seed: int = 0) -> List[TrainSample]:
    rng = np.random.default_rng(seed)
    coords = build_node_coords(grid_n)

    samples: List[TrainSample] = []
    for i in range(num_samples):
        mode = i % 3
        if mode == 0:
            eps, th = 1.0, 0.0
        elif mode == 1:
            eps, th = 0.001, 0.0
        else:
            eps, th = 0.001, float(rng.uniform(0.0, math.pi/2))

        A = get_anisotropic_problem(grid_n, epsilon=eps, theta=th)

        ei, ew = from_scipy_sparse_matrix(A)  # ei is CPU by default
        ew = ew.float()
        max_w = ew.abs().max()
        if max_w > 0:
            ew = ew / max_w

        x = node_features_for_policy(A, coords, num_vecs_in=4, iters_in=5)

        row_groups = build_row_groups(ei, num_nodes=A.shape[0])

        samples.append(TrainSample(
            A=A, grid_n=grid_n, coords=coords,
            edge_index=ei, edge_weight=ew, x=x,
            row_groups=row_groups
        ))
    return samples



# ============================================================
# 3) Policy network: edge logits + optional learned B
# ============================================================
class AMGEdgePolicy(nn.Module):
    """
    Outputs edge logits for selecting strong connections.

    Upgrades included:
    - Direction-aware edge features computed from coordinates
    - Uses logits (no sigmoid), because we do stochastic selection via categorical
    - Optional learned B candidates from node embeddings (for PyAMG near-nullspace)
    """
    def __init__(self, in_channels: int = 6, hidden: int = 64, learn_B: bool = True, B_extra: int = 2):
        super().__init__()
        self.learn_B = bool(learn_B)
        self.B_extra = int(B_extra)

        self.conv1 = GATConv(in_channels, hidden, heads=2, concat=True)
        self.conv2 = GATConv(hidden * 2, hidden, heads=1, concat=False)

        # edge features: node_i, node_j, |a_ij|, sim(relaxed), direction(6)
        edge_in = 2 * hidden + 1 + 1 + 6
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # logits
        )

        if self.learn_B:
            self.B_head = nn.Sequential(
                nn.Linear(hidden, 64),
                nn.ReLU(),
                nn.Linear(64, self.B_extra)
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
          edge_logits: (E,)
          B_extra: (N, B_extra) if learn_B else None
        """
        h = F.elu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)

        row, col = edge_index

        # similarity from relaxed input channels (first 4 assumed relaxed)
        phys_i = x[row, :4]
        phys_j = x[col, :4]
        sim = F.cosine_similarity(phys_i, phys_j, dim=1).unsqueeze(1)

        # direction features from coords at x[:,4:6]
        ci = x[row, 4:6]
        cj = x[col, 4:6]
        dxy = ci - cj
        dx = dxy[:, [0]]
        dy = dxy[:, [1]]
        adx = dx.abs()
        ady = dy.abs()
        dn = torch.sqrt(dx * dx + dy * dy + 1e-12)
        udx = dx / dn
        udy = dy / dn
        dir_feat = torch.cat([dx, dy, adx, ady, udx, udy], dim=1)

        node_i = h[row]
        node_j = h[col]
        w = edge_weight.unsqueeze(1).abs()

        edge_feat = torch.cat([node_i, node_j, w, sim, dir_feat], dim=1)
        logits = self.edge_mlp(edge_feat).squeeze(-1)

        # mask diagonals heavily (if present)
        diag = (row == col)
        logits = logits.masked_fill(diag, -1e9)

        B_extra = None
        if self.learn_B:
            B_extra = self.B_head(h)  # (N, B_extra)

        return logits, B_extra


# ============================================================
# 4) Sampling: per-row top-k WITHOUT replacement + logprob
# ============================================================
def build_row_groups(edge_index: torch.Tensor, num_nodes: int) -> List[torch.Tensor]:
    row = edge_index[0].cpu().numpy()
    buckets: List[List[int]] = [[] for _ in range(num_nodes)]
    for e, r in enumerate(row):
        buckets[int(r)].append(e)
    return [torch.tensor(b, dtype=torch.long) for b in buckets]


def sample_topk_without_replacement(
    logits: torch.Tensor,
    row_groups: List[torch.Tensor],
    k: int,
    temperature: float = 1.0,
    gumbel: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each node i:
      sample up to k edges from its outgoing edges, sequentially without replacement.
    Returns:
      selected_mask: (E,) bool
      logprob_sum: scalar tensor
    """
    device = logits.device
    E = logits.numel()
    selected = torch.zeros(E, dtype=torch.bool, device=device)
    logp_total = torch.zeros((), dtype=torch.float32, device=device)

    tau = max(float(temperature), 1e-6)

    for edges_i in row_groups:
        if edges_i.numel() and int(edges_i.max()) >= E:
            raise RuntimeError(f"row_groups contain edge id {int(edges_i.max())} but logits has E={E}")
        if edges_i.numel() == 0:
            continue
        edges_i = edges_i.to(device)

        # number to sample in this row (handle boundary nodes)
        kk = min(k, edges_i.numel())
        if kk <= 0:
            continue

        # copy local logits
        local_logits = logits[edges_i] / tau

        # sequential sampling without replacement
        remaining = edges_i.clone()
        rem_logits = local_logits.clone()

        for _t in range(kk):
            if gumbel:
                u = torch.rand_like(rem_logits)
                g = -torch.log(-torch.log(u + 1e-12) + 1e-12)
                sample_logits = rem_logits + g
            else:
                sample_logits = rem_logits

            probs = F.softmax(rem_logits, dim=0)
            j = torch.argmax(sample_logits)  # sampled index (straight-through top-1)
            chosen_edge = remaining[j]

            selected[chosen_edge] = True
            logp_total = logp_total + torch.log(probs[j] + 1e-12)

            # remove chosen
            keep = torch.ones(remaining.numel(), dtype=torch.bool, device=device)
            keep[j] = False
            remaining = remaining[keep]
            rem_logits = rem_logits[keep]
            if remaining.numel() == 0:
                break

    return selected, logp_total


# ============================================================
# 5) Build C from sampled edges + build PyAMG solver
# ============================================================
def C_from_selected_edges(A: sp.csr_matrix, edge_index_cpu: torch.Tensor, selected_mask: np.ndarray) -> sp.csr_matrix:
    """
    Build symmetric binary strength matrix C with diagonal ones.
    IMPORTANT: uses edge_index ordering (same ordering as logits/selected_mask).
    """
    r = edge_index_cpu[0].cpu().numpy()
    c = edge_index_cpu[1].cpu().numpy()

    sel = selected_mask.astype(bool)
    keep = sel & (r != c)

    rr = r[keep]
    cc = c[keep]
    vals = np.ones(rr.shape[0], dtype=np.float64)

    n = A.shape[0]
    C = sp.csr_matrix((vals, (rr, cc)), shape=(n, n))
    C = C.maximum(C.T)
    C = C + sp.eye(n, format="csr")
    return C


def build_B_for_pyamg(B_extra: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    """
    Build near-nullspace candidates for PyAMG.
    Always include constant vector; optionally add learned columns.
    """
    if B_extra is None:
        return None

    Bx = B_extra.detach().cpu().numpy().astype(np.float64)
    n, p = Bx.shape
    # normalize columns for numerical stability
    for j in range(p):
        col = Bx[:, j]
        col = col - col.mean()
        denom = np.linalg.norm(col) + 1e-12
        Bx[:, j] = col / denom

    ones = np.ones((n, 1), dtype=np.float64)
    B = np.hstack([ones, Bx])
    return B

def build_pyamg_solver(A: sp.csr_matrix, C: sp.csr_matrix, B: Optional[np.ndarray],
                      coarse_solver: str = "splu") -> pyamg.multilevel.MultilevelSolver:
    """
    Build SA solver using a predefined strength matrix C.
    """
    strength = ("predefined", {"C": C})
    if B is None:
        ml = pyamg.smoothed_aggregation_solver(A, strength=strength, coarse_solver=coarse_solver)
    else:
        ml = pyamg.smoothed_aggregation_solver(A, strength=strength, B=B, coarse_solver=coarse_solver)
    return ml


# ============================================================
# 6) Reward: one V-cycle error reduction in A-energy norm
# ============================================================
def energy_norm_sq(A: sp.csr_matrix, e: np.ndarray) -> float:
    Ae = A @ e
    return float(e @ Ae)

def one_vcycle_error_reduce_ratio(ml: pyamg.multilevel.MultilevelSolver, A: sp.csr_matrix,
                                  e0: np.ndarray, cycle: str = "V") -> float:
    """
    Apply one multigrid cycle to Ax=0 starting from x0=e0 and return ratio ||e1||_A^2 / ||e0||_A^2.
    """
    b0 = np.zeros_like(e0)
    try:
        # One cycle: maxiter=1; accel=None prevents Krylov acceleration inside solve.
        e1 = ml.solve(b0, x0=e0, tol=0.0, maxiter=1, accel=None, cycle=cycle)
    except TypeError:
        # Some PyAMG versions use different signature; fall back to tol tiny.
        e1 = ml.solve(b0, x0=e0, tol=1e-30, maxiter=1, accel=None, cycle=cycle)

    num = energy_norm_sq(A, e1)
    den = energy_norm_sq(A, e0) + 1e-30
    return num / den

def compute_reward(A: sp.csr_matrix, ml: pyamg.multilevel.MultilevelSolver,
                   num_test_vecs: int = 6, relax_iters: int = 25, omega: float = 2.0/3.0,
                   complexity_target: float = 1.35, complexity_penalty: float = 1.0) -> float:
    """
    Reward = -mean(log(ratio)) - penalty(max(0, op_complexity - target)).
    Larger is better.

    ratio = ||e_after||_A^2 / ||e_before||_A^2 after one V-cycle.
    """
    # smooth error batch (offline per reward eval)
    E = relaxed_smooth_vectors(A, num_vecs=num_test_vecs, iters=relax_iters, omega=omega)

    ratios = []
    for k in range(num_test_vecs):
        e0 = E[:, k].copy()
        ratio = one_vcycle_error_reduce_ratio(ml, A, e0, cycle="V")
        ratios.append(max(ratio, 1e-12))

    mean_log = float(np.mean(np.log(np.array(ratios))))
    base_reward = -mean_log  # maximize reduction => minimize log ratio

    # operator complexity penalty (keep things fair; prevents "make coarse denser")
    A0_nnz = ml.levels[0].A.nnz
    total_A_nnz = sum(lvl.A.nnz for lvl in ml.levels)
    op_complex = float(total_A_nnz / max(A0_nnz, 1))

    penalty = complexity_penalty * max(0.0, op_complex - complexity_target)
    return base_reward - penalty


# ============================================================
# 7) Training: REINFORCE with moving-average baseline
# ============================================================
def log_probe_metrics(model: AMGEdgePolicy, writer: SummaryWriter, step: int, device: str):
    """
    Runs the model on a fixed 16x16 anisotropic grid (strong x) and logs:
    - distinctness: mean(logit_strong) - mean(logit_weak)
    - entropy-proxy: std(logits)
    """
    # 1. Create a fixed probe 
    # Strong coupling in X (horizontal), weak in Y (vertical) => epsilon=0.001, theta=0.0
    N = 16
    A = get_anisotropic_problem(N, epsilon=0.001, theta=0.0)
    
    # Coordinates
    coords = build_node_coords(N)
    
    # Features
    x = node_features_for_policy(A, coords, num_vecs_in=4, iters_in=5).to(device)
    ei, ew = from_scipy_sparse_matrix(A)
    ew = ew.float()
    max_w = ew.abs().max()
    if max_w > 0: ew /= max_w
    
    ei = ei.to(device)
    ew = ew.to(device)
    
    model.eval()
    with torch.no_grad():
        logits, _ = model(x, ei, ew)
    model.train()
    
    # 2. Analyze edges physics
    # Horizontal edges: |row - col| == 1
    # Vertical edges: |row - col| == N
    
    row = ei[0]
    col = ei[1]
    diff = (row - col).abs()
    
    mask_horz = (diff == 1)
    mask_vert = (diff == N)
    
    if mask_horz.sum() > 0:
        mean_horz = logits[mask_horz].mean().item()
    else:
        mean_horz = 0.0
        
    if mask_vert.sum() > 0:
        mean_vert = logits[mask_vert].mean().item()
    else:
        mean_vert = 0.0
        
    discrimination = mean_horz - mean_vert
    
    writer.add_scalar("probe/logit_strong_x", mean_horz, step)
    writer.add_scalar("probe/logit_weak_y", mean_vert, step)
    writer.add_scalar("probe/discrimination", discrimination, step)
    
    # Logit distribution properties
    writer.add_histogram("probe/logit_dist", logits, step)

    return {
        "strong_x": mean_horz,
        "weak_y": mean_vert,
        "discrimination": discrimination
    }


@dataclass
class TrainConfig:
    device: str = "cpu"
    epochs: int = 40
    k_per_row: int = 3
    temperature: float = 0.9
    lr: float = 2e-3
    grad_clip: float = 1.0

    # reward evaluation
    reward_test_vecs: int = 6
    reward_relax_iters: int = 25

    # constraints
    complexity_target: float = 1.35
    complexity_penalty: float = 1.0

    # RL variance reduction
    baseline_momentum: float = 0.95

    # pyamg
    coarse_solver: str = "splu"


def plot_training_curves(history: Dict[str, List[float]], save_dir: str):
    """
    Plots training metrics (Reward) and Probe metrics (Learning Curve) to PNGs.
    """
    import matplotlib.pyplot as plt
    ensure_dir(save_dir)

    # 1. Rewards (per step or smoothed)
    if "train_reward" in history and history["train_reward"]:
        plt.figure(figsize=(10, 5))
        r = history["train_reward"]
        plt.plot(r, alpha=0.3, label="Raw Reward")
        # smoothing
        window = 50
        if len(r) > window:
            smooth = np.convolve(r, np.ones(window)/window, mode='valid')
            plt.plot(np.arange(window-1, len(r)), smooth, label=f"Smoothed (w={window})", linewidth=2)
        
        plt.xlabel("Step")
        plt.ylabel("Reward (V-Cycle Efficiency)")
        plt.title("Training Reward over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_reward.png"), dpi=150)
        plt.close()

    # 2. Probe Metrics (per epoch)
    if "epochs" in history and history["epochs"]:
        epochs = history["epochs"]
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history["probe_strong"], 'g-o', label="Strong Edge Logit (X)")
        plt.plot(epochs, history["probe_weak"], 'r-x', label="Weak Edge Logit (Y)")
        plt.plot(epochs, history["probe_disc"], 'b--', linewidth=2, label="Discrimination (Strong - Weak)")
        
        plt.xlabel("Epoch")
        plt.ylabel("Average Logit Output")
        plt.title("Probe: Learning the Anisotropy Physics")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_probe_learning_curve.png"), dpi=150)
        plt.close()
    
    print(f"Training plots saved to: {save_dir}")


def train_policy(model: AMGEdgePolicy, train_set: List[TrainSample], cfg: TrainConfig, writer: SummaryWriter) -> Dict:
    model.to(cfg.device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = {
        "train_reward": [],
        "epochs": [],
        "probe_strong": [],
        "probe_weak": [],
        "probe_disc": []
    }

    baseline = 0.0

    step = 0
    for epoch in range(cfg.epochs):
        np.random.shuffle(train_set)

        for idx, sample in enumerate(train_set):
            step += 1

            x = sample.x.to(cfg.device)
            ei = sample.edge_index.to(cfg.device)
            ew = sample.edge_weight.to(cfg.device)

            logits, B_extra = model(x, ei, ew)

            # sample edges from policy
            selected_mask_t, logp_sum = sample_topk_without_replacement(
                logits=logits,
                row_groups=sample.row_groups,
                k=cfg.k_per_row,
                temperature=cfg.temperature,
                gumbel=True,
            )

            # build C and run PyAMG to evaluate reward (black-box)
            selected_mask = selected_mask_t.detach().cpu().numpy()
            C = C_from_selected_edges(sample.A, sample.edge_index, selected_mask)

            B = build_B_for_pyamg(B_extra)  # optional learned B
            try:
                ml = build_pyamg_solver(sample.A, C, B, coarse_solver=cfg.coarse_solver)
                R = compute_reward(
                    A=sample.A, ml=ml,
                    num_test_vecs=cfg.reward_test_vecs,
                    relax_iters=cfg.reward_relax_iters,
                    complexity_target=cfg.complexity_target,
                    complexity_penalty=cfg.complexity_penalty
                )
            except Exception as e:
                # If PyAMG fails (rare), give a strong negative reward.
                R = -5.0

            # update moving baseline
            baseline = cfg.baseline_momentum * baseline + (1.0 - cfg.baseline_momentum) * R
            advantage = float(R - baseline)

            # REINFORCE loss: -advantage * logprob
            loss = -(torch.tensor(advantage, device=cfg.device, dtype=torch.float32) * logp_sum)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            # logging
            writer.add_scalar("train/reward", R, step)
            writer.add_scalar("train/baseline", baseline, step)
            writer.add_scalar("train/advantage", advantage, step)
            writer.add_scalar("train/loss", float(loss.detach().cpu().item()), step)
            
            history["train_reward"].append(R)

        # mild temperature anneal (optional)
        cfg.temperature = max(0.5, cfg.temperature * 0.97)
        writer.add_scalar("train/temperature", cfg.temperature, epoch)

        # Log Probe Metrics (how well is it separating X vs Y?)
        probe_stats = log_probe_metrics(model, writer, step, cfg.device)
        history["epochs"].append(epoch + 1)
        history["probe_strong"].append(probe_stats["strong_x"])
        history["probe_weak"].append(probe_stats["weak_y"])
        history["probe_disc"].append(probe_stats["discrimination"])

        print(f"epoch {epoch+1:03d} | baseline={baseline:.4f} | temp={cfg.temperature:.3f}")
    
    return history



# ============================================================
# 8a) Visualization helpers: P sparsity + iteration summary
# ============================================================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def plot_c_comparison_heatmap(C1: sp.spmatrix, C2: sp.spmatrix, title: str, save_path: str, bins: int = 512):
    """
    Overlay density heatmap: C1=Red, C2=Blue.
    Purple indicates overlap.
    """
    import matplotlib.pyplot as plt

    if not sp.isspmatrix(C1): C1 = sp.csr_matrix(C1)
    if not sp.isspmatrix(C2): C2 = sp.csr_matrix(C2)

    def get_hist(M, shape, bins):
        Mc = M.tocoo()
        r = Mc.row
        c = Mc.col
        nrows, ncols = shape
        br = np.minimum((r * bins) // max(nrows, 1), bins - 1)
        bc = np.minimum((c * bins) // max(ncols, 1), bins - 1)
        H = np.zeros((bins, bins), dtype=np.float32)
        np.add.at(H, (br, bc), 1)
        return np.log1p(H)

    # Use max shape
    nrows = max(C1.shape[0], C2.shape[0])
    ncols = max(C1.shape[1], C2.shape[1])
    
    H1 = get_hist(C1, (nrows, ncols), bins)
    H2 = get_hist(C2, (nrows, ncols), bins)

    # Normalize independently for visibility
    H1_norm = H1 / (H1.max() + 1e-9)
    H2_norm = H2 / (H2.max() + 1e-9)

    # RGB
    # R = C1 (Std)
    # B = C2 (Learned)
    # G = 0
    img = np.zeros((bins, bins, 3), dtype=np.float32)
    img[..., 0] = H1_norm
    img[..., 2] = H2_norm
    # add a bit of green to make intersection lighter/white-ish if desired, or keep it purple
    # purple (1, 0, 1) is good contrast against black.

    plt.figure(figsize=(6, 6))
    plt.imshow(img, origin="upper", interpolation="nearest")
    plt.title(f"{title}\nRed=Std, Blue=Lrn, Purple=Both")
    plt.xlabel("column (binned)")
    plt.ylabel("row (binned)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_c_comparison_spy(C1: sp.spmatrix, C2: sp.spmatrix, title: str, save_path: str,
                          center: Tuple[int, int], window: int = 50):
    """
    Direct spy overlay of a zoomed region.
    C1=Red, C2=Blue.
    """
    import matplotlib.pyplot as plt
    if not sp.isspmatrix(C1): C1 = sp.csr_matrix(C1)
    if not sp.isspmatrix(C2): C2 = sp.csr_matrix(C2)

    r_start = max(0, center[0] - window)
    r_end = min(C1.shape[0], center[0] + window)
    c_start = max(0, center[1] - window)
    c_end = min(C1.shape[1], center[1] + window)

    slice1 = C1[r_start:r_end, c_start:c_end].tocoo()
    slice2 = C2[r_start:r_end, c_start:c_end].tocoo()

    # Create image grid
    H = r_end - r_start
    W = c_end - c_start
    img = np.ones((H, W, 3), dtype=np.float32) # white background

    # Let's map entries:
    # 0 = white (1,1,1)
    # C1 only = Red (1, 0, 0)
    # C2 only = Blue (0, 0, 1)
    # Both = Purple (0.5, 0, 0.5) or Dark

    # Better: init with 1s. 
    # Subtract G and B where C1 is present -> leaves R
    # Subtract R and G where C2 is present -> leaves B
    # If both present -> Subtract R, G, B? -> leaves Black?
    
    # Approach 2: black background
    img = np.zeros((H, W, 3), dtype=np.float32)
    
    # Helper to fill
    def fill_channel(Sl, ch_idx):
        # sparse coords are relative to slice
        # Sl.row, Sl.col
        # we can't do indexed assignment easily for dupes, but spy is binary
        # so just mark presence
        mask = np.zeros((H, W), dtype=bool)
        mask[Sl.row, Sl.col] = True
        img[..., ch_idx][mask] = 1.0

    fill_channel(slice1, 0) # R
    fill_channel(slice2, 2) # B

    plt.figure(figsize=(6, 6))
    plt.imshow(img, origin="upper", interpolation="nearest")
    # ticks
    plt.xticks(
        [0, W], 
        [f"{c_start}", f"{c_end}"]
    )
    plt.yticks(
        [0, H],
        [f"{r_start}", f"{r_end}"]
    )
    plt.title(f"{title}\nZoom [{r_start}:{r_end}, {c_start}:{c_end}]\nRed=Std, Blue=Lrn, Purple=Both")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_sparse_pattern(M: sp.spmatrix, title: str, save_path: str, markersize: float = 0.5,
                        max_rows: Optional[int] = None, max_cols: Optional[int] = None):
    """
    Save a sparsity plot (spy) of a sparse matrix.
    Works with CSR/CSC/COO/BSR (PyAMG P is often BSR).
    Optionally crop to top-left (max_rows, max_cols).
    """
    import matplotlib.pyplot as plt

    if not sp.isspmatrix(M):
        M = sp.csr_matrix(M)

    # IMPORTANT: many PyAMG matrices (e.g., P) are BSR; BSR slicing may raise NotImplementedError.
    # Convert once up-front to a sliceable format.
    Mc = M.tocsr()

    if max_rows is not None or max_cols is not None:
        r = max_rows if max_rows is not None else Mc.shape[0]
        c = max_cols if max_cols is not None else Mc.shape[1]
        r = min(int(r), Mc.shape[0])
        c = min(int(c), Mc.shape[1])
        Mc = Mc[:r, :c]  # CSR slicing is supported

    plt.figure()
    plt.spy(Mc, markersize=markersize)
    plt.title(f"{title}\nshape={M.shape}, nnz={M.nnz}, type={type(M).__name__}")
    plt.xlabel("column")
    plt.ylabel("row")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_iteration_bar(it_sa: int, it_ml: int, title: str, save_path: str):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.bar(["Std-AMG", "LearnedC"], [it_sa, it_ml])
    plt.ylabel("PCG Iterations")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# 8) Evaluation: PCG iterations using learned C
# ============================================================
def build_C_from_model(A: sp.csr_matrix, grid_n: int, model: AMGEdgePolicy, k_per_row: int,
                       device: str = "cpu") -> Tuple[sp.csr_matrix, Optional[np.ndarray]]:
    coords = build_node_coords(grid_n)
    x = node_features_for_policy(A, coords, num_vecs_in=4, iters_in=5).to(device)

    ei, ew = from_scipy_sparse_matrix(A)
    ew = ew.float()
    max_w = ew.abs().max()
    if max_w > 0:
        ew = ew / max_w

    ei = ei.to(device)
    ew = ew.to(device)

    model.eval()
    with torch.no_grad():
        logits, B_extra = model(x, ei, ew)

    # deterministic top-k by logits per row (no sampling) for evaluation
    row_groups = build_row_groups(ei.cpu(), num_nodes=A.shape[0])
    selected = np.zeros(A.nnz, dtype=bool)
    logits_np = logits.cpu().numpy()

    # Map edge indices back to A_coo ordering:
    # from_scipy_sparse_matrix returns edges aligned with A's internal coo extraction;
    # A_coo below matches that ordering for most SciPy matrices, but to be safe,
    # we build C using the *selected mask in that same order* using A.tocoo().
    for i, edges_i in enumerate(row_groups):
        if edges_i.numel() == 0:
            continue
        edges = edges_i.numpy()
        # pick top-k edges for that row (excluding diagonal already logits-masked)
        # Dynamic K: Select top-k, but ONLY if logit > -1.0 (some threshold)
        # This allows the model to pick FEWER than k edges if confidence is low.
        kk = min(k_per_row, edges.shape[0])
        if kk <= 0:
            continue
        
        vals = logits_np[edges]
        # Sort desc
        top_indices = np.argsort(vals)[::-1]
        
        # Take up to k, but filter by threshold
        # Using a loose threshold to allow "learning" to prune. 
        # Logits can be arbitrary but roughly centered around 0 if unconstrained, 
        # or negative/positive if softmaxed. Here they are raw logits.
        
        # Hard threshold strategy:
        # candidate_indices = top_indices[:kk]
        # valid_mask = vals[candidate_indices] > -5.0 # Arbitrary low implementation check
        # final_indices = candidate_indices[valid_mask]
        
        # For now, let's keep it simple: just strict Top-K for consistency with training.
        # To truly enable dynamic K, we would need to train with a penalty on edge count or a threshold.
        # But we can simulate "checks" by printing stats on logit distribution?
        
        # User asked for "checks". I will infer they want the Eigenvalue checks enabled.
        # So I will NOT change the logic here significantly, but I will fix the N size in main.
        
        top = top_indices[:kk]
        selected[edges[top]] = True

    C = C_from_selected_edges(A, ei.cpu(), selected)

    B = build_B_for_pyamg(B_extra)
    return C, B

def compute_C_overlap_stats(C1: sp.csr_matrix, C2: sp.csr_matrix) -> Dict[str, float]:
    """
    Compare two binary strength matrices.
    Returns IoU, intersection, size1, size2.
    """
    # Assuming C contains ones for str connections + diagonal
    # Ensure binary
    c1 = (np.abs(C1) > 1e-9).astype(int)
    c2 = (np.abs(C2) > 1e-9).astype(int)
    
    # intersection
    inter = c1.multiply(c2)
    nnz_inter = inter.nnz
    nnz_c1 = c1.nnz
    nnz_c2 = c2.nnz
    
    # union = c1 + c2 - inter
    nnz_union = nnz_c1 + nnz_c2 - nnz_inter
    
    iou = nnz_inter / max(nnz_union, 1)
    
    print(f"--- C Overlap Stats ---")
    print(f"  C_std edges:    {nnz_c1}")
    print(f"  C_lrn edges:    {nnz_c2}")
    print(f"  Intersection:   {nnz_inter}")
    print(f"  IoU:            {iou:.4f}")
    print(f"  % Std in Lrn:   {nnz_inter/max(nnz_c1, 1):.4f}")
    print(f"  % Lrn in Std:   {nnz_inter/max(nnz_c2, 1):.4f}")
    print(f"-----------------------")
    
    return {
        "iou": iou,
        "n_std": nnz_c1,
        "n_lrn": nnz_c2,
        "n_inter": nnz_inter
    }

def pcg_iterations(A: sp.csr_matrix, M, tol: float = 1e-8, maxiter: int = 500) -> int:
    n = A.shape[0]
    b = np.ones(n)
    residuals = []
    def cb(xk):
        r = b - A @ xk
        residuals.append(np.linalg.norm(r))
    spla.cg(A, b, M=M, tol=tol, maxiter=maxiter, callback=cb)
    return len(residuals), residuals

def plot_eigenvalues(A: sp.csr_matrix, M_std: spla.LinearOperator, M_lrn: spla.LinearOperator, title: str, save_path: str):
    """
    Compute and plot eigenvalues of M^-1 A for both preconditioners.
    Small grids only!
    """
    import matplotlib.pyplot as plt
    try:
        from scipy.linalg import eigvals
    except ImportError:
        return

    # Densify for eigenvalue computation (only do this for small N!)
    if A.shape[0] > 2000:
        print(f"Skipping eigenvalue plot for N={A.shape[0]} (too large)")
        return

    print(f"Computing dense eigenvalues for N={A.shape[0]} (may take a moment)...")
    Ad = A.toarray()
    
    # Construct M^-1 explicitly
    def get_dense_precond(M_op, n):
        I = np.eye(n)
        Minv = np.zeros((n, n))
        for i in range(n):
            Minv[:, i] = M_op.matvec(I[:, i])
        return Minv

    Minv_std = get_dense_precond(M_std, A.shape[0])
    Minv_lrn = get_dense_precond(M_lrn, A.shape[0])

    PA_std = Minv_std @ Ad
    PA_lrn = Minv_lrn @ Ad

    evals_std = np.linalg.eigvals(PA_std)
    evals_lrn = np.linalg.eigvals(PA_lrn)

    # Sort real parts
    ev_std = np.sort(np.real(evals_std))
    ev_lrn = np.sort(np.real(evals_lrn))
    
    # Filter out near-zeros (null space) for condition number calculation
    # AMG usually leaves one zero mode (constant vectors) if not handled, or near-zero
    min_std = ev_std[ev_std > 1e-6].min() if (ev_std > 1e-6).any() else 1e-6
    max_std = ev_std.max()
    cond_std = max_std / min_std

    min_lrn = ev_lrn[ev_lrn > 1e-6].min() if (ev_lrn > 1e-6).any() else 1e-6
    max_lrn = ev_lrn.max()
    cond_lrn = max_lrn / min_lrn

    # --- Plot 1: Sorted Eigenvalues (The "Spectra" plot you liked)
    plt.figure(figsize=(8, 6))
    plt.plot(ev_std, label=f"Std-AMG ($\kappa={cond_std:.1f}$)", marker='.', linestyle='--', alpha=0.6)
    plt.plot(ev_lrn, label=f"Learned ($\kappa={cond_lrn:.1f}$)", marker='x', linestyle='-', alpha=0.6)
    plt.axhline(1.0, color='k', linestyle=':', alpha=0.3)
    plt.title(f"Eigenvalue Spectra of $M^{{-1}}A$\n{title}")
    plt.ylabel("Eigenvalue $\lambda$")
    plt.xlabel("Index")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    
    # --- Plot 2: Spectral Histogram (Density of Eigenvalues)
    # Good for seeing clustering around 1.0 (ideal for CG)
    plt.figure(figsize=(8, 6))
    plt.hist(ev_std, bins=50, alpha=0.5, label="Std-AMG", density=True, color='red')
    plt.hist(ev_lrn, bins=50, alpha=0.5, label="Learned", density=True, color='blue')
    plt.axvline(1.0, color='k', linestyle='--', alpha=0.5)
    plt.title(f"Eigenvalue Density Histogram\n{title}")
    plt.xlabel("Eigenvalue $\lambda$")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    base, ext = os.path.splitext(save_path)
    plt.savefig(f"{base}_hist{ext}", dpi=200)
    plt.close()

    # --- Plot 3: Eigenvector Localization (optional, advanced)
    # Are the worst modes low-frequency? (Smooth) or high-freq?
    # This is hard to visualize in 1D plot without context, so skipping for now 
    # unless requested.


def plot_weight_vs_logit(model: AMGEdgePolicy, A: sp.csr_matrix, grid_n: int, save_path: str, device="cpu"):
    """
    Scatter plot of Edge Weight vs Model Logit.
    Checks if the model learned a trivial mapping.
    """
    import matplotlib.pyplot as plt
    
    coords = build_node_coords(grid_n)
    x = node_features_for_policy(A, coords).to(device)
    ei, ew = from_scipy_sparse_matrix(A)
    # Re-normalize weights as in training for fair comparison
    ew_raw = ew.float()
    ew_norm = ew_raw / (ew_raw.abs().max() + 1e-9)
    
    ei = ei.to(device)
    ew_norm = ew_norm.to(device)
    
    model.eval()
    with torch.no_grad():
        logits, _ = model(x, ei, ew_norm)
    
    # Filter out diagonals (usually masked out anyway)
    row, col = ei.cpu().numpy()
    mask = (row != col)
    
    w_vals = ew_norm.cpu().numpy()[mask]
    l_vals = logits.cpu().numpy()[mask]
    
    plt.figure()
    plt.scatter(w_vals, l_vals, alpha=0.3, s=5)
    plt.xlabel("Normalized Edge Weight")
    plt.ylabel("Learned Logit")
    plt.title("Correlation: Is it just learning weights?")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def evaluate_on_case(A: sp.csr_matrix, model: AMGEdgePolicy, grid_n: int, k_per_row: int,
                     coarse_solver: str = "splu", tol: float = 1e-8,
                     out_dir: str = "outputs", tag: str = "case",
                     crop_P: bool = False):
    """
    Evaluates Std-AMG vs LearnedC on one case:
      - PCG iteration counts + residual histories
      - Sparsity visualization of prolongation P (Std vs LearnedC)
      - Iteration bar chart
      - *NEW* Strength Theta=0.25 baseline
      - *NEW* Eigenvalue plots
    """
    ensure_dir(out_dir)

    # ---- Standard SA (theta=0.0 default)
    sa_std = pyamg.smoothed_aggregation_solver(A, coarse_solver=coarse_solver)
    M_std = sa_std.aspreconditioner(cycle="V")
    it_std, res_std = pcg_iterations(A, M_std, tol=tol)

    # ---- Standard SA (theta=0.25 tuned)
    # Typical Ruge-Stuben threshold
    C_tuned = symmetric_strength_of_connection(A, theta=0.25)
    sa_tuned = pyamg.smoothed_aggregation_solver(A, strength=('predefined', {'C': C_tuned}), coarse_solver=coarse_solver)
    M_tuned = sa_tuned.aspreconditioner(cycle="V")
    it_tuned, res_tuned = pcg_iterations(A, M_tuned, tol=tol)

    # ---- LearnedC SA
    C_ml, B_ml = build_C_from_model(A, grid_n, model, k_per_row=k_per_row, device="cpu")
    ml = build_pyamg_solver(A, C_ml, B_ml, coarse_solver=coarse_solver)
    M_ml = ml.aspreconditioner(cycle="V")
    it_ml, res_ml = pcg_iterations(A, M_ml, tol=tol)

    # ---- C Comparison
    C_std = symmetric_strength_of_connection(A, theta=0.0)
    print(f"\n[Case: {tag}] C_std vs C_learned:")
    compute_C_overlap_stats(C_std, C_ml)

    # ---- Eigenvalue Analysis (only if small enough)
    plot_eigenvalues(A, M_std, M_ml, title=f"Spectra ({tag})", save_path=os.path.join(out_dir, f"{tag}_spectra.png"))

    # ---- Weight vs Logit Analysis
    plot_weight_vs_logit(model, A, grid_n, save_path=os.path.join(out_dir, f"{tag}_logits_vs_weights.png"))

    plot_c_comparison_heatmap(
        C_std, C_ml,
        title=f"Strength C Overlay ({tag})",
        save_path=os.path.join(out_dir, f"{tag}_C_overlay_heat.png"),
        bins=512
    )

    # ---- Plot convergence
    import matplotlib.pyplot as plt
    plt.figure()
    plt.semilogy(range(len(res_std)), res_std, label="PCG + Std (theta=0.0)")
    plt.semilogy(range(len(res_tuned)), res_tuned, label="PCG + Tuned (theta=0.25)", linestyle='--')
    plt.semilogy(range(len(res_ml)), res_ml, label="PCG + Learned", linewidth=2)
    plt.xlabel("PCG Iteration")
    plt.ylabel("Residual Norm")
    plt.title(f"PCG Convergence Comparison ({tag})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    conv_path = os.path.join(out_dir, f"{tag}_pcg_convergence.png")
    plt.savefig(conv_path, dpi=200)
    plt.close()

    # ---- Visualize P sparsity (prolongation operators)
    P_std = sa_std.levels[0].P
    P_lrn = ml.levels[0].P

    # Heatmap-style sparsity density (better than spy for large P)
    plot_sparse_density_heatmap(
        P_std,
        title=f"P density heatmap (Std-AMG) [{tag}]",
        save_path=os.path.join(out_dir, f"{tag}_P_std_heat.png"),
        bins=512,
        log_scale=True,
        crop=(20000, 20000) if crop_P else None,  # optional
    )

    plot_sparse_density_heatmap(
        P_lrn,
        title=f"P density heatmap (LearnedC) [{tag}]",
        save_path=os.path.join(out_dir, f"{tag}_P_learned_heat.png"),
        bins=512,
        log_scale=True,
        crop=(20000, 20000) if crop_P else None,  # optional
    )

    max_rows = 2000 if crop_P else None
    max_cols = 2000 if crop_P else None

    plot_sparse_pattern(
        P_std,
        title=f"Prolongation P (Std-AMG) [{tag}]",
        save_path=os.path.join(out_dir, f"{tag}_P_std.png"),
        markersize=0.5,
        max_rows=max_rows,
        max_cols=max_cols,
    )
    plot_sparse_pattern(
        P_lrn,
        title=f"Prolongation P (LearnedC) [{tag}]",
        save_path=os.path.join(out_dir, f"{tag}_P_learned.png"),
        markersize=0.5,
        max_rows=max_rows,
        max_cols=max_cols,
    )

    # ---- Iteration bar chart
    plot_iteration_bar(
        it_std, it_ml,
        title=f"PCG Iterations ({tag})",
        save_path=os.path.join(out_dir, f"{tag}_pcg_iters.png")
    )

    print(f"Saved: {conv_path}")
    print(f"Saved: {os.path.join(out_dir, f'{tag}_P_std.png')}")
    print(f"Saved: {os.path.join(out_dir, f'{tag}_P_learned.png')}")
    print(f"Saved: {os.path.join(out_dir, f'{tag}_pcg_iters.png')}")
    print(f"Saved: {os.path.join(out_dir, f'{tag}_logits_vs_weights.png')}")
    if A.shape[0] <= 1200:
        print(f"Saved: {os.path.join(out_dir, f'{tag}_spectra.png')}")

    print(f"PCG + Std (0.0)  | {it_std:4d} iters")
    print(f"PCG + Std (0.25) | {it_tuned:4d} iters")
    print(f"PCG + Learned    | {it_ml:4d} iters")

    return {
        "tag": tag,
        "it_std": it_std,
        "it_tuned": it_tuned,
        "it_learned": it_ml,
    }



def evaluate_suite_after_training(model: AMGEdgePolicy, out_dir: str, coarse_solver: str = "splu"):
    """
    Runs a small suite and saves:
      - per-case convergence plots
      - per-case P sparsity plots (Std vs LearnedC)
      - per-case iteration bar charts
      - one aggregate bar chart across cases
    """
    ensure_dir(out_dir)

    cases = [
        ("iso_eps1_theta0", 128, 1.0, 0.0),
        ("axis_aniso_eps1e-3_theta0", 128, 0.001, 0.0),
        ("rot_aniso_eps1e-2_theta45", 128, 0.01, float(np.pi/4)),
    ]

    results = []
    for tag, N, eps, th in cases:
        A = get_anisotropic_problem(N, epsilon=eps, theta=th)
        r = evaluate_on_case(
            A, model, grid_n=N, k_per_row=3,
            coarse_solver=coarse_solver, tol=1e-8,
            out_dir=out_dir, tag=tag,
            crop_P=True  # helpful for large grids
        )
        results.append(r)

    # Aggregate iterations plot
    import matplotlib.pyplot as plt
    labels = [r["tag"] for r in results]
    std = [r["it_std"] for r in results]
    lrn = [r["it_learned"] for r in results]

    x = np.arange(len(labels))
    w = 0.35

    plt.figure()
    plt.bar(x - w/2, std, width=w, label="Std-AMG")
    plt.bar(x + w/2, lrn, width=w, label="LearnedC")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("PCG Iterations")
    plt.title("Post-training PCG iterations across evaluation suite")
    plt.legend()
    plt.tight_layout()
    agg_path = os.path.join(out_dir, "suite_pcg_iters.png")
    plt.savefig(agg_path, dpi=200)
    plt.close()

    print(f"Saved: {agg_path}")
    return results
def plot_sparse_density_heatmap(
    M: sp.spmatrix,
    title: str,
    save_path: str,
    bins: int = 512,
    log_scale: bool = True,
    crop: Optional[Tuple[int, int]] = None,  # (max_rows, max_cols) in original indices
):
    """
    Heatmap-style visualization of sparsity: bin (row,col) locations of nonzeros.
    Does NOT densify M.

    bins: resolution of the heatmap in each dimension (bins x bins)
    log_scale: use log1p(counts) to reveal structure
    crop: if not None, restrict to top-left (max_rows, max_cols) region before binning
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if not sp.isspmatrix(M):
        M = sp.csr_matrix(M)

    # Convert to COO for easy access to coordinates
    Mc = M.tocoo(copy=False)

    r = Mc.row.astype(np.int64, copy=False)
    c = Mc.col.astype(np.int64, copy=False)

    nrows, ncols = Mc.shape

    if crop is not None:
        max_r, max_c = crop
        max_r = min(int(max_r), nrows)
        max_c = min(int(max_c), ncols)
        keep = (r < max_r) & (c < max_c)
        r = r[keep]
        c = c[keep]
        nrows, ncols = max_r, max_c

    if r.size == 0:
        # nothing to plot
        plt.figure()
        plt.title(f"{title}\n(empty)")
        plt.savefig(save_path, dpi=200)
        plt.close()
        return

    # Map coordinates -> bin indices
    # Use floor((idx / size) * bins), clamped to [0, bins-1]
    br = np.minimum((r * bins) // max(nrows, 1), bins - 1)
    bc = np.minimum((c * bins) // max(ncols, 1), bins - 1)

    # Accumulate counts into a bins x bins grid
    H = np.zeros((bins, bins), dtype=np.int32)
    np.add.at(H, (br, bc), 1)

    if log_scale:
        H_plot = np.log1p(H.astype(np.float64))
        scale_note = "log1p(count)"
    else:
        H_plot = H.astype(np.float64)
        scale_note = "count"

    plt.figure()
    # origin='upper' keeps row 0 at top like spy()
    im = plt.imshow(H_plot, origin="upper", aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04, label=scale_note)
    plt.title(f"{title}\nshape={M.shape}, nnz={M.nnz}, bins={bins} ({scale_note})")
    plt.xlabel("column (binned)")
    plt.ylabel("row (binned)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def verify_learned_logic(model: AMGEdgePolicy, device: str = "cpu"):
    """
    Probes the model on strict Anisotropic layouts to see if it learned the correct physics-based rule.
    """
    print("\n=== Verifying Learned Logic ===")
    model.eval()
    
    # Case 1: Strong X-direction Anisotropy (epsilon=0.001, theta=0)
    # Stencil: -eps in Y, -1 in X. Strong coupling in X.
    # We expect logits for horizontal edges to be higher than vertical.
    N = 16
    A_x = get_anisotropic_problem(N, epsilon=0.001, theta=0.0)
    
    # Construct features
    coords = build_node_coords(N)
    x = node_features_for_policy(A_x, coords, num_vecs_in=4, iters_in=5).to(device)
    ei, ew = from_scipy_sparse_matrix(A_x)
    ew = ew.float()
    ew = ew / (ew.abs().max() + 1e-9)
    ei = ei.to(device)
    ew = ew.to(device)
    
    with torch.no_grad():
        logits, _ = model(x, ei, ew)
    
    # Analyze middle node
    mid = (N * N) // 2
    row = ei[0].cpu().numpy()
    col = ei[1].cpu().numpy()
    
    # Find edges starting from 'mid'
    mask = (row == mid)
    neighbors = col[mask]
    l_vals = logits[mask].cpu().numpy()
    
    print(f"Probe Node {mid} (Axis-Aligned, Strong X):")
    # Identify neighbor directions based on id diff
    # +1/-1 is Horz, +N/-N is Vert
    for nbr, val in zip(neighbors, l_vals):
        diff = nbr - mid
        direction = "Unknown"
        if abs(diff) == 1: direction = "Horizontal (Strong)"
        elif abs(diff) == N: direction = "Vertical   (Weak)"
        print(f"  -> Neighbor {nbr} [{direction}]: Logit = {val:.4f}")

    # Case 2: Strong Y-direction Anisotropy (epsilon=0.001, theta=pi/2)
    A_y = get_anisotropic_problem(N, epsilon=0.001, theta=np.pi/2)
    # ... (similar setup) ...
    x2 = node_features_for_policy(A_y, coords, num_vecs_in=4, iters_in=5).to(device)
    ei2, ew2 = from_scipy_sparse_matrix(A_y)
    ew2 = ew2.float()
    ew2 = ew2 / (ew2.abs().max() + 1e-9)
    ei2 = ei2.to(device)
    ew2 = ew2.to(device)

    with torch.no_grad():
        logits2, _ = model(x2, ei2, ew2)
        
    mask2 = (ei2[0].cpu().numpy() == mid)
    neighbors2 = ei2[1].cpu().numpy()[mask2]
    l_vals2 = logits2[mask2].cpu().numpy()
    
    print(f"Probe Node {mid} (Rotated 90 deg, Strong Y):")
    for nbr, val in zip(neighbors2, l_vals2):
        diff = nbr - mid
        direction = "Unknown"
        if abs(diff) == 1: direction = "Horizontal (Weak)"
        elif abs(diff) == N: direction = "Vertical   (Strong)"
        print(f"  -> Neighbor {nbr} [{direction}]: Logit = {val:.4f}")


# ============================================================
# 9) Main
# ============================================================
if __name__ == "__main__":
    writer, log_dir = setup_tensorboard()
    tb_process = launch_tensorboard(log_dir)

    try:
        # ------------------------
        # Train configuration
        # ------------------------
        cfg = TrainConfig(
            device="cpu",          # set "cuda" if available
            epochs=10,
            k_per_row=3,           # <= typical degree; safer on boundaries
            temperature=0.9,
            lr=2e-3,
            grad_clip=1.0,
            reward_test_vecs=6,
            reward_relax_iters=25,
            complexity_target=1.35,
            complexity_penalty=1.0,
            baseline_momentum=0.95,
            coarse_solver="splu",
        )

        # ------------------------
        # Build training set
        # ------------------------
        TRAIN_GRID = 16 # original was 32
        train_set = make_train_set(num_samples=24, grid_n=TRAIN_GRID, seed=0)
        print(f"Training set: {len(train_set)} samples on {TRAIN_GRID}x{TRAIN_GRID} (matrix size {train_set[0].A.shape[0]} x {train_set[0].A.shape[1]})")

        # ------------------------
        # Model
        # ------------------------
        model = AMGEdgePolicy(in_channels=6, hidden=64, learn_B=True, B_extra=2)

        # ------------------------
        # Train
        # ------------------------
        train_history = train_policy(model, train_set, cfg, writer)
        
        # Save training plots
        plot_training_curves(train_history, save_dir=log_dir) # Save to TensorBoard dir

        # ------------------------
        # Test on a harder rotated anisotropy, larger grid
        # ------------------------
        TEST_GRID = 256
        A_test = get_anisotropic_problem(TEST_GRID, epsilon=0.01, theta=np.pi/4.5)
        print("\n=== Evaluation: Rotated anisotropy ===")
        evaluate_on_case(A_test, model, grid_n=TEST_GRID, k_per_row=3, coarse_solver="splu", tol=1e-8)

        # ------------------------
        # Verify Logic
        # ------------------------
        verify_learned_logic(model, device=cfg.device)

        # ------------------------
        # Post-training evaluation suite (includes P sparsity + iteration plots)
        # ------------------------
        out_dir = ensure_dir(os.path.join("outputs", datetime.now().strftime("%Y%m%d-%H%M%S")))
        print(f"\n=== Post-training evaluation suite ===\nSaving outputs to: {out_dir}")
        
        # Add a SMALL case specifically for Spectrum verification
        print("\n--- Small Scale Spectral Verification (N=16) ---")
        A_small = get_anisotropic_problem(16, epsilon=0.001, theta=0.0)
        evaluate_on_case(A_small, model, grid_n=16, k_per_row=3, coarse_solver="splu", tol=1e-8, out_dir=out_dir, tag="spectral_check_N16")

        suite_results = evaluate_suite_after_training(model, out_dir=out_dir, coarse_solver="splu")


    finally:
        writer.close()
        print("\n[System] Shutting down TensorBoard...")
        tb_process.terminate()
