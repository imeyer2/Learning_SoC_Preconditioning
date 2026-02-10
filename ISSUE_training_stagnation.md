# Training Stagnation: Mean Reward Does Not Improve Across Epochs

## Summary

Training runs showed **no meaningful improvement in mean reward** across dozens of epochs. 

**Root cause identified:** A critical bug where entropy was computed inside `torch.no_grad()`, making the entropy coefficient completely ineffective.

**Status:** ✅ **FIX APPLIED** — Ready for verification testing.

---

## Root Cause: CRITICAL BUG FOUND ✅ FIXED

**The entropy bonus had no effect due to a bug in `reinforce.py`.**

In the original code, entropy was computed inside `torch.no_grad()`:

```python
# BUG: Entropy computed with no_grad - contributes ZERO gradients!
with torch.no_grad():
    probs = torch.softmax(data['logits'] / temperature, dim=-1)
    log_probs = torch.log_softmax(data['logits'] / temperature, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1).mean()
```

This means:
- The entropy tensor has **no gradient information**
- When added to the loss (`entropy_bonus = -self.entropy_coef * entropy`), it contributes **zero gradients**
- The `entropy_coef` hyperparameter **does nothing** to guide training toward exploration
- This directly explains the severe entropy collapse (8.8268 → 0.0041)

**Fix:** Remove the `torch.no_grad()` wrapper so entropy gradients flow back through the policy:

```python
# FIXED: Entropy computed WITH gradients - allows exploration bonus to work
probs = torch.softmax(logits / temperature, dim=-1)
log_probs = torch.log_softmax(logits / temperature, dim=-1)
entropy = -torch.sum(probs * log_probs, dim=-1).mean()
```

The fix has been applied to `src/p_sparsity/rl/algorithms/reinforce.py` in **both** training methods:
- Single-step training: lines 496-498
- Batch training: lines 679-681 (with explicit comment documenting the fix)

---

## Evidence

### Variation A (128×128 grid, 150 samples)
```
Epoch 001 | Mean Reward: 0.4404
Epoch 005 | Mean Reward: 0.4316
Epoch 010 | Mean Reward: 0.4252
Epoch 016 | Mean Reward: 0.4245

Change over 16 epochs: -3.6% (actually getting WORSE)
```

### Variation B (32×32 grid, 150 samples)
```
Epoch 001 | Mean Reward: 1.0643
Epoch 010 | Mean Reward: 1.0555
Epoch 020 | Mean Reward: 1.0524
Epoch 021 | Mean Reward: 1.0545

Change over 21 epochs: -0.9% (essentially FLAT)
```

## Root Cause Analysis

### 1. **Policy Entropy Collapse** 🔴 Critical
The policy becomes nearly deterministic within the first 10 epochs:
```
Epoch 01: Policy Entropy = 8.8268
Epoch 10: Policy Entropy = 0.0211  ← 418x decrease!
Epoch 20: Policy Entropy = 0.0041
```
**Impact**: Without exploration, the policy cannot discover better edge selection strategies.

### 2. **Baseline Not Capturing Variance** 🟠 High
```
Value Explained Variance: 0.05 - 0.07 (only 5-7%!)
Advantage Std: 0.52 - 0.55 (high variance)
```
**Impact**: Gradient estimates have extremely high variance, making learning unstable.

### 3. **Gradient Clipping Always Active** 🟠 High
```
Total Grad Norm: 1.0000 (always hitting the clip!)
```
**Impact**: Gradients are being aggressively clipped every step, potentially preventing necessary updates.

### 4. **Policy Head Gradient Collapse** 🟡 Medium
```
Epoch 01: Policy Grad Norm = 0.9874, GNN Grad Norm = 0.1582
Epoch 10: Policy Grad Norm = 0.1587, GNN Grad Norm = 0.9873
Epoch 30: Policy Grad Norm = 0.0772, GNN Grad Norm = 0.9970
```
**Impact**: As entropy collapses, gradients through the policy head vanish.

## Current Hyperparameters

```yaml
# Training
learning_rate: 0.001 - 0.002
epochs: 50
batch_size: 1-2

# Exploration
entropy_coef: 0.01  # Too low?
temperature:
  initial: 0.9
  anneal_factor: 0.97  # Anneals too fast?
  min_temperature: 0.5

# Optimization  
grad_clip: 1.0

# Baseline
baseline:
  type: exponential_moving
  momentum: 0.95
  var_momentum: 0.99
  normalize: true
  warmup_steps: 100
```

## Success Criteria

- [ ] **Minimum**: Mean reward increases by **≥20%** over training
- [ ] **Good**: Mean reward increases by **≥50%** over training  
- [ ] **Excellent**: Mean reward increases by **≥100%** over training
- [ ] Policy entropy remains above 1.0 throughout training (exploration maintained)
- [ ] Value explained variance improves to >20%

## Suggested Investigation Areas

### Priority 0: Fix Entropy Bug ✅ COMPLETE
- [x] **Remove `torch.no_grad()` wrapper** from entropy computation — **VERIFIED FIXED** (both `_train_step` and batch training)
- [ ] **Re-run training** to verify entropy no longer collapses
- [ ] **Verify mean reward increases** by ≥20%

### Priority 1: Tune Entropy (if still collapsing after fix)
- [ ] **Increase entropy coefficient** from 0.01 → 0.05-0.1
- [ ] **Slower/no temperature annealing** - keep temperature high longer
- [ ] Consider **maximum entropy RL** formulation

### Priority 2: Improve Baseline / Reduce Variance
- [ ] **Implement a learned value function baseline** (actor-critic)
- [ ] Try **multiple rollouts per problem** and average gradients
- [ ] Experiment with **GAE (Generalized Advantage Estimation)**
- [ ] Consider **PPO** instead of vanilla REINFORCE for more stable updates

### Priority 3: Hyperparameter Tuning
- [ ] **Learning rate sweep**: Try 1e-4, 5e-4, 1e-3, 5e-3
- [ ] **Gradient clip sweep**: Try 0.5, 1.0, 5.0, 10.0, or remove entirely
- [ ] **Batch size**: Accumulate gradients over 4-8 problems before update
- [ ] **Baseline momentum**: Try 0.9, 0.95, 0.99

### Priority 4: Algorithmic Improvements
- [ ] Implement **PPO** with clipped objective
- [ ] Add **reward normalization** (running mean/std)
- [ ] Try **reward shaping** with intermediate feedback
- [ ] Consider **curriculum learning** (start easy, increase difficulty)

### Priority 5: Architecture Changes
- [ ] Larger GNN (more layers, wider hidden dim)
- [ ] Different GNN backbone (GraphSAGE, GIN)
- [ ] Separate value head for actor-critic

## Reproduction

```bash
# Run Variation A
python scripts/run_case_study.py \
    --config case_studies/configs/study_1_anisotropic.yaml \
    --variation A

# Run Variation B  
python scripts/run_case_study.py \
    --config case_studies/configs/study_1_anisotropic.yaml \
    --variation B
```

Monitor `Mean Reward` in the epoch summaries and `Policy Entropy` in the diagnostics (printed every 10 epochs).

## References

- [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290) - Maximum entropy RL
- [PPO](https://arxiv.org/abs/1707.06347) - Proximal Policy Optimization
- [GAE](https://arxiv.org/abs/1506.02438) - Generalized Advantage Estimation

## Logs

Full training logs attached (pre-fix):
- `study1_varA.log` - Variation A (128×128)
- `study1_varB.log` - Variation B (32×32)

## Next Steps

1. **Re-run training** with the entropy bug fix
2. **Compare results** - expect to see entropy maintained at reasonable levels (>1.0) and mean reward increase
3. If reward still doesn't improve after fix, investigate the other priorities (baseline variance, gradient clipping, hyperparameters)
