"""
Metrics collection for case study evaluation.

Collects detailed performance metrics for scaling studies and comparison.
"""

import time
import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import json
from pathlib import Path


@dataclass
class PCGMetrics:
    """Metrics from a single PCG solve."""
    iterations: int
    converged: bool
    wall_time: float
    residual_history: List[float]
    final_residual: float
    
    # Derived metrics
    @property
    def avg_reduction_rate(self) -> float:
        """Average residual reduction per iteration."""
        if len(self.residual_history) < 2:
            return 1.0
        ratios = [
            self.residual_history[i] / self.residual_history[i-1]
            for i in range(1, len(self.residual_history))
            if self.residual_history[i-1] > 0
        ]
        return np.mean(ratios) if ratios else 1.0
    
    @property
    def time_per_iteration(self) -> float:
        """Average time per iteration."""
        return self.wall_time / max(self.iterations, 1)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'iterations': self.iterations,
            'converged': self.converged,
            'wall_time': self.wall_time,
            'residual_history': self.residual_history,
            'final_residual': self.final_residual,
            'avg_reduction_rate': self.avg_reduction_rate,
            'time_per_iteration': self.time_per_iteration,
        }


@dataclass
class VCycleMetrics:
    """Metrics from V-cycle analysis."""
    energy_history: List[float]
    reduction_factors: List[float]
    avg_reduction: float
    wall_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'energy_history': self.energy_history,
            'reduction_factors': self.reduction_factors,
            'avg_reduction': self.avg_reduction,
            'wall_time': self.wall_time,
        }


@dataclass
class SpectralMetrics:
    """Spectral properties of preconditioned system."""
    condition_estimate: Optional[float] = None
    spectral_radius: Optional[float] = None
    convergence_rate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'condition_estimate': self.condition_estimate,
            'spectral_radius': self.spectral_radius,
            'convergence_rate': self.convergence_rate,
        }


@dataclass
class SolverMetrics:
    """Complete metrics for a solver (learned, baseline, tuned)."""
    name: str
    pcg: Optional[PCGMetrics] = None
    vcycle: Optional[VCycleMetrics] = None
    spectral: Optional[SpectralMetrics] = None
    
    # Solver construction time
    setup_time: float = 0.0
    
    # Hierarchy info
    num_levels: int = 0
    coarse_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            'name': self.name,
            'setup_time': self.setup_time,
            'num_levels': self.num_levels,
            'coarse_size': self.coarse_size,
        }
        if self.pcg:
            d['pcg'] = self.pcg.to_dict()
        if self.vcycle:
            d['vcycle'] = self.vcycle.to_dict()
        if self.spectral:
            d['spectral'] = self.spectral.to_dict()
        return d


@dataclass
class ProblemMetrics:
    """Metrics for a single problem instance."""
    problem_id: str
    grid_size: int
    n: int  # DOFs
    nnz: int
    params: Dict[str, float]
    
    learned: Optional[SolverMetrics] = None
    random: Optional[SolverMetrics] = None  # Random model (ablation)
    baseline: Optional[SolverMetrics] = None
    tuned: Optional[SolverMetrics] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            'problem_id': self.problem_id,
            'grid_size': self.grid_size,
            'n': self.n,
            'nnz': self.nnz,
            'params': self.params,
        }
        if self.learned:
            d['learned'] = self.learned.to_dict()
        if self.random:
            d['random'] = self.random.to_dict()
        if self.baseline:
            d['baseline'] = self.baseline.to_dict()
        if self.tuned:
            d['tuned'] = self.tuned.to_dict()
        return d
    
    def speedup(self, solver_name: str = 'learned') -> Optional[float]:
        """Iteration speedup of specified solver vs baseline."""
        solver = getattr(self, solver_name, None)
        if solver and self.baseline and solver.pcg and self.baseline.pcg:
            if solver.pcg.iterations > 0:
                return self.baseline.pcg.iterations / solver.pcg.iterations
        return None
    
    def speedup_learned_vs_random(self) -> Optional[float]:
        """Iteration speedup of learned vs random (ablation comparison)."""
        if self.learned and self.random and self.learned.pcg and self.random.pcg:
            if self.learned.pcg.iterations > 0:
                return self.random.pcg.iterations / self.learned.pcg.iterations
        return None


@dataclass  
class ScalingMetrics:
    """
    Aggregated metrics for a scaling study (Variation B).
    
    Tracks how performance changes with problem size.
    """
    grid_sizes: List[int]
    problem_metrics: Dict[int, List[ProblemMetrics]] = field(default_factory=dict)
    
    def add_problem(self, grid_size: int, metrics: ProblemMetrics):
        """Add a problem's metrics to the scaling study."""
        if grid_size not in self.problem_metrics:
            self.problem_metrics[grid_size] = []
        self.problem_metrics[grid_size].append(metrics)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics per grid size."""
        summary = {}
        
        for grid_size in self.grid_sizes:
            problems = self.problem_metrics.get(grid_size, [])
            if not problems:
                continue
            
            # Aggregate across problems at this grid size
            learned_iters = [p.learned.pcg.iterations for p in problems 
                           if p.learned and p.learned.pcg]
            baseline_iters = [p.baseline.pcg.iterations for p in problems 
                            if p.baseline and p.baseline.pcg]
            learned_times = [p.learned.pcg.wall_time for p in problems 
                           if p.learned and p.learned.pcg]
            baseline_times = [p.baseline.pcg.wall_time for p in problems 
                            if p.baseline and p.baseline.pcg]
            speedups = [p.speedup() for p in problems if p.speedup() is not None]
            
            n = problems[0].n if problems else 0
            
            summary[grid_size] = {
                'n': n,
                'num_problems': len(problems),
                'learned_iters_mean': np.mean(learned_iters) if learned_iters else None,
                'learned_iters_std': np.std(learned_iters) if learned_iters else None,
                'baseline_iters_mean': np.mean(baseline_iters) if baseline_iters else None,
                'baseline_iters_std': np.std(baseline_iters) if baseline_iters else None,
                'learned_time_mean': np.mean(learned_times) if learned_times else None,
                'baseline_time_mean': np.mean(baseline_times) if baseline_times else None,
                'speedup_mean': np.mean(speedups) if speedups else None,
                'speedup_std': np.std(speedups) if speedups else None,
            }
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'grid_sizes': self.grid_sizes,
            'summary': self.get_summary(),
            'detailed': {
                str(gs): [p.to_dict() for p in problems]
                for gs, problems in self.problem_metrics.items()
            }
        }


class MetricsCollector:
    """
    Collector for gathering metrics during case study evaluation.
    
    Orchestrates PCG, V-cycle, and spectral analysis.
    """
    
    def __init__(
        self,
        collect_pcg: bool = True,
        collect_vcycle: bool = True,
        collect_spectral: bool = False,
        pcg_tol: float = 1e-8,
        pcg_max_iter: int = 500,
        pcg_timeout: float = 60.0,
        vcycle_iters: int = 20,
    ):
        self.collect_pcg = collect_pcg
        self.collect_vcycle = collect_vcycle
        self.collect_spectral = collect_spectral
        self.pcg_tol = pcg_tol
        self.pcg_max_iter = pcg_max_iter
        self.pcg_timeout = pcg_timeout
        self.vcycle_iters = vcycle_iters
    
    def collect_solver_metrics(
        self,
        A: sp.csr_matrix,
        ml_solver,
        solver_name: str,
        b: Optional[np.ndarray] = None,
    ) -> SolverMetrics:
        """
        Collect all metrics for a single solver.
        
        Args:
            A: System matrix
            ml_solver: PyAMG multilevel solver
            solver_name: Name for this solver (learned/baseline/tuned)
            b: Right-hand side (random if None)
            
        Returns:
            SolverMetrics with all collected data
        """
        metrics = SolverMetrics(name=solver_name)
        
        # Hierarchy info
        metrics.num_levels = len(ml_solver.levels)
        metrics.coarse_size = ml_solver.levels[-1].A.shape[0]
        
        n = A.shape[0]
        if b is None:
            b = np.random.randn(n)
        
        # PCG analysis
        if self.collect_pcg:
            metrics.pcg = self._collect_pcg_metrics(A, ml_solver, b)
        
        # V-cycle analysis
        if self.collect_vcycle:
            metrics.vcycle = self._collect_vcycle_metrics(A, ml_solver)
        
        # Spectral analysis (expensive!)
        if self.collect_spectral:
            metrics.spectral = self._collect_spectral_metrics(A, ml_solver)
        
        return metrics
    
    def _collect_pcg_metrics(
        self,
        A: sp.csr_matrix,
        ml_solver,
        b: np.ndarray
    ) -> PCGMetrics:
        """Run PCG and collect metrics."""
        from ..evaluation.pcg_analysis import run_pcg_analysis
        
        start_time = time.time()
        result = run_pcg_analysis(
            A, ml_solver,
            b=b,
            max_iter=self.pcg_max_iter,
            tol=self.pcg_tol,
            timeout=self.pcg_timeout,
            show_progress=False,
        )
        wall_time = time.time() - start_time
        
        return PCGMetrics(
            iterations=result.iterations,
            converged=result.converged,
            wall_time=wall_time,
            residual_history=result.residuals,
            final_residual=result.final_residual,
        )
    
    def _collect_vcycle_metrics(
        self,
        A: sp.csr_matrix,
        ml_solver
    ) -> VCycleMetrics:
        """Run V-cycle analysis and collect metrics."""
        from ..evaluation.vcycle_analysis import run_vcycle_analysis
        
        n = A.shape[0]
        x_true = np.random.randn(n)
        x0 = np.random.randn(n)
        
        start_time = time.time()
        result = run_vcycle_analysis(
            A, ml_solver,
            x_true=x_true,
            x0=x0,
            num_cycles=self.vcycle_iters,
        )
        wall_time = time.time() - start_time
        
        return VCycleMetrics(
            energy_history=result.energy_norms,
            reduction_factors=result.reduction_factors,
            avg_reduction=result.avg_reduction_factor,
            wall_time=wall_time,
        )
    
    def _collect_spectral_metrics(
        self,
        A: sp.csr_matrix,
        ml_solver
    ) -> SpectralMetrics:
        """Estimate spectral properties (expensive for large systems)."""
        from ..evaluation.eigenvalue_analysis import estimate_spectral_properties_from_vcycle
        
        # Use fast convergence-based estimation
        result = estimate_spectral_properties_from_vcycle(
            A, ml_solver,
            num_vecs=5,
            max_iters=10
        )
        
        return SpectralMetrics(
            condition_estimate=result.condition_number,
            spectral_radius=result.spectral_radius,
            convergence_rate=-np.log(result.spectral_radius) if result.spectral_radius > 0 else None,
        )
    
    def collect_problem_metrics(
        self,
        problem_id: str,
        A: sp.csr_matrix,
        grid_size: int,
        params: Dict[str, float],
        ml_learned,
        ml_baseline,
        ml_tuned=None,
        ml_random=None,
    ) -> ProblemMetrics:
        """
        Collect metrics for all solvers on a single problem.
        
        Args:
            problem_id: Unique identifier for this problem
            A: System matrix
            grid_size: Grid dimension
            params: Problem parameters
            ml_learned: Learned AMG solver (trained model)
            ml_baseline: Baseline AMG solver (default PyAMG)
            ml_tuned: Optional tuned AMG solver (hand-tuned params)
            ml_random: Optional random AMG solver (untrained model for ablation)
            
        Returns:
            ProblemMetrics with all solver results
        """
        # Use same RHS for all solvers
        b = np.random.randn(A.shape[0])
        
        metrics = ProblemMetrics(
            problem_id=problem_id,
            grid_size=grid_size,
            n=A.shape[0],
            nnz=A.nnz,
            params=params,
        )
        
        # Learned (trained) solver
        if ml_learned is not None:
            print(f"    Evaluating learned solver...", end=" ", flush=True)
            metrics.learned = self.collect_solver_metrics(A, ml_learned, "learned", b)
            if metrics.learned.pcg:
                rel_res = metrics.learned.pcg.final_residual / metrics.learned.pcg.residual_history[0] if metrics.learned.pcg.residual_history else float('nan')
                print(f"{metrics.learned.pcg.iterations} iters (||r||/||r0||={rel_res:.2e})")
            else:
                print("N/A")
        
        # Random (untrained) solver for ablation
        if ml_random is not None:
            print(f"    Evaluating random solver...", end=" ", flush=True)
            metrics.random = self.collect_solver_metrics(A, ml_random, "random", b)
            if metrics.random.pcg:
                rel_res = metrics.random.pcg.final_residual / metrics.random.pcg.residual_history[0] if metrics.random.pcg.residual_history else float('nan')
                print(f"{metrics.random.pcg.iterations} iters (||r||/||r0||={rel_res:.2e})")
            else:
                print("N/A")
        
        # Baseline solver
        print(f"    Evaluating baseline solver...", end=" ", flush=True)
        metrics.baseline = self.collect_solver_metrics(A, ml_baseline, "baseline", b)
        if metrics.baseline.pcg:
            rel_res = metrics.baseline.pcg.final_residual / metrics.baseline.pcg.residual_history[0] if metrics.baseline.pcg.residual_history else float('nan')
            print(f"{metrics.baseline.pcg.iterations} iters (||r||/||r0||={rel_res:.2e})")
        else:
            print("N/A")
        
        # Tuned solver
        if ml_tuned is not None:
            print(f"    Evaluating tuned solver...", end=" ", flush=True)
            metrics.tuned = self.collect_solver_metrics(A, ml_tuned, "tuned", b)
            if metrics.tuned.pcg:
                rel_res = metrics.tuned.pcg.final_residual / metrics.tuned.pcg.residual_history[0] if metrics.tuned.pcg.residual_history else float('nan')
                print(f"{metrics.tuned.pcg.iterations} iters (||r||/||r0||={rel_res:.2e})")
            else:
                print("N/A")
        
        # Print speedup summary
        learned_speedup = metrics.speedup('learned')
        random_speedup = metrics.speedup('random')
        if learned_speedup and random_speedup:
            print(f"    Speedup: learned={learned_speedup:.2f}x, random={random_speedup:.2f}x vs baseline")
        elif learned_speedup:
            print(f"    Speedup: {learned_speedup:.2f}x")
        elif random_speedup:
            print(f"    Speedup (random): {random_speedup:.2f}x")
        
        return metrics


def save_metrics(metrics: Dict[str, Any], path: Path) -> None:
    """Save metrics to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)


def load_metrics(path: Path) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)
