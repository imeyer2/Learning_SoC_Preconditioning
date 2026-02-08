"""
Case study runner - orchestrates training and evaluation.

Main entry point for running case study experiments.
"""

import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import torch

from .config import CaseStudyConfig, VariationConfig, ProblemType
from .param_registry import ParameterRegistry
from .generators import create_generator, UnifiedProblemGenerator, ProblemInstance
from .metrics import MetricsCollector, ProblemMetrics, ScalingMetrics, save_metrics


@dataclass
class VariationResults:
    """Results from running a single variation."""
    variation_name: str
    train_params: List[Dict[str, float]]
    test_problems: List[ProblemMetrics]
    scaling_metrics: Optional[ScalingMetrics] = None
    train_time: float = 0.0
    eval_time: float = 0.0
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        speedups = [p.speedup() for p in self.test_problems if p.speedup() is not None]
        
        # Also calculate random speedup if available (4-way ablation)
        random_speedups = [p.speedup('random') for p in self.test_problems if p.speedup('random') is not None]
        
        result = {
            'variation': self.variation_name,
            'num_train_samples': len(self.train_params),
            'num_test_problems': len(self.test_problems),
            'train_time': self.train_time,
            'eval_time': self.eval_time,
            'avg_speedup': np.mean(speedups) if speedups else None,
            'std_speedup': np.std(speedups) if speedups else None,
            'min_speedup': np.min(speedups) if speedups else None,
            'max_speedup': np.max(speedups) if speedups else None,
        }
        
        # Add random comparison if available (ablation study)
        if random_speedups:
            result['avg_random_speedup'] = np.mean(random_speedups)
            result['std_random_speedup'] = np.std(random_speedups)
            # Learned vs random comparison
            learned_vs_random = [p.speedup_learned_vs_random() for p in self.test_problems 
                                 if p.speedup_learned_vs_random() is not None]
            if learned_vs_random:
                result['avg_learned_vs_random'] = np.mean(learned_vs_random)
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            'variation': self.variation_name,
            'summary': self.summary(),
            'train_params': self.train_params,
            'test_problems': [p.to_dict() for p in self.test_problems],
        }
        if self.scaling_metrics:
            d['scaling'] = self.scaling_metrics.to_dict()
        return d


class CaseStudyRunner:
    """
    Orchestrates case study experiments.
    
    Workflow:
    1. Load configuration
    2. Generate training data (track parameters)
    3. Train model
    4. Generate test data (exclude training parameters)
    5. Evaluate and collect metrics
    6. Save results
    """
    
    def __init__(
        self,
        config: CaseStudyConfig,
        output_dir: Optional[Path] = None,
        device: str = 'cpu',
        verbose: bool = True,
    ):
        self.config = config
        self.output_dir = Path(output_dir or config.output_dir) / config.name
        self.device = device
        self.verbose = verbose
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize generator
        self.generator = create_generator(config.problem.problem_type)
        
        # Results storage
        self.results: Dict[str, VariationResults] = {}
    
    def run_variation(
        self,
        variation_name: str,
        init_checkpoint: Optional[str] = None,
        skip_training: bool = False,
    ) -> VariationResults:
        """
        Run a single variation of the case study.
        
        Args:
            variation_name: Name of variation (A, B, C, ...)
            init_checkpoint: Optional checkpoint to initialize model weights from
                           (training still happens on generated data unless skip_training=True)
            skip_training: If True, skip training entirely and only run evaluation
                          (requires init_checkpoint to be set)
            
        Returns:
            VariationResults with all metrics
        """
        if variation_name not in self.config.variations:
            raise ValueError(f"Unknown variation: {variation_name}")
        
        var_config = self.config.variations[variation_name]
        
        if skip_training and not init_checkpoint and not var_config.use_random_init:
            raise ValueError("skip_training=True requires init_checkpoint or use_random_init=True")
        
        var_output = self.output_dir / f"variation_{variation_name}"
        var_output.mkdir(parents=True, exist_ok=True)
        
        self._log(f"\n{'='*60}")
        self._log(f"Running Variation {variation_name}: {var_config.description}")
        if var_config.use_random_init:
            self._log("  [RANDOM BASELINE - No training, random weights]")
        self._log(f"{'='*60}")
        
        # Initialize parameter registry
        registry = ParameterRegistry(name=f"{self.config.name}_{variation_name}")
        
        # Check if using random initialization (ablation study)
        use_random = var_config.use_random_init
        
        # Generate training data (skip if skip_training or using random init)
        train_data = []
        train_params = []
        if not skip_training and not use_random:
            self._log("\n[1/4] Generating training data...")
            train_data, train_params = self._generate_train_data(
                var_config, registry
            )
            self._log(f"  Generated {len(train_data)} training samples")
        elif use_random:
            self._log("\n[1/4] Skipping training data (random baseline mode)")
        else:
            self._log("\n[1/4] Skipping training data generation (skip_training=True)")
        
        # Save registry
        registry.save(var_output / "param_registry.json")
        
        # Train model, load from checkpoint, or use random init
        train_time = 0.0
        if use_random:
            self._log("\n[2/4] Creating randomly initialized model (no training)...")
            model = self._create_random_model(var_config)
            checkpoint_path = None
            self._log(f"  Random model created (seed from config)")
        elif skip_training:
            self._log("\n[2/4] Loading model from checkpoint (skipping training)...")
            self._log(f"  Loading from: {init_checkpoint}")
            model = self._load_model_from_checkpoint(init_checkpoint, var_config)
            checkpoint_path = init_checkpoint
            self._log(f"  Model loaded successfully")
        else:
            self._log("\n[2/4] Training model...")
            if init_checkpoint:
                self._log(f"  Initializing from: {init_checkpoint}")
            train_start = time.time()
            model, checkpoint_path = self._train_model(
                train_data, var_config, var_output, init_checkpoint=init_checkpoint
            )
            train_time = time.time() - train_start
            self._log(f"  Training completed in {train_time:.1f}s")
        
        # Generate test data
        self._log("\n[3/4] Generating test data...")
        test_data, test_params = self._generate_test_data(
            var_config, registry
        )
        self._log(f"  Generated {len(test_data)} test problems")
        
        # Load trained model for comparison if this is an ablation study
        trained_model = None
        if use_random and var_config.compare_with_trained:
            self._log(f"\n[3.5/4] Loading trained model for comparison...")
            self._log(f"  Loading from: {var_config.compare_with_trained}")
            trained_model = self._load_model_from_checkpoint(
                var_config.compare_with_trained, var_config
            )
            self._log(f"  Trained model loaded for 4-way comparison")
        
        # Evaluate
        self._log("\n[4/4] Evaluating...")
        eval_start = time.time()
        test_metrics, scaling_metrics = self._evaluate(
            model, test_data, var_config, trained_model=trained_model
        )
        eval_time = time.time() - eval_start
        self._log(f"  Evaluation completed in {eval_time:.1f}s")
        
        # Create results
        results = VariationResults(
            variation_name=variation_name,
            train_params=train_params,
            test_problems=test_metrics,
            scaling_metrics=scaling_metrics,
            train_time=train_time,
            eval_time=eval_time,
        )
        
        # Save results
        save_metrics(results.to_dict(), var_output / "results.json")
        
        # Generate plots
        self._log("\n[Plotting] Generating result plots...")
        self._generate_plots(results, var_output)
        
        # Print summary
        summary = results.summary()
        self._log(f"\n--- Variation {variation_name} Summary ---")
        
        # Check if this is a 4-way ablation (trained vs random vs baseline vs tuned)
        if 'avg_random_speedup' in summary and summary['avg_random_speedup'] is not None:
            self._log("  4-way ablation study results:")
            self._log(f"    Trained vs baseline:  {summary['avg_speedup']:.2f}x" if summary['avg_speedup'] else "    Trained vs baseline: N/A")
            self._log(f"    Random vs baseline:   {summary['avg_random_speedup']:.2f}x" if summary['avg_random_speedup'] else "    Random vs baseline:  N/A")
            if 'avg_learned_vs_random' in summary and summary['avg_learned_vs_random'] is not None:
                self._log(f"    Trained vs random:    {summary['avg_learned_vs_random']:.2f}x")
        else:
            self._log(f"  Avg speedup: {summary['avg_speedup']:.2f}x" if summary['avg_speedup'] else "  Avg speedup: N/A")
        
        self._log(f"  Train time: {summary['train_time']:.1f}s")
        self._log(f"  Eval time: {summary['eval_time']:.1f}s")
        
        self.results[variation_name] = results
        return results
    
    def run_all(self, init_checkpoint: Optional[str] = None) -> Dict[str, VariationResults]:
        """
        Run all variations in the case study.
        
        Args:
            init_checkpoint: If provided, initialize models from this checkpoint
            
        Returns:
            Dictionary of variation name -> results
        """
        for var_name in self.config.variations:
            self.results[var_name] = self.run_variation(
                var_name, 
                init_checkpoint=init_checkpoint,
            )
        
        # Save combined results
        combined = {
            'config': {
                'name': self.config.name,
                'problem_type': self.config.problem.problem_type.value,
            },
            'variations': {
                name: results.to_dict() 
                for name, results in self.results.items()
            }
        }
        save_metrics(combined, self.output_dir / "combined_results.json")
        
        return self.results
    
    def _generate_train_data(
        self,
        var_config: VariationConfig,
        registry: ParameterRegistry,
    ) -> Tuple[List[ProblemInstance], List[Dict[str, float]]]:
        """Generate training data and register parameters."""
        rng = np.random.default_rng(var_config.train.seed)
        
        data = []
        params_list = []
        
        for i in range(var_config.train.num_samples):
            # Sample parameters
            params = self.generator.sample_params(
                rng, 
                self.config.problem.param_ranges
            )
            
            # Register for tracking
            registry.register_train(params, var_config.train.grid_size)
            
            # Generate problem
            instance = self.generator.generate(var_config.train.grid_size, params)
            data.append(instance)
            params_list.append(params)
            
            if self.verbose and (i + 1) % 10 == 0:
                print(f"    Generated {i+1}/{var_config.train.num_samples} samples", flush=True)
        
        return data, params_list
    
    def _generate_test_data(
        self,
        var_config: VariationConfig,
        registry: ParameterRegistry,
    ) -> Tuple[List[ProblemInstance], List[Dict[str, float]]]:
        """Generate test data, ensuring no overlap with training."""
        rng = np.random.default_rng(var_config.test.seed)
        
        data = []
        params_list = []
        
        for grid_size in var_config.test.grid_sizes:
            self._log(f"    Generating for grid_size={grid_size}...")
            
            # Generate non-overlapping parameters
            test_params = registry.generate_test_params(
                self.config.problem.param_ranges,
                var_config.test.num_samples_per_size,
                rng,
            )
            
            for i, params in enumerate(test_params):
                # Register test params
                registry.register_test(params, grid_size, var_config.name)
                
                # Generate problem
                instance = self.generator.generate(grid_size, params)
                data.append(instance)
                params_list.append(params)
        
        return data, params_list
    
    def _train_model(
        self,
        train_data: List[ProblemInstance],
        var_config: VariationConfig,
        output_dir: Path,
        init_checkpoint: Optional[str] = None,
    ) -> Tuple[Any, Path]:
        """
        Train the model on the given data.
        
        Args:
            train_data: List of problem instances to train on
            var_config: Variation configuration
            output_dir: Where to save checkpoints
            init_checkpoint: Optional checkpoint to initialize weights from
        
        Returns:
            Trained model and checkpoint path.
        """
        # Convert ProblemInstances to training format
        from torch_geometric.utils import from_scipy_sparse_matrix
        from ..data import TrainSample, node_features_for_policy
        from ..data.dataset import build_row_groups
        from ..models import build_policy_from_config
        from ..rl import ReinforceTrainer
        from ..utils import load_config, ExperimentTracker
        
        # Set seed for reproducible model initialization
        # This ensures the same random weights when running with the same seed
        seed = var_config.train.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if self.verbose:
            print(f"    Set random seed to {seed} for reproducible initialization")
        
        # Load model config
        model_config_path = var_config.train.model_config or "configs/model/gat_default.yaml"
        model_cfg = load_config(model_config_path)
        
        # Build model
        model = build_policy_from_config(model_cfg)
        
        # Optionally initialize from existing checkpoint
        if init_checkpoint:
            checkpoint = torch.load(init_checkpoint, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            if self.verbose:
                print(f"    Initialized model weights from {init_checkpoint}")
        
        model.to(self.device)
        
        # Feature config for node features
        feature_config = {
            "normalize_weights": True,
            "num_smooth_vecs": 4,
            "smooth_steps": 10,
        }
        
        # Convert ProblemInstances to TrainSamples
        train_samples = []
        for instance in train_data:
            A = instance.A
            grid_size = instance.grid_size
            
            # Get coordinates (generate if not provided)
            if instance.coords is not None:
                coords = instance.coords
            else:
                # Generate 2D grid coordinates
                n = int(np.sqrt(A.shape[0]))
                x_vals = np.linspace(0, 1, n)
                y_vals = np.linspace(0, 1, n)
                xx, yy = np.meshgrid(x_vals, y_vals)
                coords = np.stack([xx.ravel(), yy.ravel()], axis=1)
            
            # Build graph from sparse matrix
            ei, ew = from_scipy_sparse_matrix(A)
            ew = ew.float()
            
            # Normalize edge weights
            max_w = ew.abs().max()
            if max_w > 0:
                ew = ew / max_w
            
            # Build node features
            x = node_features_for_policy(A, coords, feature_config)
            
            # Build row groups for edge sampling
            row_groups = build_row_groups(ei, num_nodes=A.shape[0])
            
            sample = TrainSample(
                A=A,
                grid_n=grid_size,
                coords=coords,
                edge_index=ei,
                edge_weight=ew,
                x=x,
                row_groups=row_groups,
                metadata=instance.metadata,
            )
            train_samples.append(sample)
        
        # Setup experiment tracking
        experiment = ExperimentTracker(str(output_dir))
        
        # Setup TensorBoard logger
        from ..utils import setup_tensorboard
        tb_logger, tb_log_dir = setup_tensorboard(
            experiment_name=f"case_study_{var_config.name}",
            base_dir=str(output_dir)
        )
        
        # Load full training config and override specific values
        train_cfg = load_config("configs/training/reinforce_default.yaml")
        train_cfg.epochs = var_config.train.epochs
        train_cfg.batch_size = var_config.train.batch_size
        train_cfg.optimizer.lr = var_config.train.learning_rate
        train_cfg.device = self.device
        train_cfg.experiment.output_dir = str(output_dir)
        
        # Override entropy and temperature from case study config if specified
        if hasattr(var_config.train, 'entropy_coef') and var_config.train.entropy_coef is not None:
            train_cfg.entropy_coef = var_config.train.entropy_coef
        if hasattr(var_config.train, 'temperature') and var_config.train.temperature is not None:
            temp_cfg = var_config.train.temperature
            if isinstance(temp_cfg, dict):
                if 'initial' in temp_cfg:
                    train_cfg.temperature.initial = temp_cfg['initial']
                if 'anneal_factor' in temp_cfg:
                    train_cfg.temperature.anneal_factor = temp_cfg['anneal_factor']
                if 'min_temperature' in temp_cfg:
                    train_cfg.temperature.min_temperature = temp_cfg['min_temperature']
                if 'anneal' in temp_cfg:
                    train_cfg.temperature.anneal = temp_cfg['anneal']
        
        # Train
        trainer = ReinforceTrainer(
            model=model,
            train_data=train_samples,
            config=train_cfg,
            experiment_tracker=experiment,
            tb_logger=tb_logger,
        )
        
        trainer.train()
        
        checkpoint_path = output_dir / "checkpoints" / "best_model.pt"
        return model, checkpoint_path
    
    def _load_model_from_checkpoint(
        self,
        checkpoint_path: str,
        var_config: VariationConfig,
    ) -> Any:
        """
        Load a trained model from checkpoint using variation config for architecture.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            var_config: Variation configuration (for model architecture)
            
        Returns:
            Loaded model ready for evaluation
        """
        from ..models import build_policy_from_config
        from ..utils import load_config
        
        # Load model config from variation config
        model_config_path = var_config.train.model_config or "configs/model/gat_default.yaml"
        model_cfg = load_config(model_config_path)
        
        # Build model with same architecture
        model = build_policy_from_config(model_cfg)
        
        # Load weights (use weights_only=False for trusted local checkpoints)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def _create_random_model(
        self,
        var_config: VariationConfig,
    ) -> Any:
        """
        Create a randomly initialized model (no training) for ablation studies.
        
        This is useful for comparing trained models against random baselines
        to demonstrate that training actually improves performance.
        
        Args:
            var_config: Variation configuration (for model architecture)
            
        Returns:
            Randomly initialized model ready for evaluation
        """
        from ..models import build_policy_from_config
        from ..utils import load_config
        
        # Set seed for reproducibility of random init
        seed = var_config.train.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if self.verbose:
            print(f"    Set random seed to {seed} for reproducible random initialization")
        
        # Load model config from variation config
        model_config_path = var_config.train.model_config or "configs/model/gat_default.yaml"
        model_cfg = load_config(model_config_path)
        
        # Build model with random initialization
        model = build_policy_from_config(model_cfg)
        
        model.to(self.device)
        model.eval()
        
        if self.verbose:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"    Created random model with {total_params:,} parameters")
        
        return model
    
    def _load_model(self, checkpoint_path: str) -> Any:
        """Load a trained model from checkpoint."""
        from ..models import build_policy_from_config
        
        # Use weights_only=False for trusted local checkpoints (PyTorch 2.6+ default changed)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model_config = checkpoint.get('model_config', {})
        
        from omegaconf import OmegaConf
        if isinstance(model_config, dict):
            model_config = OmegaConf.create(model_config)
        
        model = build_policy_from_config(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def _evaluate(
        self,
        model,
        test_data: List[ProblemInstance],
        var_config: VariationConfig,
        trained_model=None,
    ) -> Tuple[List[ProblemMetrics], Optional[ScalingMetrics]]:
        """
        Evaluate the model on test data.
        
        Args:
            model: Primary model to evaluate (could be trained or random)
            test_data: List of problem instances
            var_config: Variation configuration
            trained_model: If provided, also evaluate this trained model for comparison
                          (used in ablation studies where model is random)
        
        Returns list of problem metrics and optional scaling metrics.
        """
        from ..pyamg_interface import build_pyamg_solver
        
        # Setup metrics collector
        collector = MetricsCollector(
            collect_pcg=var_config.test.collect_iterations,
            collect_vcycle=var_config.test.collect_energy_history,
            collect_spectral=var_config.test.collect_spectral,
        )
        
        # Track scaling if this is a scaling study
        scaling = None
        if var_config.test.is_scaling_study:
            scaling = ScalingMetrics(grid_sizes=var_config.test.grid_sizes)
        
        all_metrics = []
        
        # Use consistent max_levels for fair comparison
        # Get from config or use defaults
        max_levels = getattr(var_config.train, 'max_levels', None)
        max_coarse = getattr(var_config.train, 'max_coarse', 10)
        
        # -1 or None means use PyAMG default (10 levels)
        # This provides enough levels for most problems while avoiding excessive depth
        if max_levels is None or max_levels == -1:
            max_levels = 10  # PyAMG default
        
        # Determine if this is an ablation study (random model with trained comparison)
        is_ablation = var_config.use_random_init and trained_model is not None
        
        for i, instance in enumerate(test_data):
            problem_id = f"test_{i:04d}"
            self._log(f"  Problem {i+1}/{len(test_data)}: n={instance.n}, grid={instance.grid_size}")
            
            # Build solvers
            ml_learned = None
            ml_random = None
            
            if is_ablation:
                # Ablation study: compare trained, random, baseline, tuned
                # model = random model, trained_model = trained model
                C_random, B_random = self._build_learned_C(model, instance)
                ml_random = build_pyamg_solver(
                    instance.A, C_random, B_random, 
                    max_levels=max_levels, max_coarse=max_coarse
                )
                
                C_trained, B_trained = self._build_learned_C(trained_model, instance)
                ml_learned = build_pyamg_solver(
                    instance.A, C_trained, B_trained, 
                    max_levels=max_levels, max_coarse=max_coarse
                )
            else:
                # Normal evaluation: model is the "learned" solver (trained or random)
                C_learned, B_learned = self._build_learned_C(model, instance)
                ml_learned = build_pyamg_solver(
                    instance.A, C_learned, B_learned, 
                    max_levels=max_levels, max_coarse=max_coarse
                )
            
            # Build TRUE baseline solver (default PyAMG, no learned components)
            # This is the fair comparison: default PyAMG vs our approach
            import pyamg
            ml_baseline = pyamg.smoothed_aggregation_solver(
                instance.A, 
                B=np.ones((instance.n, 1)),  # Default constant B
                max_levels=max_levels,
                max_coarse=max_coarse,
            )
            
            # Build tuned baseline (hand-tuned parameters, still no learned components)
            # This tests if learned approach beats expert-tuned heuristics
            ml_tuned = pyamg.smoothed_aggregation_solver(
                instance.A,
                strength=('symmetric', {'theta': 0.25}),
                B=np.ones((instance.n, 1)),  # Default constant B
                max_levels=max_levels,
                max_coarse=max_coarse,
            )
            
            # Collect metrics
            metrics = collector.collect_problem_metrics(
                problem_id=problem_id,
                A=instance.A,
                grid_size=instance.grid_size,
                params=instance.params,
                ml_learned=ml_learned,
                ml_baseline=ml_baseline,
                ml_tuned=ml_tuned,
                ml_random=ml_random,
            )
            
            all_metrics.append(metrics)
            
            if scaling:
                scaling.add_problem(instance.grid_size, metrics)
        
        return all_metrics, scaling
    
    def _build_learned_C(
        self,
        model,
        instance: ProblemInstance,
    ) -> Tuple:
        """Build learned strength matrix from model."""
        # Import here to avoid circular dependency
        from ..pyamg_interface import build_C_from_model
        
        k_per_row = model.config.get('edges_per_row', 3)
        use_learned_k = model.config.get('learn_k', False)
        
        C_learned, B_extra = build_C_from_model(
            instance.A,
            instance.grid_size,
            model,
            k_per_row,
            device=self.device,
            use_learned_k=use_learned_k,
        )
        
        # Prepare B
        if B_extra is not None:
            B = np.column_stack([np.ones(instance.n), B_extra])
        else:
            B = None
        
        return C_learned, B
    
    def _generate_plots(
        self,
        results: VariationResults,
        output_dir: Path,
    ) -> None:
        """
        Generate and save visualization plots for the results.
        
        Args:
            results: Evaluation results
            output_dir: Directory to save plots
        """
        from .visualization import (
            plot_iteration_histogram,
            plot_residual_curves,
            plot_energy_decay,
            plot_scaling_curves,
        )
        import matplotlib.pyplot as plt
        
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        results_dict = results.to_dict()
        test_problems = results_dict.get('test_problems', [])
        
        # Iteration histogram
        if test_problems:
            try:
                plot_iteration_histogram(
                    test_problems,
                    title=f"{results.variation_name}: PCG Iteration Distribution",
                    save_path=plots_dir / "iteration_histogram.png"
                )
                plt.close()
                self._log(f"  Saved: iteration_histogram.png")
            except Exception as e:
                self._log(f"  Warning: Could not generate iteration histogram: {e}")
        
        # Residual curves (subset of problems)
        if test_problems:
            try:
                plot_residual_curves(
                    test_problems,
                    title=f"{results.variation_name}: Residual Convergence",
                    save_path=plots_dir / "residual_curves.png",
                    max_problems=5,
                )
                plt.close()
                self._log(f"  Saved: residual_curves.png")
            except Exception as e:
                self._log(f"  Warning: Could not generate residual curves: {e}")
        
        # Energy decay curves
        if test_problems:
            try:
                plot_energy_decay(
                    test_problems,
                    title=f"{results.variation_name}: V-cycle Energy Decay",
                    save_path=plots_dir / "energy_decay.png",
                    max_problems=5,
                )
                plt.close()
                self._log(f"  Saved: energy_decay.png")
            except Exception as e:
                self._log(f"  Warning: Could not generate energy decay plot: {e}")
        
        # Scaling curves (if scaling study)
        scaling_data = results_dict.get('scaling_metrics')
        if scaling_data and scaling_data.get('summary'):
            try:
                plot_scaling_curves(
                    scaling_data,
                    title=f"{results.variation_name}: Scaling Study",
                    save_path=plots_dir / "scaling_curves.png"
                )
                plt.close()
                self._log(f"  Saved: scaling_curves.png")
            except Exception as e:
                self._log(f"  Warning: Could not generate scaling curves: {e}")
    
    def _log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(message, flush=True)


def run_case_study(
    config_path: str,
    variation: Optional[str] = None,
    output_dir: Optional[str] = None,
    checkpoint: Optional[str] = None,
    device: str = 'cpu',
) -> Dict[str, VariationResults]:
    """
    Convenience function to run a case study from config file.
    
    Args:
        config_path: Path to case study YAML config
        variation: Specific variation to run (None = all)
        output_dir: Output directory (uses config default if None)
        checkpoint: Existing checkpoint to use
        device: Device for training/inference
        
    Returns:
        Dictionary of variation name -> results
    """
    config = CaseStudyConfig.from_yaml(config_path)
    
    runner = CaseStudyRunner(
        config=config,
        output_dir=Path(output_dir) if output_dir else None,
        device=device,
    )
    
    if variation:
        results = {variation: runner.run_variation(variation, checkpoint)}
    else:
        results = runner.run_all(checkpoint)
    
    return results
