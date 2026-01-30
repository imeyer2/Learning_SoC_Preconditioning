"""
Example: Using the new modular P-Sparsity structure.

This demonstrates how to use the new modules without the full training pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from p_sparsity.utils import load_config
from p_sparsity.data import get_generator, make_dataset, list_generators
from p_sparsity.models import build_policy_from_config
import torch


def example_1_list_generators():
    """Example 1: List available problem generators."""
    print("=" * 60)
    print("Example 1: Available Problem Generators")
    print("=" * 60)
    
    generators = list_generators()
    print(f"\nRegistered generators: {generators}")
    print("\nYou can add new generators by:")
    print("  1. Creating a class that inherits from ProblemGenerator")
    print("  2. Decorating it with @register_generator('name')")
    print("  3. Importing it in data/generators/__init__.py")


def example_2_generate_single_problem():
    """Example 2: Generate a single problem."""
    print("\n" + "=" * 60)
    print("Example 2: Generate a Single Anisotropic Problem")
    print("=" * 60)
    
    # Load config
    config_path = Path(__file__).parent.parent / "configs/data/anisotropic_default.yaml"
    config = load_config(config_path)
    
    # Get generator
    generator = get_generator("anisotropic", config.anisotropic)
    
    # Generate problem
    A = generator.generate(grid_size=16, epsilon=0.001, theta=0.0)
    coords = generator.get_coordinates(A)
    metadata = generator.get_metadata(A, epsilon=0.001, theta=0.0)
    
    print(f"\nGenerated matrix:")
    print(f"  Shape: {A.shape}")
    print(f"  NNZ: {A.nnz}")
    print(f"  Coordinates shape: {coords.shape}")
    print(f"  Metadata: {metadata}")


def example_3_build_dataset():
    """Example 3: Build a complete dataset."""
    print("\n" + "=" * 60)
    print("Example 3: Build Training Dataset")
    print("=" * 60)
    
    # Load config
    config_path = Path(__file__).parent.parent / "configs/data/anisotropic_default.yaml"
    config = load_config(config_path)
    
    # Build dataset
    dataset = make_dataset(
        problem_type="anisotropic",
        num_samples=5,
        grid_size=16,
        config=config,
        seed=42,
    )
    
    print(f"\nDataset created:")
    print(f"  Number of samples: {len(dataset)}")
    print(f"\nFirst sample:")
    sample = dataset[0]
    print(f"  Matrix shape: {sample.A.shape}")
    print(f"  Node features shape: {sample.x.shape}")
    print(f"  Edge index shape: {sample.edge_index.shape}")
    print(f"  Edge weight shape: {sample.edge_weight.shape}")
    print(f"  Number of row groups: {len(sample.row_groups)}")
    print(f"  Metadata: {sample.metadata}")


def example_4_build_model():
    """Example 4: Build model from config."""
    print("\n" + "=" * 60)
    print("Example 4: Build Model from Config")
    print("=" * 60)
    
    # Load config
    config_path = Path(__file__).parent.parent / "configs/model/gat_default.yaml"
    config = load_config(config_path)
    
    # Build model
    model = build_policy_from_config(config)
    
    num_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nModel created:")
    print(f"  Backbone: {config.backbone}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Learn B: {config.learn_B}")
    print(f"  Total parameters: {num_params:,}")
    print(f"\nModel structure:")
    print(model)


def example_5_forward_pass():
    """Example 5: Run model forward pass."""
    print("\n" + "=" * 60)
    print("Example 5: Model Forward Pass")
    print("=" * 60)
    
    # Load configs
    model_config_path = Path(__file__).parent.parent / "configs/model/gat_default.yaml"
    data_config_path = Path(__file__).parent.parent / "configs/data/anisotropic_default.yaml"
    
    model_config = load_config(model_config_path)
    data_config = load_config(data_config_path)
    
    # Build model
    model = build_policy_from_config(model_config)
    model.eval()
    
    # Get a sample
    dataset = make_dataset(
        problem_type="anisotropic",
        num_samples=1,
        grid_size=16,
        config=data_config,
        seed=42,
    )
    sample = dataset[0]
    
    # Forward pass
    with torch.no_grad():
        logits, B_extra = model(
            x=sample.x,
            edge_index=sample.edge_index,
            edge_weight=sample.edge_weight,
        )
    
    print(f"\nForward pass results:")
    print(f"  Input nodes: {sample.x.shape[0]}")
    print(f"  Input edges: {sample.edge_index.shape[1]}")
    print(f"  Edge logits shape: {logits.shape}")
    print(f"  Edge logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    if B_extra is not None:
        print(f"  B candidates shape: {B_extra.shape}")
        print(f"  B candidates range: [{B_extra.min():.3f}, {B_extra.max():.3f}]")


def example_6_config_system():
    """Example 6: Configuration system."""
    print("\n" + "=" * 60)
    print("Example 6: Configuration System")
    print("=" * 60)
    
    from p_sparsity.utils import merge_configs
    
    # Load multiple configs
    model_config = load_config(Path(__file__).parent.parent / "configs/model/gat_default.yaml")
    train_config = load_config(Path(__file__).parent.parent / "configs/training/reinforce_default.yaml")
    
    print("\nModel config excerpt:")
    print(f"  Backbone: {model_config.backbone}")
    print(f"  Hidden dim: {model_config.hidden_dim}")
    
    print("\nTraining config excerpt:")
    print(f"  Algorithm: {train_config.algorithm}")
    print(f"  Epochs: {train_config.epochs}")
    print(f"  Learning rate: {train_config.optimizer.lr}")
    
    # You can also merge configs
    print("\nConfigs can be merged and overridden:")
    print("  merged = merge_configs(base_config, custom_overrides)")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("P-Sparsity Module Examples")
    print("=" * 60)
    print("\nThis demonstrates the new modular structure.")
    print("Each example shows a different capability.\n")
    
    try:
        example_1_list_generators()
        example_2_generate_single_problem()
        example_3_build_dataset()
        example_4_build_model()
        example_5_forward_pass()
        example_6_config_system()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully! ✓")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Check out MIGRATION.md for remaining work")
        print("  2. Use these modules to build your training pipeline")
        print("  3. Customize configs in configs/ directory")
        print("  4. Add new problem generators using the registry pattern")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
