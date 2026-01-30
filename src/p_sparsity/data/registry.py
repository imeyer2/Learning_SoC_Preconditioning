"""
Registry for problem generators.

Allows easy registration and retrieval of problem generators.
"""

from typing import Dict, Type, List
from .base import ProblemGenerator

# Global registry
_GENERATOR_REGISTRY: Dict[str, Type[ProblemGenerator]] = {}


def register_generator(name: str):
    """
    Decorator to register a problem generator.
    
    Usage:
        @register_generator("anisotropic")
        class AnisotropicGenerator(ProblemGenerator):
            ...
    
    Args:
        name: Name to register the generator under
    """
    def decorator(cls: Type[ProblemGenerator]):
        if name in _GENERATOR_REGISTRY:
            raise ValueError(f"Generator '{name}' is already registered")
        if not issubclass(cls, ProblemGenerator):
            raise TypeError(f"Class {cls} must inherit from ProblemGenerator")
        _GENERATOR_REGISTRY[name] = cls
        return cls
    return decorator


def get_generator(name: str, config: Dict) -> ProblemGenerator:
    """
    Get a problem generator by name.
    
    Args:
        name: Name of the generator
        config: Configuration dictionary
        
    Returns:
        generator: Instantiated generator
        
    Raises:
        KeyError: If generator not found
    """
    if name not in _GENERATOR_REGISTRY:
        raise KeyError(
            f"Generator '{name}' not found. Available generators: {list_generators()}"
        )
    generator_class = _GENERATOR_REGISTRY[name]
    return generator_class(config)


def list_generators() -> List[str]:
    """
    List all registered generators.
    
    Returns:
        names: List of generator names
    """
    return list(_GENERATOR_REGISTRY.keys())
