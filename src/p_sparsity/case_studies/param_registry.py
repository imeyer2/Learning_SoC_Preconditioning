"""
Parameter registry for tracking train/test parameter separation.

Ensures that test problems use parameters not seen during training,
enabling proper generalization evaluation.
"""

import json
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from pathlib import Path
import pickle


@dataclass
class ParameterSet:
    """
    A single set of parameters used to generate a problem.
    
    Includes a hash for quick equality checking and tracking.
    """
    params: Dict[str, float]
    grid_size: int
    _hash: str = field(default="", init=False)
    
    def __post_init__(self):
        self._hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute a stable hash of the parameters."""
        # Sort keys for deterministic ordering
        sorted_items = sorted(self.params.items())
        hash_str = f"g{self.grid_size}_" + "_".join(
            f"{k}:{v:.8f}" for k, v in sorted_items
        )
        return hashlib.md5(hash_str.encode()).hexdigest()[:16]
    
    def __hash__(self):
        return hash(self._hash)
    
    def __eq__(self, other):
        if not isinstance(other, ParameterSet):
            return False
        return self._hash == other._hash
    
    def to_dict(self) -> Dict:
        return {
            'params': self.params,
            'grid_size': self.grid_size,
            'hash': self._hash,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ParameterSet':
        ps = cls(params=d['params'], grid_size=d['grid_size'])
        # Verify hash matches
        if ps._hash != d.get('hash', ps._hash):
            raise ValueError("Parameter hash mismatch - possible corruption")
        return ps


class ParameterRegistry:
    """
    Registry for tracking parameters used in training and testing.
    
    Key features:
    - Records all parameters used during training
    - Validates that test parameters don't overlap with training
    - Supports serialization for reproducibility
    - Can generate non-overlapping test parameters
    """
    
    def __init__(self, name: str = "registry", min_distance: float = 0.02):
        self.name = name
        self.train_params: Set[ParameterSet] = set()
        self.test_params: Dict[str, Set[ParameterSet]] = {}  # variation -> params
        self._param_vectors: List[np.ndarray] = []  # For similarity checking
        self._min_distance: float = min_distance  # Minimum distance in normalized param space
    
    def register_train(self, params: Dict[str, float], grid_size: int) -> ParameterSet:
        """
        Register a parameter set used for training.
        
        Returns the ParameterSet for reference.
        """
        ps = ParameterSet(params=params, grid_size=grid_size)
        self.train_params.add(ps)
        
        # Store vector representation for similarity checking
        vec = self._params_to_vector(params)
        self._param_vectors.append(vec)
        
        return ps
    
    def register_test(
        self, 
        params: Dict[str, float], 
        grid_size: int,
        variation: str
    ) -> ParameterSet:
        """
        Register a parameter set for testing.
        
        Raises ValueError if parameters are too similar to training set.
        """
        ps = ParameterSet(params=params, grid_size=grid_size)
        
        # Check for exact match
        if ps in self.train_params:
            raise ValueError(
                f"Test parameters {params} (hash={ps._hash}) "
                "were used in training!"
            )
        
        # Check for similarity (too close in parameter space)
        if self._is_too_similar(params):
            raise ValueError(
                f"Test parameters {params} are too similar to training set "
                f"(min_distance={self._min_distance})"
            )
        
        if variation not in self.test_params:
            self.test_params[variation] = set()
        self.test_params[variation].add(ps)
        
        return ps
    
    def is_valid_test_params(
        self, 
        params: Dict[str, float],
        grid_size: int
    ) -> Tuple[bool, str]:
        """
        Check if parameters are valid for testing.
        
        Returns (is_valid, reason) tuple.
        """
        ps = ParameterSet(params=params, grid_size=grid_size)
        
        if ps in self.train_params:
            return False, "exact match with training"
        
        if self._is_too_similar(params):
            return False, "too similar to training set"
        
        return True, "ok"
    
    def _params_to_vector(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dict to normalized vector."""
        # Sort keys for consistent ordering
        keys = sorted(params.keys())
        return np.array([params[k] for k in keys])
    
    def _is_too_similar(self, params: Dict[str, float]) -> bool:
        """Check if params are too similar to any training params."""
        if not self._param_vectors:
            return False
        
        vec = self._params_to_vector(params)
        
        # Normalize by parameter range (approximate)
        # This is a simple L2 distance in normalized space
        for train_vec in self._param_vectors:
            if len(vec) != len(train_vec):
                continue  # Skip if different param sets
            
            # Compute normalized distance
            diff = vec - train_vec
            # Rough normalization assuming params in [0, 10] range
            normalized_diff = diff / 10.0
            distance = np.linalg.norm(normalized_diff)
            
            if distance < self._min_distance:
                return True
        
        return False
    
    def generate_test_params(
        self,
        param_ranges: Dict[str, 'ParameterRange'],
        num_samples: int,
        rng: np.random.Generator,
        max_attempts: int = 1000
    ) -> List[Dict[str, float]]:
        """
        Generate test parameters that don't overlap with training.
        
        Args:
            param_ranges: Parameter ranges to sample from
            num_samples: Number of parameter sets to generate
            rng: Random number generator
            max_attempts: Maximum attempts per sample
            
        Returns:
            List of valid test parameter dictionaries
        """
        from .config import ParameterRange
        
        valid_params = []
        attempts = 0
        
        while len(valid_params) < num_samples and attempts < max_attempts * num_samples:
            attempts += 1
            
            # Sample parameters
            params = {}
            for name, pr in param_ranges.items():
                if pr.fixed is not None:
                    params[name] = pr.fixed
                elif pr.values is not None:
                    params[name] = float(rng.choice(pr.values))
                else:
                    if pr.log_scale:
                        log_val = rng.uniform(np.log(pr.min_val), np.log(pr.max_val))
                        params[name] = float(np.exp(log_val))
                    else:
                        params[name] = float(rng.uniform(pr.min_val, pr.max_val))
            
            # Check validity
            is_valid, _ = self.is_valid_test_params(params, grid_size=0)
            if is_valid:
                valid_params.append(params)
        
        if len(valid_params) < num_samples:
            raise RuntimeError(
                f"Could only generate {len(valid_params)}/{num_samples} "
                f"valid test parameters after {attempts} attempts. "
                "Consider relaxing min_distance or expanding parameter ranges."
            )
        
        return valid_params
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics of the registry."""
        return {
            'name': self.name,
            'num_train_params': len(self.train_params),
            'test_variations': {
                var: len(params) 
                for var, params in self.test_params.items()
            },
            'min_distance': self._min_distance,
        }
    
    def save(self, path: Path) -> None:
        """Save registry to file."""
        data = {
            'name': self.name,
            'train_params': [ps.to_dict() for ps in self.train_params],
            'test_params': {
                var: [ps.to_dict() for ps in params]
                for var, params in self.test_params.items()
            },
            'min_distance': self._min_distance,
        }
        
        path = Path(path)
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: Path) -> 'ParameterRegistry':
        """Load registry from file."""
        path = Path(path)
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        
        registry = cls(name=data['name'])
        registry._min_distance = data.get('min_distance', 0.05)
        
        # Restore train params
        for ps_dict in data['train_params']:
            ps = ParameterSet.from_dict(ps_dict)
            registry.train_params.add(ps)
            registry._param_vectors.append(
                registry._params_to_vector(ps.params)
            )
        
        # Restore test params
        for var, ps_list in data.get('test_params', {}).items():
            registry.test_params[var] = {
                ParameterSet.from_dict(ps_dict) for ps_dict in ps_list
            }
        
        return registry
    
    def __repr__(self):
        return (
            f"ParameterRegistry(name='{self.name}', "
            f"train={len(self.train_params)}, "
            f"test_variations={list(self.test_params.keys())})"
        )
