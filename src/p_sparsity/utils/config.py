"""
Configuration loading and management.
"""

import os
from pathlib import Path
from typing import Dict, Any, Union
import yaml
from omegaconf import OmegaConf, DictConfig


def load_config(path: Union[str, Path]) -> DictConfig:
    """
    Load YAML configuration file.
    
    Args:
        path: Path to YAML file
        
    Returns:
        config: OmegaConf DictConfig
    """
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return OmegaConf.create(config_dict)


def merge_configs(*configs: DictConfig) -> DictConfig:
    """
    Merge multiple configurations.
    
    Later configs override earlier ones.
    
    Args:
        *configs: Variable number of DictConfig objects
        
    Returns:
        merged: Merged configuration
    """
    return OmegaConf.merge(*configs)


def save_config(config: Union[Dict, DictConfig], path: Union[str, Path]):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        path: Output path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
    else:
        config_dict = config
    
    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def config_to_dict(config: DictConfig) -> Dict[str, Any]:
    """Convert OmegaConf to dict."""
    return OmegaConf.to_container(config, resolve=True)
