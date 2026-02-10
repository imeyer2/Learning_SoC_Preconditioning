"""Tests for adaptive entropy coefficient in ReinforceTrainer."""

import pytest
from omegaconf import OmegaConf


class MockAdaptiveEntropyTrainer:
    """
    Minimal stand-in that replicates the adaptive entropy logic
    from ReinforceTrainer without requiring the full training stack.
    """

    def __init__(self, config):
        self.entropy_coef = getattr(config, 'entropy_coef', 0.01)

        adaptive_cfg = getattr(config, 'adaptive_entropy', None)
        if adaptive_cfg is not None and getattr(adaptive_cfg, 'enabled', False):
            self.adaptive_entropy = True
            self.entropy_target = getattr(adaptive_cfg, 'target_entropy', 1.0)
            self.entropy_adapt_rate = getattr(adaptive_cfg, 'adaptation_rate', 0.001)
            self.entropy_coef_min = getattr(adaptive_cfg, 'min_coef', 0.001)
            self.entropy_coef_max = getattr(adaptive_cfg, 'max_coef', 0.5)
        else:
            self.adaptive_entropy = False

    def _adapt_entropy_coef(self, entropy_value: float):
        if not self.adaptive_entropy:
            return
        error = self.entropy_target - entropy_value
        self.entropy_coef += self.entropy_adapt_rate * error
        self.entropy_coef = max(self.entropy_coef_min,
                                min(self.entropy_coef_max, self.entropy_coef))


def _make_config(**overrides):
    base = {
        'entropy_coef': 0.05,
        'adaptive_entropy': {
            'enabled': True,
            'target_entropy': 1.0,
            'adaptation_rate': 0.001,
            'min_coef': 0.001,
            'max_coef': 0.5,
        },
    }
    base.update(overrides)
    return OmegaConf.create(base)


class TestAdaptiveEntropy:
    """Tests for the adaptive entropy coefficient mechanism."""

    def test_adaptive_entropy_increases_coef_when_entropy_low(self):
        """If entropy is below target, coef should increase."""
        config = _make_config()
        trainer = MockAdaptiveEntropyTrainer(config)
        initial_coef = trainer.entropy_coef

        # Entropy = 0.1, well below target of 1.0
        trainer._adapt_entropy_coef(0.1)
        assert trainer.entropy_coef > initial_coef

    def test_adaptive_entropy_decreases_coef_when_entropy_high(self):
        """If entropy is above target, coef should decrease."""
        config = _make_config(entropy_coef=0.3)
        trainer = MockAdaptiveEntropyTrainer(config)
        initial_coef = trainer.entropy_coef

        # Entropy = 5.0, well above target of 1.0
        trainer._adapt_entropy_coef(5.0)
        assert trainer.entropy_coef < initial_coef

    def test_adaptive_entropy_respects_min_coef(self):
        """Coefficient should not drop below min_coef."""
        config = _make_config(entropy_coef=0.002)
        trainer = MockAdaptiveEntropyTrainer(config)

        # Drive entropy very high -> coef should decrease but not below min
        for _ in range(10000):
            trainer._adapt_entropy_coef(100.0)

        assert trainer.entropy_coef == trainer.entropy_coef_min

    def test_adaptive_entropy_respects_max_coef(self):
        """Coefficient should not exceed max_coef."""
        config = _make_config(entropy_coef=0.4)
        trainer = MockAdaptiveEntropyTrainer(config)

        # Drive entropy very low -> coef should increase but not above max
        for _ in range(10000):
            trainer._adapt_entropy_coef(0.0)

        assert trainer.entropy_coef == trainer.entropy_coef_max

    def test_adaptive_entropy_stable_at_target(self):
        """When entropy equals target, coef should stay the same."""
        config = _make_config()
        trainer = MockAdaptiveEntropyTrainer(config)
        initial_coef = trainer.entropy_coef

        trainer._adapt_entropy_coef(1.0)  # exactly at target
        assert trainer.entropy_coef == initial_coef

    def test_adaptive_entropy_disabled(self):
        """When disabled, coef should never change."""
        config = _make_config()
        config.adaptive_entropy.enabled = False
        trainer = MockAdaptiveEntropyTrainer(config)
        initial_coef = trainer.entropy_coef

        trainer._adapt_entropy_coef(0.0)
        assert trainer.entropy_coef == initial_coef

    def test_adaptive_entropy_not_configured(self):
        """When adaptive_entropy key is missing entirely, coef should not change."""
        config = OmegaConf.create({'entropy_coef': 0.05})
        trainer = MockAdaptiveEntropyTrainer(config)
        initial_coef = trainer.entropy_coef

        trainer._adapt_entropy_coef(0.0)
        assert trainer.entropy_coef == initial_coef
        assert trainer.adaptive_entropy is False

    def test_convergence_toward_target(self):
        """Repeated adaptation should push entropy_coef to maintain target."""
        config = _make_config(entropy_coef=0.05)
        config.adaptive_entropy.adaptation_rate = 0.01
        trainer = MockAdaptiveEntropyTrainer(config)

        # Simulate many steps where entropy is consistently low
        for _ in range(200):
            trainer._adapt_entropy_coef(0.2)

        # Coefficient should have increased substantially toward max
        assert trainer.entropy_coef > 0.2
