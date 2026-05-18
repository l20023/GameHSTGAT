import pytest
import torch

from src.signal_generator import PrivateSignalGenerator


def test_sample_true_state_returns_binary_value() -> None:
    generator = PrivateSignalGenerator(signal_quality=0.8)
    theta = generator.sample_true_state(seed=5)
    assert theta in {0, 1}


def test_generate_private_signals_shape_and_binary_values() -> None:
    generator = PrivateSignalGenerator(signal_quality=0.8)
    signals = generator.generate_private_signals(
        num_nodes=10,
        max_horizon=20,
        theta=1,
        seed=9,
    )
    assert signals.shape == (20, 10)
    assert signals.dtype == torch.long
    assert torch.all((signals == 0) | (signals == 1))


def test_generate_episode_exposes_initial_private_signal() -> None:
    generator = PrivateSignalGenerator(signal_quality=0.8)
    episode = generator.generate_episode(num_nodes=6, max_horizon=4, seed=21)
    assert episode["theta"] in {0, 1}
    assert torch.equal(episode["initial_private_signal"], episode["private_signals"][0])


def test_generate_episode_is_deterministic_for_fixed_seed() -> None:
    generator = PrivateSignalGenerator(signal_quality=0.75)
    episode_a = generator.generate_episode(num_nodes=12, max_horizon=7, seed=42)
    episode_b = generator.generate_episode(num_nodes=12, max_horizon=7, seed=42)
    assert episode_a["theta"] == episode_b["theta"]
    assert torch.equal(episode_a["private_signals"], episode_b["private_signals"])
    assert torch.equal(episode_a["initial_private_signal"], episode_b["initial_private_signal"])


def test_signal_quality_matches_empirical_rate() -> None:
    q = 0.8
    generator = PrivateSignalGenerator(signal_quality=q)
    theta = 1
    signals = generator.generate_private_signals(
        num_nodes=200,
        max_horizon=500,
        theta=theta,
        seed=7,
    )
    empirical_accuracy = (signals == theta).float().mean().item()
    assert abs(empirical_accuracy - q) < 0.02


@pytest.mark.parametrize("signal_quality", [0.5, 1.0, -0.1])
def test_invalid_signal_quality_raises_value_error(signal_quality: float) -> None:
    with pytest.raises(ValueError):
        PrivateSignalGenerator(signal_quality=signal_quality)


@pytest.mark.parametrize(
    ("num_nodes", "max_horizon"),
    [
        (0, 10),
        (10, 0),
    ],
)
def test_invalid_episode_shape_raises_value_error(num_nodes: int, max_horizon: int) -> None:
    generator = PrivateSignalGenerator(signal_quality=0.8)
    with pytest.raises(ValueError):
        generator.generate_episode(num_nodes=num_nodes, max_horizon=max_horizon)


def test_invalid_theta_raises_value_error() -> None:
    generator = PrivateSignalGenerator(signal_quality=0.8)
    with pytest.raises(ValueError):
        generator.generate_private_signals(
            num_nodes=10,
            max_horizon=5,
            theta=2,
            seed=1,
        )
