"""Private binary signal generation for social-learning episodes."""

from __future__ import annotations

import torch


class PrivateSignalGenerator:
    """Generate true state and private binary signals for each node and round."""

    def __init__(self, *, signal_quality: float, default_seed: int | None = None) -> None:
        self._validate_signal_quality(signal_quality)
        self.signal_quality = float(signal_quality)
        self.default_seed = default_seed

    @staticmethod
    def _validate_signal_quality(signal_quality: float) -> None:
        if not (0.5 < signal_quality < 1.0):
            raise ValueError("signal_quality must satisfy 0.5 < signal_quality < 1.0.")

    @staticmethod
    def _validate_episode_shape(num_nodes: int, max_horizon: int) -> None:
        if num_nodes < 1:
            raise ValueError("num_nodes must be >= 1.")
        if max_horizon < 1:
            raise ValueError("max_horizon must be >= 1.")

    @staticmethod
    def _validate_theta(theta: int) -> None:
        if theta not in {0, 1}:
            raise ValueError("theta must be either 0 or 1.")

    def _resolve_generator(self, *, seed: int | None, generator: torch.Generator | None) -> torch.Generator:
        if seed is not None and generator is not None:
            raise ValueError("Pass either seed or generator, not both.")

        if generator is not None:
            return generator

        final_seed = self.default_seed if seed is None else seed
        resolved_generator = torch.Generator(device="cpu")
        if final_seed is not None:
            resolved_generator.manual_seed(final_seed)
        return resolved_generator

    def sample_true_state(
        self,
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> int:
        """Sample the binary ground-truth state theta in {0,1}."""
        resolved_generator = self._resolve_generator(seed=seed, generator=generator)
        return int(torch.randint(0, 2, (1,), generator=resolved_generator).item())

    def generate_private_signals(
        self,
        *,
        num_nodes: int,
        max_horizon: int,
        theta: int,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Generate private binary signals with shape (max_horizon, num_nodes).

        A signal equals theta with probability signal_quality and equals 1-theta otherwise.
        """
        self._validate_episode_shape(num_nodes, max_horizon)
        self._validate_theta(theta)
        resolved_generator = self._resolve_generator(seed=seed, generator=generator)

        is_correct_signal = torch.rand(
            (max_horizon, num_nodes), generator=resolved_generator
        ) < self.signal_quality
        theta_tensor = torch.full((max_horizon, num_nodes), theta, dtype=torch.long)
        opposite_theta_tensor = 1 - theta_tensor
        return torch.where(is_correct_signal, theta_tensor, opposite_theta_tensor)

    def generate_episode(
        self,
        *,
        num_nodes: int,
        max_horizon: int,
        seed: int | None = None,
    ) -> dict[str, int | torch.Tensor]:
        """Sample one full episode and expose round-0 private input explicitly."""
        self._validate_episode_shape(num_nodes, max_horizon)
        episode_generator = self._resolve_generator(seed=seed, generator=None)
        theta = self.sample_true_state(generator=episode_generator)
        private_signals = self.generate_private_signals(
            num_nodes=num_nodes,
            max_horizon=max_horizon,
            theta=theta,
            generator=episode_generator,
        )
        return {
            "theta": theta,
            "private_signals": private_signals,
            "initial_private_signal": private_signals[0].clone(),
        }


__all__ = ["PrivateSignalGenerator"]
