"""Training-loss curve plots per condition."""

from __future__ import annotations

from pathlib import Path


def train_loss_plot_path(*, artifacts_dir: Path, seed: int, condition_key: str) -> Path:
    """Return the PNG path for one condition's training-loss curve."""
    safe_name = condition_key.replace("/", "__")
    return artifacts_dir / f"seed_{seed}" / "plots" / f"{safe_name}__train_loss.png"


def save_train_loss_plot(
    *,
    output_path: Path,
    train_loss_history: list[float],
    condition_key: str,
) -> Path:
    """Plot per-episode training loss and write a PNG artifact."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not train_loss_history:
        raise ValueError("train_loss_history must be non-empty to plot training loss.")

    episodes = list(range(1, len(train_loss_history) + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(episodes, train_loss_history, "-", color="#1f77b4", linewidth=1.2)
    ax.set_title(f"{condition_key}  |  training loss", fontsize=11)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Loss (mean CE over rounds × nodes)")
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


__all__ = ["save_train_loss_plot", "train_loss_plot_path"]
