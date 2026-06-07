"""Regenerate metrics and plots using empirical anchor t=2 from saved artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hst_bound import compute_beta_hst_max
from src.learning_rate_plots import learning_rate_plot_path, save_learning_rate_plot
from src.training_pipeline import fit_beta_from_epsilon, normalize_fit_anchor


def _load_metrics(metrics_path: Path) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "conditions" not in payload:
        raise ValueError(f"Invalid metrics file: {metrics_path}")
    return payload


def _save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _render_t2_plots(
    metrics: dict[str, Any],
    metrics_path: Path,
    source_root: Path,
    target_root: Path,
    plot_variant: str = "anchored_t2",
) -> None:
    signal_quality = float(metrics.get("signal_quality"))

    for condition_key, condition in metrics["conditions"].items():
        epsilon_series = condition.get("epsilon_series")
        if not isinstance(epsilon_series, list) or not epsilon_series:
            continue
        beta_fit = fit_beta_from_epsilon(
            [float(v) for v in epsilon_series],
            fit_window_t_max=None,
            anchor="t2",
        )
        beta_hst_max = compute_beta_hst_max(signal_quality)
        epsilon_inf_value = beta_fit.get("epsilon_inf")
        convergence_warning = bool(
            isinstance(epsilon_inf_value, (int, float))
            and float(epsilon_inf_value) == float(epsilon_inf_value)
            and float(epsilon_inf_value) > 0.05
        )
        beta_gap = None
        exceeds_hst_bound = None
        if bool(beta_fit.get("fit_success", False)):
            beta_raw = beta_fit.get("beta")
            if isinstance(beta_raw, (int, float)) and float(beta_raw) == float(beta_raw):
                beta_gap = float(beta_raw) - beta_hst_max
                exceeds_hst_bound = (beta_gap > 0.0) and not convergence_warning

        condition["beta_fit"] = beta_fit
        condition["beta_hst_max"] = beta_hst_max
        condition["beta_gap"] = beta_gap
        condition["exceeds_hst_bound"] = exceeds_hst_bound
        condition["convergence_warning"] = convergence_warning

        relative_dir = metrics_path.parent.relative_to(source_root)
        plot_dir = target_root / relative_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f"{condition_key.replace('/', '__')}__{plot_variant}.png"

        save_learning_rate_plot(
            output_path=plot_path,
            epsilon_series=[float(v) for v in epsilon_series],
            beta_fit=beta_fit,
            beta_hst_max=beta_hst_max,
            condition_key=condition_key,
            signal_quality=signal_quality,
            beta_gap=beta_gap,
            exceeds_hst_bound=exceeds_hst_bound,
            convergence_warning=convergence_warning,
            plot_variant=plot_variant,
            fit_anchor="t2",
        )
        condition["learning_rate_plot_anchored_t2"] = str(plot_path.relative_to(PROJECT_ROOT))


def _process_metrics_file(metrics_path: Path, source_root: Path, target_root: Path) -> None:
    metrics = _load_metrics(metrics_path)
    metrics["anchor"] = "t2"
    relative_path = metrics_path.relative_to(source_root)
    target_metrics_path = target_root / relative_path
    target_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    _render_t2_plots(
        metrics,
        metrics_path=metrics_path,
        source_root=source_root,
        target_root=target_root,
    )
    _save_metrics(metrics, target_metrics_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate anchor-t2 fit artifacts from existing metrics JSON files."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("artifacts"),
        help="Source artifact root containing metrics.json files.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("artifacts/anchor_t2"),
        help="Target root for regenerated anchor-t2 artifacts.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/metrics.json",
        help="Glob pattern for locating metrics files under source-root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    target_root = args.target_root.expanduser().resolve()
    metrics_files = sorted(source_root.glob(args.pattern))
    if not metrics_files:
        raise FileNotFoundError(f"No metrics files found under {source_root} matching {args.pattern}")

    print(f"Reprocessing {len(metrics_files)} metrics files from {source_root} into {target_root}")
    for metrics_path in metrics_files:
        print(f"Processing {metrics_path}")
        _process_metrics_file(metrics_path, source_root=source_root, target_root=target_root)

    print("Done.")


if __name__ == "__main__":
    main()
