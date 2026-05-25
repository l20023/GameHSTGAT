"""Re-plot anchored learning-rate decay from saved metrics.json logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hst_bound import compute_beta_hst_max
from src.learning_rate_plots import (
    PlotVariant,
    learning_rate_plot_path,
    save_learning_rate_plot,
)
from src.training_pipeline import (
    DEFAULT_CONVERGENCE_WARNING_THRESHOLD,
    FitAnchor,
    fit_beta_from_epsilon,
    normalize_fit_anchor,
)


def _load_metrics(metrics_path: Path) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "conditions" not in payload:
        raise ValueError(f"Invalid metrics file (missing 'conditions'): {metrics_path}")
    return payload


def _infer_seed(metrics_path: Path, payload: dict[str, Any]) -> int:
    if isinstance(payload.get("seed"), int):
        return int(payload["seed"])
    parent = metrics_path.parent.name
    if parent.startswith("seed_"):
        return int(parent.removeprefix("seed_"))
    raise ValueError(
        "Could not infer seed from metrics.json; pass --seed explicitly."
    )


def _convergence_warning(
    beta_fit: dict[str, Any],
    *,
    threshold: float = DEFAULT_CONVERGENCE_WARNING_THRESHOLD,
) -> bool:
    epsilon_inf = beta_fit.get("epsilon_inf")
    return bool(
        isinstance(epsilon_inf, (int, float))
        and np.isfinite(float(epsilon_inf))
        and float(epsilon_inf) > threshold
    )


def _beta_gap_and_exceed(
    beta_fit: dict[str, Any],
    *,
    beta_hst_max: float,
    convergence_warning: bool,
) -> tuple[float | None, bool | None]:
    if not bool(beta_fit.get("fit_success", False)):
        return None, None
    beta_raw = beta_fit.get("beta")
    if not isinstance(beta_raw, (int, float)) or not np.isfinite(float(beta_raw)):
        return None, None
    gap = float(beta_raw) - beta_hst_max
    exceeds = (gap > 0.0) and not convergence_warning
    return gap, exceeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild anchored learning-rate plots from metrics.json. "
            "Default fit-anchor is t1 (empirical epsilon at round 1). "
            "Use --fit-window-t-max to choose how many leading rounds enter the beta fit."
        )
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        required=True,
        help="Path to metrics.json (e.g. artifacts/_smoke/seed_1/metrics.json).",
    )
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        help="Condition key inside metrics, e.g. n_10/complete.",
    )
    parser.add_argument(
        "--list-conditions",
        action="store_true",
        help="Print condition keys in the metrics file and exit.",
    )
    parser.add_argument(
        "--fit-window-t-max",
        type=int,
        default=None,
        help=(
            "Last round index included in the fit (1-based, inclusive). "
            "Omit to use auto truncation (perfect-error suffix only)."
        ),
    )
    parser.add_argument(
        "--fit-anchor",
        type=str,
        choices=["t1", "t0"],
        default="t1",
        help=(
            "Decay anchor: t1 uses empirical epsilon(1) (default, matches training pipeline); "
            "t0 uses prior epsilon(0)=0.5 at round 0 for sensitivity analysis."
        ),
    )
    parser.add_argument(
        "--signal-quality",
        type=float,
        default=None,
        help="Signal quality q for the HST bound (must match the training run).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for default output path naming (inferred from metrics when omitted).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Default: alongside metrics under plots/.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Base artifacts dir for default plot path (default: metrics file parent).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = args.metrics.expanduser().resolve()
    payload = _load_metrics(metrics_path)
    conditions = payload["conditions"]
    if not isinstance(conditions, dict):
        raise ValueError(f"'conditions' must be a dict in {metrics_path}")

    if args.list_conditions:
        for key in sorted(conditions):
            print(key)
        return

    condition_key = args.condition
    if condition_key not in conditions:
        available = ", ".join(sorted(conditions))
        raise KeyError(
            f"Unknown condition {condition_key!r}. Available: {available}"
        )

    condition = conditions[condition_key]
    epsilon_series = condition.get("epsilon_series")
    if not isinstance(epsilon_series, list) or not epsilon_series:
        raise ValueError(
            f"Condition {condition_key!r} has no epsilon_series in {metrics_path}. "
            "Re-run training with save_epsilon_series: true."
        )

    if args.signal_quality is None:
        raise ValueError("--signal-quality is required (must match the training run).")

    seed = args.seed if args.seed is not None else _infer_seed(metrics_path, payload)
    fit_anchor: FitAnchor = normalize_fit_anchor(args.fit_anchor)
    plot_variant: PlotVariant = "anchored_t1" if fit_anchor == "t1" else "anchored_t0"
    beta_fit = fit_beta_from_epsilon(
        [float(v) for v in epsilon_series],
        fit_window_t_max=args.fit_window_t_max,
        anchor=fit_anchor,
    )
    beta_hst_max = compute_beta_hst_max(float(args.signal_quality))
    convergence_warning = _convergence_warning(beta_fit)
    beta_gap, exceeds_hst_bound = _beta_gap_and_exceed(
        beta_fit,
        beta_hst_max=beta_hst_max,
        convergence_warning=convergence_warning,
    )

    if args.output is not None:
        output_path = args.output.expanduser().resolve()
    else:
        artifacts_dir = (
            args.artifacts_dir.expanduser().resolve()
            if args.artifacts_dir is not None
            else metrics_path.parent.parent
        )
        output_path = learning_rate_plot_path(
            artifacts_dir=artifacts_dir,
            seed=seed,
            condition_key=condition_key,
            plot_variant=plot_variant,
        )
        if args.fit_window_t_max is not None:
            safe = condition_key.replace("/", "__")
            output_path = output_path.with_name(
                f"{safe}__{plot_variant}__tmax_{args.fit_window_t_max}.png"
            )

    save_learning_rate_plot(
        output_path=output_path,
        epsilon_series=[float(v) for v in epsilon_series],
        beta_fit=beta_fit,
        beta_hst_max=beta_hst_max,
        condition_key=condition_key,
        signal_quality=float(args.signal_quality),
        beta_gap=beta_gap,
        exceeds_hst_bound=exceeds_hst_bound,
        convergence_warning=convergence_warning,
        plot_variant=plot_variant,
        fit_anchor=fit_anchor,
    )

    print(f"Wrote plot: {output_path}")
    print(
        f"fit_window_t_max={beta_fit.get('fit_window_t_max')} "
        f"n_full_series={beta_fit.get('n_full_series')} "
        f"plateau_detected={beta_fit.get('plateau_detected')} "
        f"beta={beta_fit.get('beta')} "
        f"epsilon_inf={beta_fit.get('epsilon_inf')}"
    )


if __name__ == "__main__":
    main()
