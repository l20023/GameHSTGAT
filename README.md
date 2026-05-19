# Game Theory Project

RGAT-based social learning experiments to compare empirical learning dynamics against the HST bound.

## Setup

1. Install dependencies:
   - `pip install -e .`
2. Login to Weights & Biases:
   - `wandb login`
   - for local runs, the login token is used automatically

## Single Run (YAML-based)

The training script runs one configuration per invocation.

- Base config file: `configs/default.yaml`
- CLI overrides are supported and take precedence over YAML.
- Default config highlights:
  - `wandb_project: game-theory-project`
  - `wandb_entity: GameHSTGAT`
  - `max_horizon: 100`
  - `communication_mode: fair_1bit` (default fair HST benchmark)

Example:

`python scripts/train.py --config configs/default.yaml --seed 3 --num-nodes 50`

This run automatically:
- builds/loads proposal graph conditions (`complete`, `ws p=0.0`, `ws p=0.1`)
- trains and evaluates one seed/node setup
- logs to W&B (`wandb_entity` default: `GameHSTGAT`)
- saves local metrics JSON under `artifacts_dir/seed_<seed>/metrics.json`

## Useful CLI Overrides

- `--seed`
- `--num-nodes`
- `--train-episodes`
- `--test-episodes`
- `--max-horizon`
- `--signal-quality`
- `--hidden-dim`
- `--num-heads`
- `--communication-mode` (`fair_1bit` or `vector`)
- `--communication-dim` (used for `vector` mode)
- `--learning-rate`
- `--graph-cache-dir`
- `--artifacts-dir`
- `--wandb-project`
- `--wandb-entity`
- `--disable-beta-fit`

## Bash orchestration for multiple runs

Run 3 replication seeds (single setting, e.g. `n=50`):

```bash
for seed in $(seq 0 2); do
  python scripts/train.py --config configs/default.yaml --seed "$seed" --num-nodes 50
done
```

Or use the helper script:

```bash
bash scripts/run_replication.sh configs/default.yaml 50
# optional third arg: number of seeds (default 3 -> seeds 0..2)
```

Run multiple node sizes:

```bash
for n in 10 50 100; do
  python scripts/train.py --config configs/default.yaml --seed 0 --num-nodes "$n"
done
```

## Grid runner (proposal matrix)

### One-command experiment scripts (recommended)

**Fair HST benchmark** (1-bit communication, separate artifacts):

```bash
bash scripts/run_experiment_fair.sh
```

**Vector ablation** (higher-dimensional messages, not the strict fair benchmark):

```bash
bash scripts/run_experiment_vector.sh
```

Each script runs the full matrix (`n={10,50,100}`, `q={0.6,0.8}`, **3 seeds** `0..2`, `T=100` from config),
writes a grid summary JSON, and builds per-run + seed-aggregated CSV tables.

| Script | Mode | Artifacts | Summary |
|--------|------|-----------|---------|
| `run_experiment_fair.sh` | `fair_1bit` | `artifacts/training_metrics_fair/` | `artifacts/grid_summary_fair.json` |
| `run_experiment_vector.sh` | `vector` (dim 32) | `artifacts/training_metrics_vector/` | `artifacts/grid_summary_vector.json` |

Smoke test (3 seeds):

```bash
bash scripts/run_experiment_fair.sh --seeds 0,1,2
```

### Low-level grid CLI

`python scripts/run_grid.py --config configs/default.yaml`

By default this runs **3 seeds** (`0..2`) from `num_seeds: 3` in `configs/default.yaml`.
Override with `--seeds 0,1,2,3` or set an explicit list in YAML (`seeds: [0, 1, 2, ...]`).

This writes:
- per-run metrics under `<artifacts_dir>/grid_runs/...`
- an aggregated summary JSON at `artifacts/grid_summary.json` (override via `--output`)

The summary also includes `regime_classification.headline_label` with one compact status:
- `empirical_counter_evidence`
- `boundary_condition_evidence`
- `supports_information_theoretic_limit`
- `inconclusive`

HST reference bound used in code:
- `beta_HST_max(q) = 2 * log(q/(1-q))` for binary symmetric signals
- based on HST bound `M = 2 * sup |log-likelihood|` (Eq. (1), Theorem 1)
- source: [https://arxiv.org/pdf/2112.14265](https://arxiv.org/pdf/2112.14265)

Fairness protocol for HST comparison:
- default mode `communication_mode=fair_1bit` enforces a strict 1-bit visible channel between agents
- this uses an STE bridge so forward communication stays binary while training remains differentiable
- `communication_mode=vector` is kept for ablations and is not the strict fair benchmark

## Artifacts

- Graph cache: `artifacts/graphs`
- Metrics: `artifacts/training_metrics/seed_<seed>/metrics.json` (compact by default)
- Plots: `artifacts/.../seed_<seed>/plots/<condition>.png` (empirical `ε(t)`, GAT fit, HST bound curve)

Each metrics file stores beta fit, HST comparison, and `train_loss_final`. Full
`train_loss_history` / `epsilon_series` arrays are omitted unless enabled in config.

Config flags (`configs/default.yaml`):

- `save_train_loss_history: false`
- `save_epsilon_series: false`
- `save_learning_rate_plots: true`

### Summarize all metrics into one table

```bash
python scripts/summarize_metrics.py \
  --root artifacts/training_metrics/grid_runs \
  --csv artifacts/metrics_summary.csv \
  --aggregate-csv artifacts/metrics_summary_by_topology.csv \
  --aggregate-markdown artifacts/metrics_summary_by_topology.md
```

`--aggregate-csv` reports mean/std of `beta_gat` and `beta_gap` across seeds per topology,
plus the best run (`beta_gat_best`, `beta_gat_best_seed`, `beta_gap_at_best`).

Scan multiple artifact trees (e.g. grid + smoke) with repeated `--root`. Columns include
`beta_gat`, `beta_hst_max`, `beta_gap`, `exceeds_hst_bound`, and fit diagnostics per condition.

For a full technical walkthrough of the model and design decisions, see `MODEL_README.md`.
