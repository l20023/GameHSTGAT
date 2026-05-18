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
  - `max_horizon: 50`

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
- `--learning-rate`
- `--graph-cache-dir`
- `--artifacts-dir`
- `--wandb-project`
- `--wandb-entity`
- `--disable-beta-fit`

## Bash orchestration for multiple runs

Run multiple seeds:

```bash
for seed in 0 1 2 3 4; do
  python scripts/train.py --config configs/default.yaml --seed "$seed" --num-nodes 50
done
```

Run multiple node sizes:

```bash
for n in 10 50 100; do
  python scripts/train.py --config configs/default.yaml --seed 0 --num-nodes "$n"
done
```

## Grid runner (proposal matrix)

Run the proposal matrix (`n={10,50,100}`, `q={0.6,0.8}` by default) with one command:

`python scripts/run_grid.py --config configs/default.yaml --seeds 0,1,2`

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

## Artifacts

- Graph cache: `artifacts/graphs`
- Metrics: `artifacts/training_metrics/seed_<seed>/metrics.json`

Each metrics file contains condition-level outputs (`epsilon(t)`, beta fit, final train loss).

For a full technical walkthrough of the model and design decisions, see `MODEL_README.md`.
