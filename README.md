# Game Theory Project

RGAT-based social learning experiments to compare empirical learning dynamics against the HST bound.

## Setup

1. Install dependencies:
   - `pip install -e .`

## Single Run (YAML-based)

The training script runs one configuration per invocation.

- Base config file: `configs/default.yaml`
- CLI overrides are supported and take precedence over YAML.
- Default config highlights:
  - `device: auto` (prefers CUDA, then MPS, then CPU)
  - `num_seeds: 5`
  - `train_episodes: 5000` (fallback; grid uses adaptive `train_episodes_per_n`: 10→5000, 100→7000, 1000→10000)
  - `max_horizon: 100`
  - `communication_mode: fair_1bit` (default fair HST benchmark)
  - grid defaults: `num_nodes_list=10,100,1000`, `signal_quality_list=0.55,0.6,0.7,0.8`

Example:

`python scripts/train.py --config configs/default.yaml --seed 3 --num-nodes 50`

This run automatically:
- builds/loads proposal graph conditions (`complete`, `ws p=0.0`, `ws p=0.1`)
- trains and evaluates one seed/node setup
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
- `--device` (`auto`, `cpu`, `cuda`, `mps`)
- `--graph-cache-dir`
- `--artifacts-dir`
- `--disable-beta-fit`

Device behavior:
- `auto` chooses `cuda` when available, otherwise `mps` when available, otherwise `cpu`.
- Explicit `cuda` / `mps` requests fail fast if that backend is unavailable.

## Bash orchestration for multiple runs

Run 5 replication seeds (single setting, e.g. `n=10`):

```bash
for seed in $(seq 0 4); do
  python scripts/train.py --config configs/default.yaml --seed "$seed" --num-nodes 10
done
```

Or use the helper script:

```bash
bash scripts/run_replication.sh configs/default.yaml 10
# optional third arg: number of seeds (default 5 -> seeds 0..4)
```

Run multiple node sizes:

```bash
for n in 10 100 1000; do
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

Each script runs the full matrix (`n={10,100,1000}`, `q={0.55,0.6,0.7,0.8}`, **5 seeds** `0..4`, `T=100` from config),
writes a grid summary JSON, builds per-run + seed-aggregated CSV tables, and emits aggregate plots
(beta_GAT vs q and beta_GAT vs n with error bars and HST reference lines).

| Script | Mode | Artifacts | Summary |
|--------|------|-----------|---------|
| `run_experiment_fair.sh` | `fair_1bit` | `artifacts/training_metrics_fair/` | `artifacts/grid_summary_fair.json` |
| `run_experiment_vector.sh` | `vector` (dim 32) | `artifacts/training_metrics_vector/` | `artifacts/grid_summary_vector.json` |

Aggregate plots are written under `<artifacts_dir>/grid_runs/aggregate_plots/`.

Smoke test (3 seeds, small grid):

```bash
bash scripts/run_experiment_fair.sh --seeds 0,1,2 --num-nodes-list 10 --signal-quality-list 0.6
```

### SLURM grid runs (cluster)

Parallel submission via SLURM job array (one task = one `(seed, n, q)` cell, all graph conditions in that job).
Scripts live under `scripts/slurm/` and follow the same pattern as the example in `Beispiel für slurms/`.

**Fair benchmark (max 20 parallel jobs):**

```bash
bash scripts/run_experiment_fair_slurm.sh 20
# or directly:
bash scripts/slurm/run_batch_slurms.sh 20 fair_1bit
```

**Vector ablation:**

```bash
bash scripts/run_experiment_vector_slurm.sh 20
```

**Smoke subset on cluster:**

```bash
SEEDS=0,1 NUM_NODES_LIST=10 SIGNAL_QUALITY_LIST=0.6 \
  bash scripts/slurm/run_batch_slurms.sh 4 fair_1bit
```

Each submission:
- submits a throttled array (`--array=1-N%MAX_PARALLEL`) where `N = seeds × |n| × |q|`
- runs `scripts/run_grid_task.py --task-index $((SLURM_ARRAY_TASK_ID - 1))` per GPU job
- optionally submits a finalize job (`finalize_job.sh`) that writes `grid_summary_*.json` and CSV tables

Cluster defaults (in `scripts/slurm/job.sh`):
- `REPO_DIR=/rds/user/lmk66/hpc-work/GameHSTGAT`
- account `LIO-SL3-GPU`, partition `ampere`, `conda activate poly-shgnn-gpu`

Adaptive walltime — **one SLURM array per n** (shorter requests queue faster on busy GPU partitions):

| n | default walltime | env override |
|---|------------------|--------------|
| 10 | 02:00:00 | `SBATCH_TIME_N10` |
| 100 | 04:00:00 | `SBATCH_TIME_N100` |
| 1000 | 08:00:00 | `SBATCH_TIME_N1000` |

Shorter requests improve queue priority on busy partitions. If a tier times out, re-submit only that tier with a higher override — do not raise all tiers at once.

Example:

```bash
SBATCH_TIME_N1000=08:00:00 bash scripts/slurm/run_batch_slurms.sh 20 fair_1bit
```

Skip finalize (e.g. while debugging array tasks):

```bash
SUBMIT_FINALIZE=0 bash scripts/slurm/run_batch_slurms.sh 20 fair_1bit
```

Logs are written to `logs/`.

### Low-level grid CLI

`python scripts/run_grid.py --config configs/default.yaml`

By default this runs **5 seeds** (`0..4`) from `num_seeds: 5` in `configs/default.yaml`.
Override with `--seeds 0,1,2,3,4` or set an explicit list in YAML (`seeds: [0, 1, 2, ...]`).

This writes:
- per-run metrics under `<artifacts_dir>/grid_runs/...`
- an aggregated summary JSON at `artifacts/grid_summary.json` (override via `--output`)

The summary also includes `regime_classification.headline_label` with one compact status:
- `inconclusive` (insufficient fit quality or high `convergence_warning` rate)
- `empirical_counter_evidence`
- `boundary_condition_evidence`
- `consistent_with_equilibrium_bound`

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
- Plots: `artifacts/.../seed_<seed>/plots/<condition>__anchored_t1.png` and `<condition>__train_loss.png`

Each metrics file stores beta fit, HST comparison, and `train_loss_final`. Full
`train_loss_history` / `epsilon_series` arrays in JSON are omitted unless enabled in config.

Learning-rate plots use the default **fit-anchor `t1`** (`__anchored_t1` filename suffix):
- Fit model: `epsilon(t) = (epsilon(1) - epsilon_inf) * exp(-beta * (t-1)) + epsilon_inf` for `t = 1..T`.
- `epsilon(1)` is the **empirical** test error at round 1 (agents already use private signals); the curve passes through that point exactly.
- GAT and HST reference curves share fitted `epsilon_inf` and the same `epsilon(1)`; HST uses `beta_HST_max` (max permitted slope, not a forecast).
- **Log panel:** `epsilon(t) - epsilon_inf` with the same `t=1` anchor for slope comparison.

**Fit-anchor `t0`** (sensitivity only, not the pipeline default): prior `epsilon(0)=0.5` at round 0,
`epsilon(t) = (0.5 - epsilon_inf) * exp(-beta * t) + epsilon_inf`. Select via
`--fit-anchor t0` in `plot_learning_rate_from_logs.py`; output suffix is `__anchored_t0`.

### Re-plot learning-rate curves from saved logs

Requires `save_epsilon_series: true` in the training config so `metrics.json` contains `epsilon_series`.
`--signal-quality` must match the original run (it is not stored per condition in metrics).

List conditions in a metrics file:

```bash
python scripts/plot_learning_rate_from_logs.py \
  --metrics artifacts/_smoke/seed_1/metrics.json \
  --condition dummy \
  --list-conditions
```

Auto-fit with default anchor `t1` (truncates only a trailing suffix with perfect error rate `epsilon=0`):

```bash
python scripts/plot_learning_rate_from_logs.py \
  --metrics artifacts/_smoke/seed_1/metrics.json \
  --condition n_10/complete \
  --signal-quality 0.6
```

Prior anchor `t0` (uninformed prior at round 0):

```bash
python scripts/plot_learning_rate_from_logs.py \
  --metrics artifacts/_smoke/seed_1/metrics.json \
  --condition n_10/complete \
  --signal-quality 0.6 \
  --fit-anchor t0
```

Manual fit window — last round **included** in the beta fit (`1`-based, inclusive):

```bash
python scripts/plot_learning_rate_from_logs.py \
  --metrics artifacts/_smoke/seed_1/metrics.json \
  --condition n_10/complete \
  --signal-quality 0.6 \
  --fit-window-t-max 17
```

Custom output path:

```bash
python scripts/plot_learning_rate_from_logs.py \
  --metrics artifacts/training_metrics_fair/grid_runs/n_10/q_0p6/seed_0/metrics.json \
  --condition n_10/complete \
  --signal-quality 0.6 \
  --fit-window-t-max 15 \
  --output artifacts/replots/complete_t15.png
```

Without `--output`, a manual `--fit-window-t-max` writes
`<artifacts>/seed_<S>/plots/<condition>__anchored_t1__tmax_<T>.png` (or `__anchored_t0__` with `--fit-anchor t0`)
so original plots are not overwritten.

Reading order for plots:
1. Aggregate plots in `<artifacts_dir>/grid_runs/aggregate_plots/` (or `aggregate/plots/` after replication) give the headline view (beta vs q, beta vs n with HST reference).
2. Per-condition `__anchored_t1` plots show the empirical decay vs both fits for a single seed/setting (use only for diagnostics).
3. Per-seed `metrics.json` contains per-fit Wald CIs (`beta_std`, `beta_ci_lower`, `beta_ci_upper`) for sanity checks.

Each condition now also includes:
- `convergence_warning` (true when fitted `epsilon_inf > 0.05`)

Config flags (`configs/default.yaml`):

- `save_train_loss_history: false`
- `save_epsilon_series: true`
- `save_learning_rate_plots: true`

### Summarize all metrics into one table

```bash
python scripts/summarize_metrics.py \
  --root artifacts/training_metrics_fair/grid_runs \
  --csv artifacts/metrics_summary_fair.csv \
  --aggregate-csv artifacts/metrics_summary_fair_aggregated.csv \
  --aggregate-json artifacts/metrics_summary_fair_aggregated.json \
  --aggregate-plots-dir artifacts/training_metrics_fair/grid_runs/aggregate_plots
```

`--aggregate-csv` reports mean/std of `beta_gat` and `beta_gap` across seeds per `(n, q, topology)` cell,
plus the best run (`beta_gat_best`, `beta_gat_best_seed`, `beta_gap_at_best`).
`scripts/run_replication.sh` runs the same aggregation automatically after all seeds finish.

Scan multiple artifact trees (e.g. grid + smoke) with repeated `--root`. Columns include
`beta_gat`, `beta_hst_max`, `beta_gap`, `exceeds_hst_bound`, and fit diagnostics per condition.

For a full technical walkthrough of the model and design decisions, see `MODEL_README.md`.

## Interpreting HST comparisons

- `beta_GAT <= beta_HST_max` means the trained RGAT is slower than (or equal to) the HST equilibrium benchmark for that setup.
- This is consistent with the equilibrium bound, but does **not** prove a universal information-theoretic impossibility result for all learning algorithms.
- `beta_GAT > beta_HST_max` is empirical counter-evidence relative to the equilibrium model assumptions; it does not invalidate the HST theorem itself.

## Artifact state in this repository

- Active path for current fair runs: `artifacts/training_metrics_fair/`.
- Historical runs are under `artifacts/First Trainings/` and include the broader `n={10,50,100}` sweeps.
- The single observed exceedance in local artifacts appears at `n=50, q=0.6, complete, seed=0` in `First Trainings`; other seeds for that setting do not exceed.

## Proposal vs implementation defaults

- The implementation currently uses `max_horizon: 100`, adaptive `train_episodes_per_n` (5000/7000/10000 for n=10/100/1000), and `test_episodes: 1000`.
- Older proposal text and early artifacts may reference `T_max=50` and larger training budgets; treat those as historical planning values.
