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
  - `max_horizon: 50` (100 only for q=0.55 on ws, per condition)
  - `communication_mode: fair_1bit` (default fair HST benchmark)
  - grid defaults: `num_nodes_list=10,100,1000`, `signal_quality_list=0.55,0.65,0.8`

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

Each script runs the full matrix (`n={10,100,1000}`, `q={0.55,0.65,0.8}`, **5 seeds** `0..4`). Horizon: **T=50** by default; **T=100** only for **q=0.55** on **non-complete** topologies (ws).
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

### Majority-vote baseline (CPU, no training)

Cumulative neighbor + private counting baseline aligned with the RGAT protocol (1-bit broadcasts, tie-break with seeded RNG). Evaluates the same grid (**135 cells**: `n×q×5 seeds`, 3 topologies per cell) without GPU training.

```bash
python scripts/run_majority_baseline_grid.py --all   # skips cells that already have metrics.json
python scripts/summarize_metrics.py \
  --root artifacts/training_metrics_majority/grid_runs \
  --csv artifacts/metrics_summary_majority.csv \
  --aggregate-csv artifacts/metrics_summary_majority_aggregated.csv \
  --aggregate-plots-dir artifacts/training_metrics_majority/grid_runs/aggregate_plots
```

Single seed / SLURM task:

```bash
python scripts/run_majority_baseline_grid.py --task-index 0
python scripts/run_majority_baseline.py --seed 0 --num-nodes 10 --signal-quality 0.55
```

Artifacts: `artifacts/training_metrics_majority/grid_runs/` (metrics JSON + anchored t≥2 plots tagged **majority vote baseline**).

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

Canonical layout (fair channel; vector mirrors under `training_metrics_vector/`):

| What | Path |
|------|------|
| Graph cache | `artifacts/graphs/` |
| Per-seed metrics | `artifacts/training_metrics_fair/grid_runs/n_*/q_*/seed_*/metrics.json` |
| Per-seed plots | `.../seed_*/plots/<condition>__anchored_t2.png` |
| Cell mean plots | `.../n_*/q_*/plots/<condition>__anchored_t2_aggregated.png` |
| Aggregate plots | `.../grid_runs/aggregate_plots/beta_vs_{n,q}.png` |
| **Analysis tables** | `artifacts/metrics_summary_{fair,vector,majority}.csv` (per seed) |
| Cell rollups | `artifacts/metrics_summary_*_aggregated.csv` (mean over seeds) |
| Grid overview | `artifacts/grid_summary_{fair,vector}.json` (optional monitoring) |

**For analysis, use `metrics_summary_*.csv`** (per-seed β, exceed flags). Use aggregated CSV for mean curves; `grid_summary_*.json` is optional.

### Beta fitting (t≥2 only)

All fits use **anchor t=2** — social information has mixed into the GAT input from round 2 onward:

`ε(t) = α · exp(−β·(t−2)) + ε∞` for `t ≥ 2`

- GAT: free `α`, `β`, `ε∞` on `t ≥ 2`
- HST reference: same `α` and `ε∞`, `β = β_HST_max(q)` — curves meet at **t=2**
- Plot suffix: `__anchored_t2.png`
- `convergence_warning`: true when fitted `ε∞ > 0.05` (bound comparison suppressed)

Config flags (`configs/default.yaml`):

- `save_train_loss_history: false`
- `save_epsilon_series: true`
- `save_learning_rate_plots: true`

### Re-plot from saved logs

Requires `save_epsilon_series: true`. `--signal-quality` must match the training run.

```bash
python scripts/plot_learning_rate_from_logs.py \
  --metrics artifacts/training_metrics_fair/grid_runs/n_10/q_0p55/seed_0/metrics.json \
  --condition n_10/complete \
  --signal-quality 0.55
```

Refit entire grid + regenerate plots + CSVs:

```bash
python scripts/refit_and_regenerate.py \
  --artifacts-root artifacts/training_metrics_fair/grid_runs \
  --communication-mode fair_1bit
```

### Summarize metrics

```bash
python scripts/summarize_metrics.py \
  --root artifacts/training_metrics_fair/grid_runs \
  --csv artifacts/metrics_summary_fair.csv \
  --aggregate-csv artifacts/metrics_summary_fair_aggregated.csv \
  --aggregate-plots-dir artifacts/training_metrics_fair/grid_runs/aggregate_plots
```

### Interactive episode viewer (cluster → local)

Grid training saves **metrics only**, not model weights. The default export is an **interactive HTML** file:

- Nodes on a **circle** (size scales with `n`), **thin edges**
- **Green** = correct vs θ, **red** = wrong
- **Scrubbable timeline** + Play/Pause
- **Unanimous consensus** status and first consensus round marked on the timeline

**1. On the cluster** — train one cell and save a checkpoint + viewer:

```bash
python scripts/animate_episode.py \
  --seed 0 --num-nodes 10 --signal-quality 0.55 --topology complete \
  --train-episodes 5000 --format html \
  --save-checkpoint artifacts/checkpoints/fair_1bit/n_10/q_0p55/complete/seed_0.pt \
  --output artifacts/animations/fair_1bit/n_10/q_0p55/complete/seed_0.html
```

**2. Download** checkpoint + HTML (+ graph cache if regenerating locally):

```bash
scp cluster:.../artifacts/checkpoints/.../seed_0.pt ./artifacts/checkpoints/...
scp cluster:.../artifacts/animations/.../seed_0.html ./artifacts/animations/...
```

**3. Locally** — regenerate viewer from checkpoint (no training):

```bash
python scripts/animate_episode.py \
  --checkpoint artifacts/checkpoints/fair_1bit/n_10/q_0p55/complete/seed_0.pt \
  --skip-train \
  --seed 0 --num-nodes 10 --signal-quality 0.55 --topology complete \
  --format html --output artifacts/animations/demo.html
```

Open `demo.html` in a browser. Optional legacy GIF: `--format gif` or `--format both`.

Regenerate all checkpoint viewers after template changes:

```bash
bash scripts/render_all_animations.sh
```

### Interactive viewers on GitHub

GitHub README cannot run JavaScript, so viewers are hosted via **[GitHub Pages](https://pages.github.com/)** (deployed from `main` by [`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml)).

**One-time setup (if you see a Pages 404):**

1. Open **Settings → Pages** on [github.com/l20023/GameHSTGAT](https://github.com/l20023/GameHSTGAT/settings/pages)
2. Under **Build and deployment**, set **Source** to **GitHub Actions** (not “Deploy from a branch”)
3. Push to `main` or re-run the **Deploy GitHub Pages** workflow under **Actions**
4. After a green run (~1–2 min), open the launcher below

If **Setup Pages** fails in Actions, step 2 was not done yet.

**[Interactive launcher](https://l20023.github.io/GameHSTGAT/docs/viewers.html)** — pick **model** (fair_1bit / vector / **majority vote baseline**), **size** (10 / 100 / 1000), and **topology** (complete / ws_p_0.0 / ws_p_0.1). Each viewer shows the graph on a circle with scrubbable rounds. URL hash is shareable, e.g. `viewers.html#majority_vote|n100|ws_p_0.1`.

Regenerate majority viewers locally:

```bash
bash scripts/render_majority_animations.sh
```

**[Results & plots](https://l20023.github.io/GameHSTGAT/docs/results.html)** — aggregate β vs q / β vs n for **fair_1bit**, **vector**, and **majority vote baseline**, plus per-run learning-rate plots. Regenerate data before deploy:

```bash
python scripts/summarize_metrics.py \
  --root artifacts/training_metrics_majority/grid_runs \
  --csv artifacts/metrics_summary_majority.csv \
  --aggregate-csv artifacts/metrics_summary_majority_aggregated.csv \
  --aggregate-plots-dir artifacts/training_metrics_majority/grid_runs/aggregate_plots
python scripts/export_results_data.py
```

Inside each viewer:

- scrub rounds with slider, timeline, **← / →**, or Play
- **New signal** — switches to another pre-rendered private-signal episode (12 bundled per checkpoint; true re-roll needs `animate_episode.py` locally)

All experiments (q=0.55, training seed 0) — click to open full-screen on Pages:

| Channel | n | Complete | WS p=0.0 | WS p=0.1 |
|---------|---|----------|----------|----------|
| **fair_1bit** | 10 | [viewer](https://l20023.github.io/GameHSTGAT/artifacts/animations/fair_1bit/n_10/q_0p55/complete/seed_0.html) | [viewer](https://l20023.github.io/GameHSTGAT/artifacts/animations/fair_1bit/n_10/q_0p55/ws_p_0.0/seed_0.html) | [viewer](https://l20023.github.io/GameHSTGAT/artifacts/animations/fair_1bit/n_10/q_0p55/ws_p_0.1/seed_0.html) |
| **fair_1bit** | 100 | [viewer](https://l20023.github.io/GameHSTGAT/artifacts/animations/fair_1bit/n_100/q_0p55/complete/seed_0.html) | [viewer](https://l20023.github.io/GameHSTGAT/artifacts/animations/fair_1bit/n_100/q_0p55/ws_p_0.0/seed_0.html) | [viewer](https://l20023.github.io/GameHSTGAT/artifacts/animations/fair_1bit/n_100/q_0p55/ws_p_0.1/seed_0.html) |
| **fair_1bit** | 1000 | [viewer](https://l20023.github.io/GameHSTGAT/artifacts/animations/fair_1bit/n_1000/q_0p55/complete/seed_0.html) | [viewer](https://l20023.github.io/GameHSTGAT/artifacts/animations/fair_1bit/n_1000/q_0p55/ws_p_0.0/seed_0.html) | [viewer](https://l20023.github.io/GameHSTGAT/artifacts/animations/fair_1bit/n_1000/q_0p55/ws_p_0.1/seed_0.html) |
| **vector** | 10 | [viewer](https://l20023.github.io/GameHSTGAT/artifacts/animations/vector/n_10/q_0p55/complete/seed_0.html) | [viewer](https://l20023.github.io/GameHSTGAT/artifacts/animations/vector/n_10/q_0p55/ws_p_0.0/seed_0.html) | [viewer](https://l20023.github.io/GameHSTGAT/artifacts/animations/vector/n_10/q_0p55/ws_p_0.1/seed_0.html) |
| **vector** | 100 | [viewer](https://l20023.github.io/GameHSTGAT/artifacts/animations/vector/n_100/q_0p55/complete/seed_0.html) | [viewer](https://l20023.github.io/GameHSTGAT/artifacts/animations/vector/n_100/q_0p55/ws_p_0.0/seed_0.html) | [viewer](https://l20023.github.io/GameHSTGAT/artifacts/animations/vector/n_100/q_0p55/ws_p_0.1/seed_0.html) |
| **vector** | 1000 | [viewer](https://l20023.github.io/GameHSTGAT/artifacts/animations/vector/n_1000/q_0p55/complete/seed_0.html) | [viewer](https://l20023.github.io/GameHSTGAT/artifacts/animations/vector/n_1000/q_0p55/ws_p_0.0/seed_0.html) | [viewer](https://l20023.github.io/GameHSTGAT/artifacts/animations/vector/n_1000/q_0p55/ws_p_0.1/seed_0.html) |

Source files: `artifacts/animations/<channel>/n_<N>/q_0p55/<topology>/seed_0.html` (commit to repo for Pages). Locally, open any `.html` file in a browser.

### SLURM (one seed, all q=0.55 cells)

Submits **3 array jobs** (one per `n`), each with **3 tasks** (complete, ws_p_0.0, ws_p_0.1) — **9 GPU jobs total** per channel.

**Fair:**

```bash
SEED=0 bash scripts/run_animate_fair_slurm.sh
```

**Vector:**

```bash
SEED=0 bash scripts/run_animate_vector_slurm.sh
```

Low-level (same as grid pattern):

```bash
SEED=0 bash scripts/slurm/run_animate_slurms.sh 3 fair_1bit
```

Optional: only `n=1000`, or longer walltime if jobs time out:

```bash
SEED=0 NUM_NODES_LIST=1000 SBATCH_TIME_N1000=12:00:00 \
  bash scripts/slurm/run_animate_slurms.sh 3 fair_1bit
```

Outputs: `artifacts/checkpoints/<mode>/n_*/q_0p55/<topo>/seed_<SEED>.pt` and matching `artifacts/animations/.../*.html` (open in browser).

For a full technical walkthrough of the model and design decisions, see `MODEL_README.md`.

## Interpreting HST comparisons

- `beta_GAT <= beta_HST_max` means the trained RGAT is slower than (or equal to) the HST equilibrium benchmark for that setup.
- This is consistent with the equilibrium bound, but does **not** prove a universal information-theoretic impossibility result for all learning algorithms.
- `beta_GAT > beta_HST_max` is empirical counter-evidence relative to the equilibrium model assumptions; it does not invalidate the HST theorem itself.

## Proposal vs implementation defaults

- Grid runs use adaptive `train_episodes_per_n` (5000/7000/10000 for n=10/100/1000), `test_episodes: 1000`, and the horizon policy above (`max_horizon: 50` in config).
- Older proposal text and early artifacts may reference `T_max=50` and larger training budgets; treat those as historical planning values.
