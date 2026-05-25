# Model README: RGAT Social Learning System

This document explains what the model does, how data flows through the system, and why key design decisions were made.

## 1) Research goal

The project studies whether a recurrent Graph Attention Network (RGAT) can match or exceed the social-learning speed bound discussed in Huang, Strack, Tamuz (HST).
The comparison is between a gradient-trained neural learner and a bound proven for strategic agents in Nash equilibrium.

Operationally, the pipeline trains an RGAT to predict the binary hidden state over time from:
- each agent's private signals
- graph-structured neighbor communication

Then it evaluates:
- empirical error trajectory `epsilon(t)`
- fitted empirical learning rate `beta_GAT`
- theoretical HST bound `beta_HST_max(q)`
- gap and exceedance (`beta_GAT - beta_HST_max`)

## 2) System components

Core modules:
- `src/signal_generator.py`: synthetic episode generation (`theta`, private signals)
- `src/graph_generator.py`: proposal topologies (complete, Watts-Strogatz)
- `src/models/recurrent_gat_agent.py`: recurrent GAT model + shared sequential loss
- `src/training_pipeline.py`: train/eval loop and beta fitting
- `src/hst_bound.py`: theoretical bound helper (`compute_beta_hst_max`)
- `src/reporting.py`: regime classification for aggregated outputs
- `scripts/train.py`: single configured run
- `scripts/run_grid.py`: orchestrated proposal grid runs and aggregation
- `scripts/run_grid_task.py`: single grid cell (SLURM array worker)
- `scripts/finalize_grid.py`: post-grid aggregation from artifacts
- `scripts/slurm/run_batch_slurms.sh`: throttled SLURM array submission

## 3) Data and graph setup

### Private signal process

For each episode:
- sample hidden state `theta in {0,1}`
- for each node/time pair, sample a private signal that equals `theta` with probability `q` (signal quality)

Design choice:
- binary signal process is intentionally minimal to match the theoretical setup and keep interpretation clean.

### Graph conditions

Supported conditions:
- complete graph
- Watts-Strogatz with `p=0.0` (localized ring-like communication)
- Watts-Strogatz with `p=0.1` (small-world shortcuts)

Design choice:
- these topologies map directly to the proposal and separate "global visibility" from "local shortcuts" effects.

## 4) Model architecture

The model in `src/models/recurrent_gat_agent.py` combines:
- `GATv2Conv` over the visible communication channel
- `GRUCell` for temporal state update
- per-step MLP head for binary logits

Communication modes:
- `fair_1bit` (default): only previous binary actions are visible to neighbors
- `vector`: optional ablation mode with higher-dimensional visible messages

STE fairness for `fair_1bit`:
- Forward pass on the visible channel is strictly 1-bit (hard argmax of the previous-step prediction).
- Backward pass routes gradients through the soft probability of the same prediction (straight-through estimator).
- At evaluation time the channel is strictly 1-bit; the STE bridge is a training-only trick.
- Therefore neighbors never observe more than one bit per round, which matches the HST communication channel.

Per time step:
1. message passing on previous visible neighbor messages
2. concatenate current private signal
3. GRU update of latent belief
4. predict current state logits
5. construct next-round visible message (STE-binarized action in fair mode)

Design choices:
- **Recurrent state (`GRUCell`)**: keeps temporal memory instead of reprocessing full sequence each step
- **Attention message passing (`GATv2Conv`)**: allows learned neighbor weighting
- **Per-step predictions**: enables anytime evaluation, not only final horizon output
- **Fair 1-bit default channel**: matches HST communication bottleneck constraints

## 5) Training objective

`SharedSequentialLoss` averages cross-entropy across all time steps.

Design choice:
- this enforces useful representations at every horizon step, aligned with the "generalist anytime predictor" objective in the proposal.

Trade-off:
- may slightly reduce specialization for very late horizons versus a final-step-only loss, but improves consistency across time.

## 6) Evaluation and bound comparison

From trained models:
- compute `epsilon(t)` over test episodes
- fit anchored exponential decay `epsilon(t) = (0.5 - epsilon_inf) * exp(-beta * t) + epsilon_inf` (default anchor `t0`, prior at round 0)
  - optional anchor `t1`: `epsilon(t) = (epsilon(1) - epsilon_inf) * exp(-beta * (t-1)) + epsilon_inf` with empirical `epsilon(1)`
  - fit uses discrete test indices `t = 1..T`
  - primary method: `scipy.optimize.curve_fit`
  - fallback: log-linear fit when scipy fit fails
- generate `anchored_t0` learning-rate plots per condition (default **fit-anchor `t0`** for GAT fit and HST reference):
  - GAT and HST curves use prior `epsilon(0)=0.5` and share fitted `epsilon_inf`
  - HST reference decays with `beta_HST_max`; both curves span the full plotted horizon at save time
  - optional **fit-anchor `t1`**: empirical `epsilon(1)` (`plot_learning_rate_from_logs.py --fit-anchor t1`, suffix `__anchored_t1`)

Bound comparison:
- `beta_HST_max(q)` follows HST Theorem 1 in `src/hst_bound.py`
- store:
  - `beta_hst_max`
  - `beta_gap = beta_GAT - beta_HST_max`
  - `exceeds_hst_bound` (when fit succeeds)
  - `convergence_warning` (true when fitted `epsilon_inf > 0.05`)

Consensus diagnostics (test eval, per condition, both modes in `metrics.json` → `consensus`):
- **unanimous:** all agents share the same prediction at step `t`
- **majority:** strict majority (`max(count) > N/2`) agrees on one label
- episode scalars (per mode): reach rate, correct-consensus rate, wrong-only rate, correct-at-first-consensus, mean/median first consensus time
- optional time series when `save_consensus_series: true` (rates per `t` plus mean agreement fraction)

Design choice:
- keep theoretical function in one module so theory updates remain isolated.

Current implementation detail (binary symmetric signals):
- with `P(s=theta)=q` and `P(s!=theta)=1-q`,
- per-signal log-likelihood takes values `+-log(q/(1-q))`,
- HST bound is `M = 2 * sup |ell| = 2 * log(q/(1-q))`.

Reference:
- W. Huang, P. Strack, O. Tamuz, *Learning in Repeated Interactions on Networks*,
  arXiv:2112.14265, Eq. (1) and Theorem 1:
  [https://arxiv.org/pdf/2112.14265](https://arxiv.org/pdf/2112.14265)

## 7) Experiment execution

### Single run

`scripts/train.py` executes one seed / one node-count configuration across graph conditions and writes:
- local JSON artifacts (`metrics.json`, optional plots)

### Grid run

`scripts/run_grid.py` executes proposal-style sweeps over:
- seeds
- node counts
- signal qualities

It writes:
- per-run condition metrics under `grid_runs/n_{N}/q_{key}/seed_{S}/` (e.g. `q_0p55` vs `q_0p6`)
- aggregate summary with:
  - `by_setting`
  - `by_condition`
  - `by_setting_and_condition`
  - `regime_classification`

### SLURM grid (cluster)

- `bash scripts/slurm/run_batch_slurms.sh 20 fair_1bit` submits 60 tasks (5 seeds × 3 n × 4 q) with throttling
- each task runs `scripts/run_grid_task.py --task-index ...`
- `scripts/slurm/finalize_job.sh` aggregates artifacts after the array completes
- walltime is per network size (one array submit per n): 2h / 4h / 8h for n=10/100/1000 — override via `SBATCH_TIME_N10`, `SBATCH_TIME_N100`, `SBATCH_TIME_N1000`

## 8) Regime classification logic

`src/reporting.py` outputs:
- `consistent_with_equilibrium_bound`
- `empirical_counter_evidence`
- `boundary_condition_evidence`
- `insufficient_fit_quality` / `high_convergence_warning_rate` (quality gates)
- `headline_label` (single dashboard-friendly status)

Headline precedence:
1. `inconclusive` if fit success rate is low or convergence warnings are frequent
2. `empirical_counter_evidence`
3. `boundary_condition_evidence`
4. `consistent_with_equilibrium_bound`
5. `inconclusive` (fallback)

Design choice:
- precedence is intentional: if strong counter-evidence exists, it should dominate summary labeling.

## 9) Configuration philosophy

Config is merged in this order:
1. code defaults
2. YAML (`configs/default.yaml`)
3. CLI overrides

Current important defaults:
- `num_seeds: 5` (seeds `0..4` in grid runs)
- `train_episodes: 5000` (fallback; grid uses `train_episodes_per_n`: 10→5000, 100→7000, 1000→10000)
- `max_horizon: 100`
- `save_epsilon_series: true`
- `communication_mode: fair_1bit`

Design choice:
- flat config schema keeps single-run and grid-run scripts simple and easy to override from shell experiments.

## 10) Known simplifications and future upgrades

Current simplifications:
- per-fit Wald CIs are reported, but cross-seed CI use mean of per-fit SE; full hierarchical model not used
- vector mode is an ablation for channel bandwidth, not the strict HST-fair benchmark
- RGAT is not a rational-equilibrium agent, so comparisons should be interpreted as benchmark comparisons against HST assumptions, not theorem refutations
- 100 percent homophily on the ground-truth label is fixed across topologies
- synthetic Bernoulli signals only; real social-data calibration is out of scope

Recommended next upgrades:
- add confidence intervals/bootstrapping for `beta_GAT`
- add optional plot generation (`epsilon(t)`, beta-gap distributions)
- add explicit result tables per `(n, q, topology)` for publication export
