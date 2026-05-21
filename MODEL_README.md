# Model README: RGAT Social Learning System

This document explains what the model does, how data flows through the system, and why key design decisions were made.

## 1) Research goal

The project studies whether a recurrent Graph Attention Network (RGAT) can match or exceed the social-learning speed bound discussed in Huang, Strack, Tamuz (HST).

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
- fit exponential decay `epsilon(t) = alpha * exp(-beta * t) + epsilon_inf`
  - primary method: `scipy.optimize.curve_fit`
  - fallback: log-linear fit when scipy fit fails
- generate two learning-rate plot variants per condition:
  - `free_alpha`: baseline visualization where GAT/HST share fitted `(alpha, epsilon_inf)`
  - `anchored_t1`: HST reference anchored at empirical `epsilon(1)`, then decays with `beta_HST_max`

Bound comparison:
- `beta_HST_max(q)` follows HST Theorem 1 in `src/hst_bound.py`
- store:
  - `beta_hst_max`
  - `beta_gap = beta_GAT - beta_HST_max`
  - `exceeds_hst_bound` (when fit succeeds)

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

`scripts/train.py` executes one seed / one node-count configuration across graph conditions and logs:
- local JSON artifacts
- W&B run metrics

### Grid run

`scripts/run_grid.py` executes proposal-style sweeps over:
- seeds
- node counts
- signal qualities

It writes:
- per-run condition metrics
- aggregate summary with:
  - `by_setting`
  - `by_condition`
  - `by_setting_and_condition`
  - `regime_classification`

## 8) Regime classification logic

`src/reporting.py` outputs:
- `supports_information_theoretic_limit`
- `empirical_counter_evidence`
- `boundary_condition_evidence`
- `headline_label` (single dashboard-friendly status)

Headline precedence:
1. `empirical_counter_evidence`
2. `boundary_condition_evidence`
3. `supports_information_theoretic_limit`
4. `inconclusive`

Design choice:
- precedence is intentional: if strong counter-evidence exists, it should dominate summary labeling.

## 9) Configuration philosophy

Config is merged in this order:
1. code defaults
2. YAML (`configs/default.yaml`)
3. CLI overrides

Current important defaults:
- `max_horizon: 100`
- `wandb_project: game-theory-project`
- `wandb_entity: GameHSTGAT`
- `communication_mode: fair_1bit`

Design choice:
- flat config schema keeps single-run and grid-run scripts simple and easy to override from shell experiments.

## 10) Known simplifications and future upgrades

Current simplifications:
- no uncertainty intervals around beta estimates in summary
- no automatic plotting in core pipeline

Recommended next upgrades:
- add confidence intervals/bootstrapping for `beta_GAT`
- add optional plot generation (`epsilon(t)`, beta-gap distributions)
- add explicit result tables per `(n, q, topology)` for publication export
