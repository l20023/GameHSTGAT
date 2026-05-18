# Release Checklist

Use this checklist before sharing the repository, opening a PR, or tagging a release.

## 1) Environment and dependencies

- [ ] Python version is compatible with `pyproject.toml` (`>=3.10`)
- [ ] Dependencies install cleanly: `pip install -e .`
- [ ] Dev dependencies install if needed: `pip install -e .[dev]`

## 2) Configuration sanity

- [ ] `configs/default.yaml` uses intended defaults:
  - `wandb_project: game-theory-project`
  - `wandb_entity: GameHSTGAT`
  - `max_horizon: 50`
- [ ] W&B login is valid on the execution machine (`wandb login`)
- [ ] No local-only paths are hardcoded in configs/scripts

## 3) Quality gates

- [ ] Full test suite passes: `pytest`
- [ ] New/modified modules have tests where applicable
- [ ] No IDE linter errors on changed files

## 4) Data and artifact hygiene

- [ ] Large generated artifacts are not tracked (`artifacts/`, model checkpoints, logs)
- [ ] `.gitignore` covers local caches, secrets, and generated outputs
- [ ] No secrets committed (`.env`, keys, tokens)

## 5) Documentation

- [ ] `README.md` reflects current CLI and defaults
- [ ] Technical model documentation is up to date (`MODEL_README.md`)
- [ ] Paper-oriented summaries are up to date:
  - `PAPER_METHODS_SUMMARY.md`
  - `CAMERA_READY_METHODS.md`

## 6) Reproducibility checks (recommended)

- [ ] Run one smoke train command:
  - `python scripts/train.py --config configs/default.yaml --seed 0 --num-nodes 10`
- [ ] Run one smoke grid command:
  - `python scripts/run_grid.py --config configs/default.yaml --seeds 0 --num-nodes-list 10 --signal-quality-list 0.8`
- [ ] Confirm expected outputs exist under `artifacts/` and summary JSON is generated

## 7) Final release/PR prep

- [ ] Commit message(s) clearly explain why changes were made
- [ ] Changelog/notes updated (if used)
- [ ] Reviewer instructions include exact validation commands
