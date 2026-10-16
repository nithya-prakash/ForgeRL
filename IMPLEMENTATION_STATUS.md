# Implementation status (audit pass, 2026-10-16)

## Completed
Simulator + validator, 5 scenarios, 6 rules, static/rolling CP-SAT, Gym env with masking, MaskablePPO + MLflow, benchmark/stats/sweeps, 6-page dashboard, CLI, Makefile, Dockerfile/compose, CI workflow, docs, RL failure analysis + BC/fine-tune experiment.

## Audit findings and fixes
* README claimed Wilcoxon p<0.001 in all scenarios: false (breakdowns 0.0019, hetero 0.0008). Corrected; all stats now quoted from recomputation.
* Dashboard (viewed in a browser): fixed an unreadable mixed-unit MLflow chart, red "idle" bars that looked like downtime, raw JSON config dump, `174.000000` formatting, confusing CP-SAT table, no overall ranking (stress dominated the chart), unlabeled pooled paired stats, robustness x-axis label, empty-state handling. Added `?page=` deep links.
* Dashboard tests previously did not test saved results (the test pointed at the real folder). Now `FACTORY_RESULTS_DIR` is honoured, tests assert the dashboard shows exactly the policies in a freshly written CSV and the empty state.
* Linux lockfile pulled multi-GB CUDA torch (contradicting the "CPU-friendly" claim). `pyproject.toml` now pins the CPU wheel index for Linux; `uv.lock` regenerated (macOS torch version unchanged).
* Dockerfile no longer needs ghcr.io (the pull timed out); installs uv with pip.
* Not changed: historical experiment results (originals kept; `repro_compare.csv` is the audit re-run).

## Commands run and outcomes (all executed)
* Reproduction: `factory-scheduler compare --model artifacts/models/ppo_1m_tardy/best.zip --n-seeds 20 --name repro_compare` (the same checkpoint now ships as `models/ppo_1m_tardy_best.zip`) -> all deterministic columns identical to `compare.csv` (700 rows, max abs diff 0); `heuristics.csv` rule rows identical. CIs, Wilcoxon p and % differences recomputed independently with scipy: match. CP-SAT: 100/100 OPTIMAL, max gap 0.
* Not re-run: the 1M-step training itself (the shipped checkpoint is used instead). The wall-clock `compare_rolling.csv` is kept as history; the deterministic-budget replacement is below.
* `uv lock --check`, then after the torch change `uv lock`: ok.
* Clean checkout (`git checkout-index` to a temp dir): `uv sync --frozen`, `make install`, `pytest` -> 43 passed (118 s cold), CLI benchmark ran. The macOS `.pth` hidden-flag problem did **not** reproduce in the fresh venv; it did occur repeatedly in the working venv after `uv` rebuilt the editable install, and `chflags nohidden` fixes it (in `make install`). Root cause not identified.
* In the working repo (latest, after the reproducibility pass): `pytest` -> 47 passed; `ruff check`, `ruff format --check`, `mypy src` clean.
* Dashboard: started with `streamlit run`, all 6 pages inspected in the built-in browser and via real headless-Chrome screenshots (`scripts/capture_screenshots.py`); phone-width capture of the Gantt page reviewed. Other pages were not reviewed at phone width.
* RL: `scripts/rl_diagnostics.py`, `scripts/rl_experiment_bc.py 0|1|2 300000`, `scripts/summarize_bc_experiment.py` (see `docs/rl_failure_analysis.md`).

## Reproducibility pass (after the first push)
* Shipped `models/ppo_1m_tardy_best.zip` (+ checksum, README, test). `compare --model models/ppo_1m_tardy_best.zip --n-seeds 20` and `sweep load|breakdown|arrival --model ...` re-run: 700 comparison rows and all 160-200 sweep rows per file matched the saved results exactly (max abs diff 0 on every deterministic column). PPO is now in the breakdown and arrival sweeps too.
* `compare` now picks the reference rule on validation seeds (`--reference val`, default) and writes `compare_ppo_vs_ref.csv` with Holm-adjusted p-values and a pooled row. Reference = ATC in all 5 scenarios, as before. The older `compare_ppo_vs_best.csv` is kept.
* Rolling CP-SAT: added `--rolling-det-time` (CP-SAT deterministic time). Two independent runs (`compare_rolling_det_a/b.csv`, 8 seeds x 5 scenarios x 8 policies = 320 rows) were identical on every deterministic column. Result 4282 vs ATC 3066 (6th of 8), ~380 ms/decision; the older wall-clock run (4810) is kept and labelled.
* `scripts/rl_experiment_bc_test.py`: rebuilt ATC clones reproduce the saved validation scores exactly; test-set reporting: clone 3408 (+5.6% vs ATC), fine-tuned 3612 (+11.9%).
* Pending in this pass: dashboard switch to `compare_ppo_vs_ref.csv`, unit tests for Holm and the reference table, full lint/mypy/pytest run, commit and push.

* Final audit found a regression from the LICENSE commit: `pyproject.toml` declares `license-files = ["LICENSE"]` but the Dockerfile did not copy `LICENSE`, so `docker build` failed on the pushed `main` (CI does not build the image, so it stayed green). Fixed (`COPY ... LICENSE`), rebuilt (exit 0, 2.29 GB, license metadata `MIT` in the image), `docker compose up --build` -> dashboard and MLflow health 200, Policy Comparison page renders 5 tables in the container.
* Final run on the working tree: `uv lock --check`, Ruff (lint+format), mypy (31 files), pytest 47 passed, CLI smoke, model checksum OK, screenshots recaptured (6 pages), demo GIF regenerated.

## Not verified
* **Docker (now verified)**: the first build failed pulling `ghcr.io` (timeout); the second failed with a DNS error fetching the torch CPU wheel from `download.pytorch.org` inside the Docker VM; the third succeeded (82 s final stage after a slow torch download). Image 2.29 GB, `torch 2.14.1+cpu`, container health endpoint 200, AppTest of the Policy Comparison page inside the container: no exception, 5 tables. `docker compose up -d dashboard` -> health 200; `docker compose config` valid. The network flakiness is environmental, not a code fault, but it means the build is sometimes slow or needs a retry.
* **GitHub Actions**: first run on `main` (commit e0d0891) passed all steps on ubuntu-latest, including `uv lock --check`, `uv sync --frozen` with the CPU torch wheel, Ruff, mypy, 43 tests and the CLI smoke step.
* `make demo`: not run.
* Release-prep pass: Dockerfile reordered (deps layer before source), rebuilt OK (2.29 GB); `docker compose up` -> dashboard and MLflow both health 200; `make docker` added (touches `mlflow.db` so Docker does not create a directory). CI gained `uv lock --check` and a CLI smoke step (both passed in the first GitHub run).

## Known issues / notes
* Package author metadata in `pyproject.toml` is the maintainer's git identity; confirm before publishing.
* Reference rule is now chosen on validation seeds and per-scenario p-values are Holm-adjusted (`docs/evaluation_protocol.md`); the pooled row treats instances as independent pairs.
* The only model weights tracked are `models/ppo_1m_tardy_best.zip` (1.2 MB, sha256 in `models/SHA256SUMS`); other checkpoints, the MLflow DB and logs are ignored.

## Next
1. Run the compose `mlflow` service once.
2. Push to a scratch branch to see CI run (needs your go-ahead to commit/push).
3. Only if new evidence appears (see decision rule in `docs/rl_failure_analysis.md`), continue RL work; otherwise stop at the current negative result.
