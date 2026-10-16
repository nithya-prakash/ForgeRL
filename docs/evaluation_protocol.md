# Evaluation protocol

* Seeds: train 0.., val 100000.., test 200000... A (scenario, seed) pair gives an identical `Instance` (jobs, arrivals, breakdown realisation) to every policy. Policy-dependent: only event timing that follows from the schedule itself.
* Every schedule is validated before its metrics are saved. Results CSV + `.meta.json` (scenario configs, seeds, model path) are written together.
* Statistics: mean, std, 95% t-interval over seeds; paired differences with t-interval, Wilcoxon signed-rank p, Cohen's dz (`evaluation/stats.py`). The PPO comparison is against the best rule per scenario **chosen on validation seeds** (`compare --reference val`, default; `--reference test` reproduces the older test-data selection). Per-scenario Wilcoxon p-values are Holm-adjusted across scenarios; a pooled row over all pairs treats instances as independent pairs and reports a CI only.
* The PPO model reported was picked by validation score among the runs tried (two 1M-step runs, differing only in reward weights). Test seeds were not used for any selection. The ATC-clone and fine-tuned models were evaluated once on test seeds for reporting only (`scripts/rl_experiment_bc_test.py`).
