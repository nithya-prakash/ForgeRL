# Evaluation protocol

* Seeds: train 0.., val 100000.., test 200000... A (scenario, seed) pair gives an identical `Instance` (jobs, arrivals, breakdown realisation) to every policy. Policy-dependent: only event timing that follows from the schedule itself.
* Every schedule is validated before its metrics are saved. Results CSV + `.meta.json` (scenario configs, seeds, model path) are written together.
* Statistics: mean, std, 95% t-interval over seeds; paired differences with t-interval, Wilcoxon signed-rank p, Cohen's dz (`evaluation/stats.py`). The PPO comparison is against the strongest heuristic per scenario, chosen on the same test data, which if anything favours the heuristic. No multiple-comparison correction.
* The PPO model reported was picked by validation score among the runs tried (two 1M-step runs, differing only in reward weights). Test seeds were not used for any selection.
