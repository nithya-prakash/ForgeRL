# RL design

* Algorithm: `sb3_contrib.MaskablePPO` (the Stable-Baselines3 team's package; plain SB3 PPO has no action masking). CPU, MLP 128x128.
* Step = one dispatch. Action = one of 24 slots (feasible (op, machine) pairs sorted by due date, job, machine). Invalid slots are masked; if unmasked use happens anyway the env does nothing, applies a small penalty and truncates after 20 repeats. Limitation: with >24 feasible pairs the extra pairs (latest due dates) are invisible.
* Observation (`environment/features.py`): global state, 8 machine slots (status, eligible ready work), 24 candidate slots (processing, setup, slack, priority, wait, ops left, work left, speed ratio). No future information.
* Reward: `r = -(cost(t) - cost(t_prev)) / 500`, `cost = a*sum w_j*flow_j + b*sum w_j*overdue_j`, accrued up to completion or the episode end. The return equals the negative weighted flow + weighted tardiness exactly (unit-tested), so idling or ignoring a job cannot improve it, and truncation still charges unfinished jobs. Makespan and cost are not in the reward; they are evaluated independently.
* Isolation: train seeds 0.., validation 100000.. (checkpoint selection only), test 200000.. (reporting only).
* Convergence of training reward is not evidence of scheduling quality; only the held-out comparison is.
