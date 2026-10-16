# Why did PPO underperform ATC? (analysis, not a proof)

Main result (test seeds, 100 instances): PPO 4250 vs ATC 3227 mean weighted tardiness; PPO is worse in all five scenarios. This note lists hypotheses, the evidence we actually have, and what would separate them. Diagnostics and the decision rule use **validation** seeds (100000+). After the decision rule was applied, the cloned and fine-tuned models were also evaluated once on the test seeds for reporting only (section "Test-set check"); no choice depends on those numbers.

## Code audit of the RL setup (what is and is not a problem)
| Item | Finding |
|---|---|
| Information | Env and policies use the same released-job information; no future arrivals/breakdowns are exposed (`environment/features.py`). |
| Train/eval separation | Train seeds 0-199, checkpoint selection on val 100000+, reporting on test 200000+. Same five scenario *configs* for train and test (different instances), so this measures in-distribution generalisation, not scenario shift. |
| Termination | `terminated` = all jobs done, `truncated` = horizon. Horizon (5000 min) was never hit in the benchmark (0 unfinished jobs in all 700 runs), so truncation logic is unit-tested but not exercised by the reported experiments. |
| Reward | Return equals -(a*weighted flow + b*weighted tardiness)/500 exactly (tested). No idling/ignore exploit. Evaluation metric is weighted tardiness only; the first run (a=b=1) optimised a flow-dominated objective. |
| Observation scaling | Fixed constants (e.g. proc/25, slack/100 clipped to +-5), no learned normalisation; slack can saturate under stress. |
| Action space | 24 slots sorted by due date. **Diagnostic:** ATC's choice is visible in 100% of val decisions; >24 candidates occur in only 0-2.8% of decisions (`data/results/rl_diagnostics.json`). Slot truncation is therefore *not* the explanation (it did drop some training samples for cloning, see E1). |
| Decision opportunities | 57-120 dispatches per episode; 12-26% have a single candidate (no choice), mean candidates 3.7-6.6. Choices are few per episode, so per-step credit is noisy but the episodes are not tiny. |
| Hyperparameters | 1M steps, lr 3e-4, n_steps 512 x 4 envs, 128x128 MLP; no search. Validation curves are noisy and flat after ~200k steps (val 4690-5530 across checkpoints). |

## Hypotheses and evidence
1. **H1 - observation/network cannot represent a strong rule.** Test: E1 (clone ATC with the same obs/network). *Evidence against:* the clone reaches 4084 vs ATC 3884 (+5.1%) on 50 val instances, seed 0 per-scenario results within -2% to +4% of ATC (`bc_finetune/seed0.json`).
2. **H2 - RL optimisation (exploration, credit assignment, noisy returns) is the bottleneck.** *Evidence for:* PPO from scratch scores 4980 on the same 50 val instances (+28% vs ATC; this checkpoint was chosen on 4 of those 10 val seeds per scenario, so slightly optimistic), and PPO fine-tuning *from the clone* (E2) gets steadily worse: val 4080-4175 after 50k steps to 4421/4487/4643 after 300k steps (3 seeds). Moving away from the cloned rule hurt, so the reward gradient was not leading to better schedules than ATC.
3. **H3 - reward/metric mismatch.** Weak support: the tardiness-heavy reward (0.2/2.0) beat equal weights on validation (4690 vs 4823 on 20 instances, one seed each; not conclusive). Still far from ATC under both.
4. **H4 - too little training/variance.** Not excluded: one seed per from-scratch configuration, 1M steps. E2 shows the trend under 300k steps from a good start is downward, which argues against "just train longer" from that start.
5. **H5 - the task has little room above ATC for a myopic non-delay policy.** Unknown. CP-SAT offline gap is 40% in weighted tardiness on 10-job static instances (26 vs 43), but that includes future-arrival knowledge and idling, which this action space cannot use.

## Experiments run (budget: ~25 CPU-minutes, `scripts/rl_experiment_bc.py`)
* E1: behaviour cloning of ATC, 25,276 samples from 60 train seeds x 5 scenarios, 30 epochs.
* E2: PPO fine-tune from E1, lr 5e-5, clip 0.1, ent_coef 0, 300k steps, seeds 0/1/2, val every 50k.
* Summary (`python scripts/summarize_bc_experiment.py`, saved in `data/results/experiments/bc_finetune/summary.txt`), 50 val instances, ATC = 3884:

| policy | per-seed val | mean | vs ATC (paired 95% CI) |
|---|---|---|---|
| clone of ATC | 3992 / 4209 / 4051 | 4084 | +200 [+24, +376], Wilcoxon p=0.077 |
| PPO fine-tuned from clone | 4421 / 4487 / 4643 | 4517 | +633 [+249, +1017], p=4e-9 |
| PPO from scratch (1M) | one run | 4980 | n/a |

### Test-set check (reporting only; `scripts/rl_experiment_bc_test.py`)
100 test instances (5 scenarios x 20 seeds), ATC = 3227, averaged over the 3 training seeds, paired vs ATC (pairs treated as independent):

| policy | per-seed test mean | mean | vs ATC (95% CI, Wilcoxon p) |
|---|---|---|---|
| clone of ATC | 3409 / 3376 / 3437 | 3408 | +181 [+92, +270] (+5.6%), p = 1.8e-5 |
| PPO fine-tuned from clone | 3531 / 3555 / 3750 | 3612 | +385 [+242, +528] (+11.9%), p = 4.9e-12 |
| PPO from scratch (1M) | one run | 4250 | +1023 (+31.7%, pooled CI [747, 1299]) |

The rebuilt clone reproduced the saved validation scores exactly (3991.6 / 4208.9 / 4051.3), so the cloning step is deterministic. On test, fine-tuning again moves away from ATC, though by less than on validation (+11.9% vs +16%), and even the pure clone is significantly worse than the rule it imitates.

Caveats: val-based, 3 seeds, one reward setting for E2, the critic is untrained at the start of fine-tuning (a likely contributor to early drift), and the clone was not DAgger-corrected.

## Decision rule (defined before the runs; not externally pre-registered)
Further PPO work is justified only if PPO fine-tuned from the clone matches or beats ATC on validation (mean over 3 seeds, paired CI of the difference including 0 or below), after which a *single* test-set confirmation would be run. **Outcome: not met** (+16% worse, CI excludes 0). Recommendation: stop PPO tuning on this environment. Remaining ideas, each needing its own justification: pretrain the critic before fine-tuning; allow idling/lookahead actions; learn a residual on top of ATC; harder scenarios where ATC is demonstrably weak. Do not claim RL is superior here: the evidence says it is not.
