# Shipped PPO model

`ppo_1m_tardy_best.zip` (1.2 MB, sb3-contrib MaskablePPO, CPU) is the exact checkpoint behind the PPO rows in the README.
Integrity: `shasum -a 256 -c SHA256SUMS`.

* Selected by validation weighted tardiness (seeds 100000+) among checkpoints saved every 100,000 steps; best validation value 4690.3 (4 val seeds per scenario).
* Training: 1,000,000 steps, 4 envs, seed 0, train seeds 0-199, scenarios stable, breakdowns, dynamic_arrivals, hetero_full, stress, reward weights flow 0.2 / tardiness 2.0 (scale 500), 592 s on a laptop CPU.
* Command: `python scripts/train_run.py ppo_1m_tardy 1000000 0.2 2.0` (retraining is seeded but is not guaranteed to be bit-identical across machines).
* Evaluate: `factory-scheduler compare --model models/ppo_1m_tardy_best.zip --n-seeds 20`.
