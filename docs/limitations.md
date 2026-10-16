# Limitations

* Synthetic, simplified twin; no industrial validation. Exponential failures, preempt-resume, no transport, no operators, no machine-specific setup matrices.
* Non-delay dispatching only; policies cannot idle a machine on purpose (a known cost vs. optimal schedules).
* Candidate truncation to 24 slots; fixed max 8 machines.
* PPO: 1M steps, one seed per configuration, two reward settings tried, no hyperparameter search. A longer or tuned run might do better; the result is "this recipe", not "RL cannot".
* ATC (k=2) is untuned but is a strong, well-known rule; it is the real bar.
* CP-SAT offline results used 10-job instances only; rolling CP-SAT used 0.1 s per solve on 8 seeds.
