# Pre-registration additions

## 2026-11-27T23:57:04Z canary-bank-fpr proposal:c241b7c0aec286c7f58e59314797adcc34a60f1241fdd227786f0b88613f22f1

Hypothesis: for a never-trained responder with per-item chance hit rate 0.05 and a flag rule of at least 20% of canaries hit, the simulated false-accusation rate falls from above 0.10 at bank size 10 to below 0.01 at bank size 100 (MIN_BANK_SIZE), while an always-answer degenerate responder is accused at rate 1.0 at every size.

Metric: /false_accusation_rate_by_bank_size/100 in results/745-canary-bank-fpr.json. Expected: below 0.01 (range 0.0 to 0.01), and above 0.10 at size 10.

Falsifier: a simulated rate at size 100 of 0.01 or more, or a rate at size 10 of 0.10 or less.

Method: model-free Monte Carlo, own xorshift64* PRNG, seed 7, 20000 trials per size, sizes 10,25,50,100,200. A trial flags when hits >= ceil(threshold_frac * n). Both parts of the hypothesis are registered; either failing refutes it and the result ships as stated.

Outcome (appended after the run): rate at size 100 is 0.0 (below 0.01, holds); rate at size 10 is 0.08685 (not above 0.10), so the falsifier's second clause fired and the hypothesis is refuted as stated. Exact binomial tail at n=10 is 1 - 0.95^10 - 10*0.05*0.95^9, which the simulation matches.
