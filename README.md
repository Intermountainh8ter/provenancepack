# provenancepack

**Fine-tuning data provenance via membership-inference.** Three
state-of-the-art signals (loss-ratio to paraphrase, min-k% token
log-probability, zlib-ratio) with proper controls against a known-out
document bank, pre-registered thresholds, and a `likely_in` verdict.

![status: initial scaffold](https://img.shields.io/badge/status-initial%20scaffold-lightgrey)
![language: Rust](https://img.shields.io/badge/language-Rust-blue)
![license: MIT](https://img.shields.io/badge/license-MIT-blue)

## Why

Membership-inference papers exist in plenty; a usable auditor tool with
pre-registered thresholds and false-positive controls does not.
Lawsuits over training data (NYT v OpenAI, etc.) rely on hand-crafted
evidence; a reusable, documented tool would be cited.

## What is in this scaffold

- `loss_ratio_signal(original_loss, paraphrase_loss)`
- `min_k_signal(token_log_probs, k_frac)` with the Carlini-style
  low-probability tail.
- `zlib_ratio_signal(zlib_entropy_bits, nll_sum_nats)`
- `likely_in(signal, threshold)` single-condition shortcut.
- `empirical_p(target, bank)` upper-tail p-value against a known-out
  bank: `(1 + #{bank >= target}) / (1 + N)`.
- `MIN_BANK_SIZE = 100` constant (pre-reg §4).
- `SignalReport` + `score_signal(signal, bank, threshold, p_cutoff)`:
  §3's two-part flagging rule (signal >= threshold AND p < cutoff),
  returning `LikelyIn` / `NotIn` / `BankTooSmall` (bank below 100).
- `SummaryReport` + `summarize_three(...)`: §5's agreement count
  across the three signals, refusing to average them into a composite.
- CLI: one subcommand per signal (back-compat), plus `score-signal`
  and `summary`. Exit codes on `summary`: 0 none / 1 one / 2 two or
  three / 3 any bank too small.

Not in this commit (openly tracked):

- `provenancepack probe --model <endpoint> --document <path>` end-to-end
  driver that computes the three signals from a document.
- Known-out document bank (public text known to be excluded from the
  model's training window).

## License

MIT. See `LICENSE`.
