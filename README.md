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
- `likely_in(signal, threshold)` verdict.
- `provenancepack` CLI with one subcommand per signal.
- `docs/PREREGISTRATION.md` with the thresholds and the known-out
  control bank requirement.

Not in this commit (openly tracked):

- `provenancepack probe --model <endpoint> --document <path>` end-to-end
  driver that computes the three signals from a document.
- Known-out document bank (public text known to be excluded from the
  model's training window).
- Paired-resample FPR calibration.

## License

MIT. See `LICENSE`.
