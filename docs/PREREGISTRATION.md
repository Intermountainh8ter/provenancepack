# provenancepack pre-registration

Locked before the first live-probe result is committed. Append-only.

## 1. Scope

Models under probe are OpenAI-compatible endpoints that expose
per-token log-probabilities (or a `/v1/completions`-style endpoint
whose log-probs can be inferred). Models that return only text are
out of scope for the min-k signal.

Target documents are text whose membership in the training set is the
question. The audit always runs against a **known-out control bank**
of documents the model's publisher has stated are not in training (or
documents published after the training cutoff with public timestamp
evidence); any absolute claim is calibrated against that bank.

## 2. The three signals

**Loss-ratio.** `paraphrase_loss / original_loss`, clipped to `[0, 2]`
and rescaled to `[0, 1]`. Membership increases the ratio because the
model has specifically memorized the original wording.

**Min-k% token log-probability.** Mean log-probability of the lowest-
probability `k`% of tokens in the document; `exp()`-rescaled to
`[0, 1]`. Membership raises this quantity because training lifted the
tail. Default `k = 20%` (Shi et al., 2024).

**Zlib ratio.** `nll_bits / zlib_entropy_bits`, flipped and clipped so
membership → low ratio → high signal (Carlini et al.).

## 3. Flagging rule

A document is **likely in** when the per-signal score exceeds
`threshold` AND the paired-resample p-value against the known-out
control bank is < 0.05.

Default thresholds, locked:

| signal | threshold |
|--------|-----------|
| loss-ratio | 0.75 |
| min-k | 0.75 |
| zlib | 0.75 |

A document is **highly likely in** when at least two of the three
signals clear the threshold. One signal alone is a flag; two agreeing
is a strong flag; all three agreeing is the strongest flag this repo
publishes.

## 4. Known-out control bank

The audit refuses to publish any absolute claim without pairing the
target document against ≥ 100 known-out documents drawn from the same
distribution (length, genre, source). The paired resample under the
null hypothesis "signal is the same for in and out" gives the
false-positive-controlled p-value.

The control bank is committed in `fixtures/known_out/`. Each document
ships with its provenance evidence.

## 5. What this repo refuses to do

- No claim about a specific private document the plaintiff did not
  authorize the repo to publish.
- No "composite" signal that averages the three signals into one
  number. The three are reported separately and the agreement count is
  the published summary.
- No numbers in README or leaderboard not produced by committed
  `results/*.json`.
- No disclosure of a specific break before a 90-day disclosure window
  if the finding would reveal a specific private dataset by name.

## 6. Threats to validity

- **Paraphrase quality.** A bad paraphrase makes the loss-ratio
  signal meaningless. The paraphrase set per fixture ships with a
  manual review pass.
- **Distribution drift of the known-out bank.** Known-out documents
  drawn from a different distribution than the target produce a false
  positive. We report the drift metric (length and vocabulary KL)
  between target and bank for every result.
- **Model that refuses continuation.** Chat-tuned models that refuse
  to continue cannot have min-k measured; we fall back to loss-ratio
  and zlib-ratio and flag the result as "two-signal only."
