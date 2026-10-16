//! Membership-inference signal primitives for provenancepack.
//!
//! Three signals, each one a per-document scalar in [0, 1] where higher
//! means "more likely in the training set." Each signal is paired with
//! a known-out control bank (pre-reg §4) and reports a per-document
//! verdict against both a signal threshold AND a paired-resample
//! p-value < 0.05 (pre-reg §3). The repo refuses to publish an
//! absolute claim when the bank has fewer than 100 documents (§4).

use serde::{Deserialize, Serialize};

/// Loss-ratio: `loss(paraphrase) / loss(original)`. Values > 1 mean the
/// model is specifically fluent on the original wording, a membership
/// signal. Clipped to [0, 2] and rescaled to [0, 1].
pub fn loss_ratio_signal(original_loss: f64, paraphrase_loss: f64) -> f64 {
    if original_loss <= 0.0 {
        return 1.0;
    }
    let r = paraphrase_loss / original_loss;
    (r.clamp(0.0, 2.0) / 2.0).clamp(0.0, 1.0)
}

/// Min-k% token log-probability: the mean log-probability of the lowest-
/// probability `k`% of tokens. Higher (less negative) is a membership
/// signal. The scalar signal here is `exp(min_k_mean_logp)` so it sits
/// in [0, 1].
pub fn min_k_signal(token_log_probs: &[f64], k_frac: f64) -> f64 {
    assert!((0.0..=1.0).contains(&k_frac));
    assert!(!token_log_probs.is_empty());
    let mut lp = token_log_probs.to_vec();
    lp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let k = ((lp.len() as f64 * k_frac).ceil() as usize).max(1);
    let mean = lp.iter().take(k).copied().sum::<f64>() / k as f64;
    mean.exp().clamp(0.0, 1.0)
}

/// Zlib-ratio signal: `zlib_entropy(text) / nll_sum(text)`. The original
/// paper (Carlini et al.) uses this ratio where membership corresponds
/// to a low ratio (model's NLL is small compared to zlib's compressed
/// length). We flip-and-rescale so the signal is in [0, 1] with higher =
/// membership.
pub fn zlib_ratio_signal(zlib_entropy_bits: f64, nll_sum_nats: f64) -> f64 {
    if zlib_entropy_bits <= 0.0 {
        return 0.0;
    }
    // NLL is in nats; convert to bits for comparison.
    let nll_bits = nll_sum_nats / 2f64.ln();
    let ratio = nll_bits / zlib_entropy_bits;
    // Membership = low ratio. Signal 1 when ratio -> 0, 0 when ratio >= 1.
    (1.0 - ratio.clamp(0.0, 1.0)).clamp(0.0, 1.0)
}

/// Pre-registered: a document is **likely in** when the signal exceeds
/// `threshold`.
pub fn likely_in(signal: f64, threshold: f64) -> bool {
    signal >= threshold
}

/// Pre-registered minimum known-out control bank size (§4).
pub const MIN_BANK_SIZE: usize = 100;

/// Empirical p-value against a known-out control bank. H0: "the target
/// signal is drawn from the same distribution as the bank." The
/// (one-sided, upper-tail) empirical p-value is `(1 + count(bank >=
/// target)) / (1 + len(bank))`. Pre-reg §4's paired-resample idea
/// reduces to this for an iid bank; the +1s are the standard
/// finite-sample correction.
pub fn empirical_p(target_signal: f64, bank_signals: &[f64]) -> f64 {
    if bank_signals.is_empty() {
        return f64::NAN;
    }
    let as_or_above = bank_signals
        .iter()
        .filter(|v| **v >= target_signal)
        .count();
    (1.0 + as_or_above as f64) / (1.0 + bank_signals.len() as f64)
}

/// Per-signal verdict against the pre-reg §3 flagging rule.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SignalVerdict {
    /// Signal cleared the threshold AND the paired-resample p < p_cutoff.
    LikelyIn,
    /// At least one of the two conditions failed.
    NotIn,
    /// Bank was below the pre-reg §4 minimum. No verdict is published.
    BankTooSmall,
}

/// Per-signal report (both the threshold check and the bank p-value).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SignalReport {
    pub signal: f64,
    pub threshold: f64,
    pub bank_n: usize,
    pub p_value: f64,
    pub p_cutoff: f64,
    pub verdict: SignalVerdict,
}

/// Score one signal against a control bank with the pre-reg's two-part
/// flagging rule. Returns BankTooSmall when the bank is thin (§4).
pub fn score_signal(
    signal: f64,
    bank_signals: &[f64],
    threshold: f64,
    p_cutoff: f64,
) -> SignalReport {
    let bank_n = bank_signals.len();
    if bank_n < MIN_BANK_SIZE {
        return SignalReport {
            signal,
            threshold,
            bank_n,
            p_value: f64::NAN,
            p_cutoff,
            verdict: SignalVerdict::BankTooSmall,
        };
    }
    let p = empirical_p(signal, bank_signals);
    let verdict = if likely_in(signal, threshold) && p < p_cutoff {
        SignalVerdict::LikelyIn
    } else {
        SignalVerdict::NotIn
    };
    SignalReport {
        signal,
        threshold,
        bank_n,
        p_value: p,
        p_cutoff,
        verdict,
    }
}

/// Full three-signal summary. Pre-reg §5 forbids averaging the signals
/// into a composite; the "agreement count" is the published summary.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SummaryReport {
    pub loss_ratio: SignalReport,
    pub min_k: SignalReport,
    pub zlib: SignalReport,
    /// Count of signals whose verdict is LikelyIn. 0 = no flag,
    /// 1 = a flag, 2 = a strong flag, 3 = the strongest flag this repo
    /// publishes.
    pub n_signals_flagged: usize,
    /// Count of signals disqualified because their bank was too small
    /// (§4). When this is non-zero, no absolute claim is published.
    pub n_signals_bank_too_small: usize,
}

/// Combine the three per-signal reports under the pre-reg §5 "agreement
/// count" rule; refuses to compute an averaged composite.
pub fn summarize_three(
    loss_ratio: SignalReport,
    min_k: SignalReport,
    zlib: SignalReport,
) -> SummaryReport {
    let rs = [&loss_ratio, &min_k, &zlib];
    let n_flagged = rs
        .iter()
        .filter(|r| r.verdict == SignalVerdict::LikelyIn)
        .count();
    let n_bank_too_small = rs
        .iter()
        .filter(|r| r.verdict == SignalVerdict::BankTooSmall)
        .count();
    SummaryReport {
        loss_ratio,
        min_k,
        zlib,
        n_signals_flagged: n_flagged,
        n_signals_bank_too_small: n_bank_too_small,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loss_ratio_high_when_original_is_lower() {
        // Model confident on original, high loss on paraphrase -> higher signal
        assert!(loss_ratio_signal(0.3, 1.5) > 0.9);
        // Equal losses -> middle signal
        let eq = loss_ratio_signal(1.0, 1.0);
        assert!((eq - 0.5).abs() < 1e-12);
    }

    #[test]
    fn min_k_signal_bounded_in_unit_interval() {
        let lp = vec![-0.1, -0.2, -5.0, -0.4, -3.0, -0.5];
        let s = min_k_signal(&lp, 0.2);
        assert!((0.0..=1.0).contains(&s));
    }

    #[test]
    fn zlib_signal_high_when_nll_small() {
        // nll ~ 0 vs zlib positive -> ratio ~ 0 -> signal ~ 1
        assert!(zlib_ratio_signal(100.0, 1.0) > 0.9);
        // nll >> zlib -> ratio capped at 1 -> signal 0
        assert!(zlib_ratio_signal(10.0, 1000.0) < 0.01);
    }

    #[test]
    fn likely_in_requires_signal_at_or_above_threshold() {
        assert!(likely_in(0.85, 0.8));
        assert!(!likely_in(0.75, 0.8));
    }

    #[test]
    fn empirical_p_is_small_when_target_sits_above_the_bank() {
        let bank: Vec<f64> = (0..100).map(|i| i as f64 / 1000.0).collect();
        // Target well above the bank (bank max ≈ 0.099): p ≈ 1/101.
        let p = empirical_p(0.9, &bank);
        assert!(p < 0.02, "p={p}");
    }

    #[test]
    fn empirical_p_is_near_half_when_target_is_at_the_median() {
        let bank: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let p = empirical_p(0.5, &bank);
        assert!(0.45 < p && p < 0.55, "p={p}");
    }

    #[test]
    fn score_signal_bank_too_small_when_fewer_than_100_rows() {
        let bank: Vec<f64> = (0..50).map(|i| i as f64 / 100.0).collect();
        let r = score_signal(0.9, &bank, 0.75, 0.05);
        assert_eq!(r.verdict, SignalVerdict::BankTooSmall);
        assert!(r.p_value.is_nan());
    }

    #[test]
    fn score_signal_likely_in_requires_both_threshold_and_p() {
        let bank: Vec<f64> = (0..100).map(|i| i as f64 / 1000.0).collect();
        // High signal (0.9), tight bank -> likely in.
        let r = score_signal(0.9, &bank, 0.75, 0.05);
        assert_eq!(r.verdict, SignalVerdict::LikelyIn);
        // Signal below threshold -> NotIn even if p is small.
        let r2 = score_signal(0.60, &bank, 0.75, 0.05);
        assert_eq!(r2.verdict, SignalVerdict::NotIn);
    }

    #[test]
    fn score_signal_rejects_when_p_is_too_large() {
        // Bank skewed high, target not special.
        let bank: Vec<f64> = (0..100).map(|_| 0.95).collect();
        let r = score_signal(0.85, &bank, 0.75, 0.05);
        assert_eq!(r.verdict, SignalVerdict::NotIn);
        assert!(r.p_value > 0.5, "p={}", r.p_value);
    }

    fn mk_report(verdict: SignalVerdict) -> SignalReport {
        SignalReport {
            signal: 0.8,
            threshold: 0.75,
            bank_n: 120,
            p_value: 0.01,
            p_cutoff: 0.05,
            verdict,
        }
    }

    #[test]
    fn summary_counts_signal_agreement() {
        let s = summarize_three(
            mk_report(SignalVerdict::LikelyIn),
            mk_report(SignalVerdict::LikelyIn),
            mk_report(SignalVerdict::NotIn),
        );
        assert_eq!(s.n_signals_flagged, 2);
        assert_eq!(s.n_signals_bank_too_small, 0);
    }

    #[test]
    fn summary_counts_thin_banks_separately() {
        let s = summarize_three(
            mk_report(SignalVerdict::LikelyIn),
            mk_report(SignalVerdict::BankTooSmall),
            mk_report(SignalVerdict::BankTooSmall),
        );
        assert_eq!(s.n_signals_flagged, 1);
        assert_eq!(s.n_signals_bank_too_small, 2);
    }

    #[test]
    fn summary_round_trips_through_json() {
        let s = summarize_three(
            mk_report(SignalVerdict::LikelyIn),
            mk_report(SignalVerdict::NotIn),
            mk_report(SignalVerdict::LikelyIn),
        );
        let j = serde_json::to_string(&s).unwrap();
        let s2: SummaryReport = serde_json::from_str(&j).unwrap();
        assert_eq!(s, s2);
    }
}
