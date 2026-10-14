//! Membership-inference signal primitives for provenancepack.
//!
//! Three signals, each one a per-document scalar in [0, 1] where higher
//! means "more likely in the training set."

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
}
