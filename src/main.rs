//! provenancepack CLI — scaffold.

use clap::{Parser, Subcommand};
use provenancepack::{
    likely_in, loss_ratio_signal, min_k_signal, score_signal, summarize_three, zlib_ratio_signal,
    SignalReport,
};

#[derive(Parser)]
#[command(
    name = "provenancepack",
    version,
    about = "Training-data provenance via membership inference"
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Compute one of the three signals from pre-computed stats.
    LossRatio {
        #[arg(long)]
        original_loss: f64,
        #[arg(long)]
        paraphrase_loss: f64,
        #[arg(long, default_value_t = 0.8)]
        threshold: f64,
    },
    MinK {
        /// Path to a JSON file: {"token_log_probs": [f64, ...]}.
        token_log_probs: std::path::PathBuf,
        #[arg(long, default_value_t = 0.2)]
        k_frac: f64,
        #[arg(long, default_value_t = 0.8)]
        threshold: f64,
    },
    Zlib {
        #[arg(long)]
        zlib_entropy_bits: f64,
        #[arg(long)]
        nll_sum_nats: f64,
        #[arg(long, default_value_t = 0.8)]
        threshold: f64,
    },
    /// Score one signal against a known-out control bank per pre-reg
    /// §3 flagging rule. Bank is a JSON file: `{"signals": [f64, ...]}`.
    /// Prints the SignalReport. Exit codes: 0 LikelyIn, 1 NotIn,
    /// 2 BankTooSmall.
    ScoreSignal {
        /// The signal value (in [0,1]) for the target document.
        #[arg(long)]
        signal: f64,
        /// Path to a JSON file of control-bank signals.
        #[arg(long)]
        bank: std::path::PathBuf,
        #[arg(long, default_value_t = 0.75)]
        threshold: f64,
        #[arg(long, default_value_t = 0.05)]
        p_cutoff: f64,
    },
    /// Combine three per-signal reports under the pre-reg §5
    /// agreement-count rule. Input bundle: `{"loss_ratio": SignalReport,
    /// "min_k": SignalReport, "zlib": SignalReport}`. Exit codes:
    /// 0 no signals flagged, 1 one signal, 2 two or three signals,
    /// 3 any bank was too small (no absolute claim published).
    Summary {
        bundle: std::path::PathBuf,
    },
    Version,
}

#[derive(serde::Deserialize)]
struct Lps {
    token_log_probs: Vec<f64>,
}

#[derive(serde::Deserialize)]
struct Bank {
    signals: Vec<f64>,
}

#[derive(serde::Deserialize)]
struct SummaryBundle {
    loss_ratio: SignalReport,
    min_k: SignalReport,
    zlib: SignalReport,
}

fn print_signal(signal: f64, threshold: f64) {
    println!("signal    = {signal:.4}");
    println!("threshold = {threshold:.4}");
    println!(
        "verdict   = {}",
        if likely_in(signal, threshold) {
            "LIKELY IN"
        } else {
            "NOT IN"
        }
    );
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Cmd::LossRatio {
            original_loss,
            paraphrase_loss,
            threshold,
        } => {
            print_signal(loss_ratio_signal(original_loss, paraphrase_loss), threshold);
        }
        Cmd::MinK {
            token_log_probs,
            k_frac,
            threshold,
        } => {
            let lps: Lps = serde_json::from_str(&std::fs::read_to_string(&token_log_probs)?)?;
            print_signal(min_k_signal(&lps.token_log_probs, k_frac), threshold);
        }
        Cmd::Zlib {
            zlib_entropy_bits,
            nll_sum_nats,
            threshold,
        } => {
            print_signal(
                zlib_ratio_signal(zlib_entropy_bits, nll_sum_nats),
                threshold,
            );
        }
        Cmd::ScoreSignal {
            signal,
            bank,
            threshold,
            p_cutoff,
        } => {
            let b: Bank = serde_json::from_str(&std::fs::read_to_string(&bank)?)?;
            let r = score_signal(signal, &b.signals, threshold, p_cutoff);
            let verdict = r.verdict;
            println!("{}", serde_json::to_string_pretty(&r)?);
            use provenancepack::SignalVerdict::*;
            match verdict {
                LikelyIn => {}
                NotIn => std::process::exit(1),
                BankTooSmall => std::process::exit(2),
            }
        }
        Cmd::Summary { bundle } => {
            let b: SummaryBundle = serde_json::from_str(&std::fs::read_to_string(&bundle)?)?;
            let s = summarize_three(b.loss_ratio, b.min_k, b.zlib);
            let n = s.n_signals_flagged;
            let too_small = s.n_signals_bank_too_small;
            println!("{}", serde_json::to_string_pretty(&s)?);
            if too_small > 0 {
                std::process::exit(3);
            } else if n >= 2 {
                std::process::exit(2); // strong flag
            } else if n == 1 {
                std::process::exit(1); // single flag
            }
        }
        Cmd::Version => println!("provenancepack {}", env!("CARGO_PKG_VERSION")),
    }
    Ok(())
}
