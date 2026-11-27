//! provenancepack CLI — scaffold.

use clap::{Parser, Subcommand};
use provenancepack::{
    accusation_rate, likely_in, loss_ratio_signal, min_k_signal, score_signal, summarize_three,
    zlib_ratio_signal, SignalReport,
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
    /// Model-free simulation of the false-accusation rate of a canary
    /// bank of size n against a never-trained responder.
    BankSim {
        #[arg(long)]
        seed: u64,
        #[arg(long)]
        trials: usize,
        #[arg(long, value_delimiter = ',')]
        sizes: Vec<usize>,
        #[arg(long)]
        chance_hit: f64,
        #[arg(long)]
        threshold_frac: f64,
        #[arg(long)]
        out: std::path::PathBuf,
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
        Cmd::BankSim {
            seed,
            trials,
            sizes,
            chance_hit,
            threshold_frac,
            out,
        } => {
            let t0 = std::time::Instant::now();
            let started = utc_now();
            let mut rng = provenancepack::Xorshift64::new(seed);
            let mut fpr = serde_json::Map::new();
            let mut degen = serde_json::Map::new();
            for &n in &sizes {
                let r = accusation_rate(n, chance_hit, threshold_frac, trials, &mut rng);
                let d = accusation_rate(n, 1.0, threshold_frac, trials, &mut rng);
                fpr.insert(n.to_string(), r.into());
                degen.insert(n.to_string(), d.into());
            }
            let argv: Vec<String> = std::env::args().skip(1).collect();
            let doc = serde_json::json!({
                "false_accusation_rate_by_bank_size": fpr,
                "always_answer_accusation_rate_by_bank_size": degen,
                "params": {"trials": trials, "sizes": sizes, "chance_hit": chance_hit,
                           "threshold_frac": threshold_frac, "min_bank_size": provenancepack::MIN_BANK_SIZE},
                "provenance": {
                    "started_at": started,
                    "seeds": [seed],
                    "command": format!("provenancepack {}", argv.join(" ")),
                    "hardware": "cpu, single thread",
                    "wall_s": t0.elapsed().as_secs_f64(),
                    "data_sources": [{"url": "https://arxiv.org/pdf/2505.23001",
                        "license": "arXiv preprint, cited only; simulation uses no external data"}]
                }
            });
            if let Some(dir) = out.parent() {
                std::fs::create_dir_all(dir)?;
            }
            std::fs::write(&out, serde_json::to_string_pretty(&doc)? + "\n")?;
            println!(
                "{}",
                serde_json::to_string_pretty(&doc["false_accusation_rate_by_bank_size"])?
            );
        }
        Cmd::Version => println!("provenancepack {}", env!("CARGO_PKG_VERSION")),
    }
    Ok(())
}

fn utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    // civil-from-days (Howard Hinnant)
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}
