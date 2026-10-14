//! provenancepack CLI — scaffold.

use clap::{Parser, Subcommand};
use provenancepack::{likely_in, loss_ratio_signal, min_k_signal, zlib_ratio_signal};

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
    Version,
}

#[derive(serde::Deserialize)]
struct Lps {
    token_log_probs: Vec<f64>,
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
        Cmd::Version => println!("provenancepack {}", env!("CARGO_PKG_VERSION")),
    }
    Ok(())
}
