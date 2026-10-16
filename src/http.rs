//! Live HTTP sample-emitter for MIA signals.
//!
//! Each MIA signal needs a specific measurement:
//!
//! * `loss_ratio_signal` needs the model's loss on an original text and
//!   on a paraphrase of it.
//! * `min_k_signal` needs the per-token log-probabilities of a text.
//! * `zlib_ratio_signal` needs the model's NLL in nats; zlib entropy is
//!   computed locally.
//!
//! A `MiaProfile` names where to POST text and which JSON paths carry
//! the loss and (optionally) the token log-probs. Non-fuzzy fields:
//! both are floats; log-probs are a non-empty finite array. The
//! emitter returns a `MiaSample` so the caller then feeds the three
//! signal functions and `score_signal`.

use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MiaProfile {
    pub name: String,
    pub url: String,
    #[serde(default = "default_text_field")]
    pub text_field: String,
    /// JSON path to the scalar loss (nats or bits, caller's choice; the
    /// signal functions are unit-agnostic).
    pub loss_json_path: String,
    /// Optional JSON path to an array of per-token log-probabilities.
    /// If absent, `MiaSample.token_log_probs` is None and callers that
    /// need min_k must switch to a different profile.
    #[serde(default)]
    pub token_log_probs_json_path: Option<String>,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default = "default_timeout_s")]
    pub timeout_s: u64,
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
}

fn default_text_field() -> String {
    "text".into()
}

fn default_timeout_s() -> u64 {
    60
}

fn default_max_retries() -> u32 {
    2
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MiaSample {
    pub loss: f64,
    pub token_log_probs: Option<Vec<f64>>,
}

#[derive(Debug, Error)]
pub enum MiaError {
    #[error("api_key_env `{0}` is set but the environment variable is not set")]
    AuthMissing(String),
    #[error("endpoint refused authentication (HTTP {status})")]
    AuthRejected { status: u16 },
    #[error("endpoint returned HTTP {status} after {attempts} attempts")]
    HttpStatus { status: u16, attempts: u32 },
    #[error("response missing field at JSON path `{path}`")]
    MissingField { path: String },
    #[error("response field at `{path}` is not a number")]
    NotANumber { path: String },
    #[error("response field at `{path}` is not a non-empty array of numbers")]
    NotANumberArray { path: String },
    #[error("response field at `{path}` is not finite")]
    NotFinite { path: String },
    #[error("response could not be parsed as JSON")]
    MalformedJson,
    #[error("transport error after {attempts} attempts: {source}")]
    Transport {
        attempts: u32,
        #[source]
        source: reqwest::Error,
    },
}

fn pointer_from_path(path: &str) -> String {
    let trimmed = path.trim_matches('/');
    if trimmed.is_empty() {
        String::new()
    } else {
        let mut s = String::with_capacity(trimmed.len() + 1);
        for seg in trimmed.split('/') {
            s.push('/');
            s.push_str(seg);
        }
        s
    }
}

/// Emit one MIA sample for one text.
pub fn emit_sample(profile: &MiaProfile, text: &str) -> Result<MiaSample, MiaError> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("current-thread runtime builds");
    rt.block_on(emit_sample_async(profile, text))
}

pub async fn emit_sample_async(profile: &MiaProfile, text: &str) -> Result<MiaSample, MiaError> {
    let auth = match &profile.api_key_env {
        None => None,
        Some(name) => Some(std::env::var(name).map_err(|_| MiaError::AuthMissing(name.clone()))?),
    };
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(profile.timeout_s))
        .build()
        .map_err(|e| MiaError::Transport {
            attempts: 0,
            source: e,
        })?;
    let body = serde_json::json!({ profile.text_field.clone(): text });
    let doc = send_with_retry(&client, &profile.url, auth.as_deref(), &body, profile.max_retries).await?;
    let loss_val = doc
        .pointer(&pointer_from_path(&profile.loss_json_path))
        .ok_or(MiaError::MissingField {
            path: profile.loss_json_path.clone(),
        })?;
    let loss = loss_val.as_f64().ok_or(MiaError::NotANumber {
        path: profile.loss_json_path.clone(),
    })?;
    if !loss.is_finite() {
        return Err(MiaError::NotFinite {
            path: profile.loss_json_path.clone(),
        });
    }
    let token_log_probs = match &profile.token_log_probs_json_path {
        None => None,
        Some(p) => {
            let arr_val = doc
                .pointer(&pointer_from_path(p))
                .ok_or(MiaError::MissingField { path: p.clone() })?;
            let arr = arr_val.as_array().ok_or(MiaError::NotANumberArray { path: p.clone() })?;
            if arr.is_empty() {
                return Err(MiaError::NotANumberArray { path: p.clone() });
            }
            let mut out = Vec::with_capacity(arr.len());
            for item in arr {
                let v = item.as_f64().ok_or(MiaError::NotANumberArray { path: p.clone() })?;
                if !v.is_finite() {
                    return Err(MiaError::NotFinite { path: p.clone() });
                }
                out.push(v);
            }
            Some(out)
        }
    };
    Ok(MiaSample {
        loss,
        token_log_probs,
    })
}

async fn send_with_retry(
    client: &reqwest::Client,
    url: &str,
    auth: Option<&str>,
    body: &serde_json::Value,
    max_retries: u32,
) -> Result<serde_json::Value, MiaError> {
    let mut attempt: u32 = 0;
    loop {
        attempt += 1;
        let mut req = client.post(url).json(body);
        if let Some(k) = auth {
            req = req.bearer_auth(k);
        }
        match req.send().await {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    return resp.json().await.map_err(|_| MiaError::MalformedJson);
                }
                if status.as_u16() == 401 || status.as_u16() == 403 {
                    return Err(MiaError::AuthRejected {
                        status: status.as_u16(),
                    });
                }
                let retryable = status.as_u16() == 429 || status.is_server_error();
                if !retryable || attempt > max_retries {
                    return Err(MiaError::HttpStatus {
                        status: status.as_u16(),
                        attempts: attempt,
                    });
                }
            }
            Err(e) => {
                let retryable = e.is_timeout() || e.is_connect() || e.is_request();
                if !retryable || attempt > max_retries {
                    return Err(MiaError::Transport {
                        attempts: attempt,
                        source: e,
                    });
                }
            }
        }
        let delay_ms = 100u64.saturating_mul(1u64 << (attempt - 1).min(10));
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn profile(base: &str, logprobs: bool) -> MiaProfile {
        MiaProfile {
            name: "mock".into(),
            url: format!("{base}/mia"),
            text_field: "text".into(),
            loss_json_path: "loss".into(),
            token_log_probs_json_path: if logprobs {
                Some("token_log_probs".into())
            } else {
                None
            },
            api_key_env: None,
            timeout_s: 2,
            max_retries: 2,
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn emits_a_sample_with_logprobs_when_configured() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/mia"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "loss": 1.5,
                "token_log_probs": [-0.1, -0.5, -1.2]
            })))
            .mount(&server)
            .await;
        let s = emit_sample_async(&profile(&server.uri(), true), "hello")
            .await
            .expect("ok");
        assert!((s.loss - 1.5).abs() < 1e-12);
        assert_eq!(s.token_log_probs.as_deref().map(|v| v.len()), Some(3));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn emits_a_sample_without_logprobs_when_profile_omits_them() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/mia"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "loss": 2.0, "token_log_probs": [-0.1]
            })))
            .mount(&server)
            .await;
        let s = emit_sample_async(&profile(&server.uri(), false), "x")
            .await
            .expect("ok");
        assert!(s.token_log_probs.is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn nonfinite_loss_errors_with_the_path() {
        let server = MockServer::start().await;
        // JSON cannot represent NaN; use a very large payload sent as a
        // non-finite inf is also not representable. Use a schema error:
        // if the server returns a string here, we hit NotANumber instead.
        Mock::given(method("POST"))
            .and(path("/mia"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "loss": "huge"
            })))
            .mount(&server)
            .await;
        match emit_sample_async(&profile(&server.uri(), false), "x").await {
            Err(MiaError::NotANumber { path }) => assert_eq!(path, "loss"),
            e => panic!("wrong variant: {e:?}"),
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn empty_logprobs_array_is_a_schema_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/mia"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "loss": 1.0, "token_log_probs": []
            })))
            .mount(&server)
            .await;
        match emit_sample_async(&profile(&server.uri(), true), "x").await {
            Err(MiaError::NotANumberArray { path }) => assert_eq!(path, "token_log_probs"),
            e => panic!("wrong variant: {e:?}"),
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn auth_rejected_is_fatal() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/mia"))
            .respond_with(ResponseTemplate::new(401))
            .expect(1)
            .mount(&server)
            .await;
        let err = emit_sample_async(&profile(&server.uri(), false), "x").await.expect_err("auth");
        assert!(matches!(err, MiaError::AuthRejected { status: 401 }));
    }
}
