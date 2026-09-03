// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Strict command-line and environment grammar for the service process.

use crate::admission::{RequestBodyLimit, ServiceAdmission};
use gigaam_audio::{FrontendMode, SampleRate};
use gigaam_model_package::EncoderPrecision;
use gigaam_recognition::{
    CudaAssignmentEnvironment, Device, OrtConfig, OrtEnvironment, ProviderPlan,
    RequiredEncoderRoles,
};
use std::path::PathBuf;
use std::time::Duration;

pub(crate) const DEFAULT_MODEL: &str = "model";
pub(crate) const DEFAULT_PORT: u16 = 8080;
pub(crate) const WINDOW_DURATION: Duration = Duration::from_secs(30);
pub(crate) const OVERLAP_DURATION: Duration = Duration::from_secs(6);
const NANOS_PER_SECOND: u128 = 1_000_000_000;
const BYTES_PER_MEBIBYTE: usize = 1024 * 1024;

pub(crate) fn encoder_precision(fp16: bool) -> EncoderPrecision {
    match fp16 {
        true => EncoderPrecision::Fp16Io32,
        false => EncoderPrecision::Fp32,
    }
}

pub(crate) fn precision_name(precision: EncoderPrecision) -> &'static str {
    match precision {
        EncoderPrecision::Fp32 => "fp32",
        EncoderPrecision::Fp16Io32 => "fp16",
    }
}

#[derive(Debug, Default)]
struct CliOptions {
    model: Option<PathBuf>,
    port: Option<u16>,
    provider: Option<String>,
    ort_dylib: Option<PathBuf>,
    ctc_fp16: bool,
    no_rnnt: bool,
    rnnt_fp16: bool,
}

#[derive(Debug)]
pub(crate) struct Startup {
    pub(crate) model: PathBuf,
    pub(crate) port: u16,
    pub(crate) plan: ProviderPlan,
    pub(crate) ort_dylib: PathBuf,
    pub(crate) ctc_fp16: bool,
    pub(crate) with_rnnt: bool,
    pub(crate) rnnt_fp16: bool,
    pub(crate) frontend_mode: FrontendMode,
    pub(crate) trace_enabled: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct StartupEnvironment<'a> {
    pub(crate) encoder_ep: Option<&'a str>,
    pub(crate) ort_dylib: Option<&'a str>,
    pub(crate) ort_memory_pattern: Option<&'a str>,
    pub(crate) ort_cuda_arena: Option<&'a str>,
    pub(crate) ort_intra_threads: Option<&'a str>,
    pub(crate) trt_cache: Option<&'a str>,
    pub(crate) trt_profile_min: Option<&'a str>,
    pub(crate) trt_profile_opt: Option<&'a str>,
    pub(crate) trt_profile_max: Option<&'a str>,
    pub(crate) frontend: Option<&'a str>,
    pub(crate) trace: Option<&'a str>,
    pub(crate) cuda_assignment_policy: Option<&'a str>,
    pub(crate) cuda_ctc_assignment_sha256: Option<&'a str>,
    pub(crate) cuda_rnnt_assignment_sha256: Option<&'a str>,
}

impl Startup {
    pub(crate) fn from_process() -> Result<Self, String> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let encoder_ep = process_env("ASR_ENCODER_EP")?;
        let ort_dylib = process_env("ORT_DYLIB_PATH")?;
        let ort_memory_pattern = process_env("ASR_ORT_MEMPATTERN")?;
        let ort_cuda_arena = process_env("ASR_ORT_ARENA")?;
        let ort_intra_threads = process_env("ASR_ORT_THREADS")?;
        let trt_cache = process_env("ASR_TRT_CACHE")?;
        let trt_profile_min = process_env("ASR_TRT_PROFILE_MIN")?;
        let trt_profile_opt = process_env("ASR_TRT_PROFILE_OPT")?;
        let trt_profile_max = process_env("ASR_TRT_PROFILE_MAX")?;
        let frontend = process_env("ASR_FRONTEND")?;
        let trace = process_env("ASR_TRACE")?;
        let cuda_assignment_policy = process_env("ASR_CUDA_ASSIGNMENT_POLICY")?;
        let cuda_ctc_assignment_sha256 = process_env("ASR_CUDA_CTC_ASSIGNMENT_SHA256")?;
        let cuda_rnnt_assignment_sha256 = process_env("ASR_CUDA_RNNT_ASSIGNMENT_SHA256")?;
        Self::parse(
            &args,
            StartupEnvironment {
                encoder_ep: encoder_ep.as_deref(),
                ort_dylib: ort_dylib.as_deref(),
                ort_memory_pattern: ort_memory_pattern.as_deref(),
                ort_cuda_arena: ort_cuda_arena.as_deref(),
                ort_intra_threads: ort_intra_threads.as_deref(),
                trt_cache: trt_cache.as_deref(),
                trt_profile_min: trt_profile_min.as_deref(),
                trt_profile_opt: trt_profile_opt.as_deref(),
                trt_profile_max: trt_profile_max.as_deref(),
                frontend: frontend.as_deref(),
                trace: trace.as_deref(),
                cuda_assignment_policy: cuda_assignment_policy.as_deref(),
                cuda_ctc_assignment_sha256: cuda_ctc_assignment_sha256.as_deref(),
                cuda_rnnt_assignment_sha256: cuda_rnnt_assignment_sha256.as_deref(),
            },
        )
    }

    pub(crate) fn parse(
        args: &[String],
        environment: StartupEnvironment<'_>,
    ) -> Result<Self, String> {
        let cli = parse_cli(args)?;
        if cli.no_rnnt && cli.rnnt_fp16 {
            return Err("--rnnt-fp16 conflicts with --no-rnnt".into());
        }
        let provider =
            Device::resolve_with_config(cli.provider.as_deref(), environment.encoder_ep)?;
        let ort_config = OrtConfig::from_environment(OrtEnvironment {
            memory_pattern: environment.ort_memory_pattern,
            cuda_arena: environment.ort_cuda_arena,
            tensorrt_cache: environment.trt_cache,
            tensorrt_profile_min: environment.trt_profile_min,
            tensorrt_profile_opt: environment.trt_profile_opt,
            tensorrt_profile_max: environment.trt_profile_max,
            intra_threads: environment.ort_intra_threads,
        })?;
        let required_roles = match cli.no_rnnt {
            true => RequiredEncoderRoles::ctc(),
            false => RequiredEncoderRoles::ctc_and_rnnt(),
        };
        let plan = ProviderPlan::new_with_cuda_assignment(
            provider,
            ort_config,
            required_roles,
            CudaAssignmentEnvironment {
                policy: environment.cuda_assignment_policy,
                ctc_sha256: environment.cuda_ctc_assignment_sha256,
                rnnt_sha256: environment.cuda_rnnt_assignment_sha256,
            },
        )?;
        let frontend_mode = match environment.frontend {
            Some(value) => {
                FrontendMode::parse(value).map_err(|error| format!("ASR_FRONTEND: {error}"))?
            }
            None => FrontendMode::Scalar,
        };
        let trace_enabled = match environment.trace {
            None => false,
            Some("") => return Err("ASR_TRACE must not be empty when present".into()),
            Some(_) => true,
        };
        let ort_dylib = configured_path(
            cli.ort_dylib,
            environment.ort_dylib,
            "--ort-dylib",
            "ORT_DYLIB_PATH",
        )?;
        Ok(Self {
            model: cli.model.unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL)),
            port: cli.port.unwrap_or(DEFAULT_PORT),
            plan,
            ort_dylib,
            ctc_fp16: cli.ctc_fp16,
            with_rnnt: !cli.no_rnnt,
            rnnt_fp16: cli.rnnt_fp16,
            frontend_mode,
            trace_enabled,
        })
    }
}

fn process_env(key: &str) -> Result<Option<String>, String> {
    match std::env::var(key) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!("{key} must contain UTF-8 text")),
    }
}

fn parse_cli(args: &[String]) -> Result<CliOptions, String> {
    let mut options = CliOptions::default();
    let mut position = 0;
    while position < args.len() {
        match args[position].as_str() {
            "--model" => set_once(
                &mut options.model,
                PathBuf::from(next_value(args, &mut position, "--model")?),
                "--model",
            )?,
            "--port" => {
                let value = next_value(args, &mut position, "--port")?;
                let port = value
                    .parse::<u16>()
                    .map_err(|_| format!("--port must be an integer in 1..65535, got {value:?}"))?;
                if port == 0 {
                    return Err("--port must be in 1..65535".into());
                }
                set_once(&mut options.port, port, "--port")?;
            }
            "--ep" => set_once(
                &mut options.provider,
                next_value(args, &mut position, "--ep")?,
                "--ep",
            )?,
            "--ort-dylib" => set_once(
                &mut options.ort_dylib,
                PathBuf::from(next_value(args, &mut position, "--ort-dylib")?),
                "--ort-dylib",
            )?,
            "--ctc-fp16" => set_flag(&mut options.ctc_fp16, "--ctc-fp16")?,
            "--no-rnnt" => set_flag(&mut options.no_rnnt, "--no-rnnt")?,
            "--rnnt-fp16" => set_flag(&mut options.rnnt_fp16, "--rnnt-fp16")?,
            option if option.starts_with('-') => return Err(format!("unknown option {option}")),
            value => return Err(format!("unexpected positional argument {value:?}")),
        }
        position += 1;
    }
    Ok(options)
}

fn next_value(args: &[String], position: &mut usize, option: &str) -> Result<String, String> {
    *position += 1;
    let value = args
        .get(*position)
        .ok_or_else(|| format!("{option} requires a value"))?
        .clone();
    if value.is_empty() {
        return Err(format!("{option} must not be empty"));
    }
    Ok(value)
}

fn set_once<T>(slot: &mut Option<T>, value: T, option: &str) -> Result<(), String> {
    if slot.is_some() {
        return Err(format!("{option} may be configured only once"));
    }
    *slot = Some(value);
    Ok(())
}

fn set_flag(slot: &mut bool, option: &str) -> Result<(), String> {
    if *slot {
        return Err(format!("{option} may be specified only once"));
    }
    *slot = true;
    Ok(())
}

fn configured_path(
    cli: Option<PathBuf>,
    environment: Option<&str>,
    cli_name: &str,
    environment_name: &str,
) -> Result<PathBuf, String> {
    match (cli, environment) {
        (Some(cli), Some(environment)) => {
            if environment.is_empty() {
                return Err(format!("{environment_name} must not be empty"));
            }
            let environment_path = PathBuf::from(environment);
            if cli != environment_path {
                return Err(format!(
                    "{cli_name} ({}) conflicts with {environment_name} ({environment})",
                    cli.display()
                ));
            }
            Ok(cli)
        }
        (Some(cli), None) => Ok(cli),
        (None, Some("")) => Err(format!("{environment_name} must not be empty")),
        (None, Some(environment)) => Ok(PathBuf::from(environment)),
        (None, None) => Err(format!(
            "{cli_name} or {environment_name} must name the ONNX Runtime library"
        )),
    }
}

/// Borrowed process limit inputs, separated from process environment access for deterministic tests.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ServerLimitEnvironment<'a> {
    pub(crate) max_http: Option<&'a str>,
    pub(crate) max_ws: Option<&'a str>,
    pub(crate) body_mb: Option<&'a str>,
    pub(crate) timeout_seconds: Option<&'a str>,
    pub(crate) dedup_default: Option<&'a str>,
    pub(crate) dedup_window: Option<&'a str>,
    pub(crate) dedup_threshold: Option<&'a str>,
    pub(crate) backchannel_max: Option<&'a str>,
}

#[derive(Debug)]
pub(crate) struct ServerLimits {
    pub(crate) admission: ServiceAdmission,
    pub(crate) request_body_limit: RequestBodyLimit,
    pub(crate) dedup_default: bool,
    pub(crate) dedup_window: PositiveDuration,
    pub(crate) dedup_threshold: f32,
    pub(crate) backchannel_max: NonnegativeDuration,
}

/// Limits requiring the validated model sample rate before frontend/session construction.
#[derive(Debug)]
pub(crate) struct ModelBoundLimits {
    pub(crate) limits: ServerLimits,
    pub(crate) dedup_window_samples: SampleCount,
}

/// A nonnegative decimal duration parsed without floating-point string round-trips.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NonnegativeDuration(Duration);

impl NonnegativeDuration {
    fn parse(value: &str, key: &str, nanoseconds_per_unit: u128) -> Result<Self, String> {
        if value.is_empty() {
            return Err(format!("{key} must not be empty"));
        }
        let (whole, fraction) = match value.split_once('.') {
            Some((whole, fraction)) if !fraction.contains('.') => (whole, Some(fraction)),
            Some(_) => return Err(format!("{key} must be a decimal duration, got {value:?}")),
            None => (value, None),
        };
        if whole.is_empty() || !whole.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(format!(
                "{key} must be a non-negative decimal duration, got {value:?}"
            ));
        }
        let whole = whole
            .parse::<u128>()
            .map_err(|_| format!("{key} exceeds the supported duration"))?;
        let mut total_nanoseconds = whole
            .checked_mul(nanoseconds_per_unit)
            .ok_or_else(|| format!("{key} exceeds the supported duration"))?;
        if let Some(fraction) = fraction {
            if fraction.is_empty()
                || fraction.len() > 9
                || !fraction.bytes().all(|byte| byte.is_ascii_digit())
            {
                return Err(format!(
                    "{key} must have one to nine decimal places, got {value:?}"
                ));
            }
            let numerator = fraction
                .parse::<u128>()
                .map_err(|_| format!("{key} exceeds the supported duration"))?
                .checked_mul(nanoseconds_per_unit)
                .ok_or_else(|| format!("{key} exceeds the supported duration"))?;
            let denominator = 10_u128
                .checked_pow(
                    u32::try_from(fraction.len())
                        .map_err(|_| format!("{key} has too many decimal places"))?,
                )
                .ok_or_else(|| format!("{key} exceeds the supported duration"))?;
            let fractional_nanoseconds = numerator
                .checked_add(denominator / 2)
                .ok_or_else(|| format!("{key} exceeds the supported duration"))?
                / denominator;
            total_nanoseconds = total_nanoseconds
                .checked_add(fractional_nanoseconds)
                .ok_or_else(|| format!("{key} exceeds the supported duration"))?;
        }
        let nanoseconds = u64::try_from(total_nanoseconds)
            .map_err(|_| format!("{key} exceeds the supported duration"))?;
        Ok(Self(Duration::from_nanos(nanoseconds)))
    }

    pub(crate) fn as_secs_f32(self) -> f32 {
        self.0.as_secs_f32()
    }
}

/// A duration that must represent at least one nanosecond.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PositiveDuration(NonnegativeDuration);

impl PositiveDuration {
    fn parse_seconds(value: &str, key: &str) -> Result<Self, String> {
        let duration = NonnegativeDuration::parse(value, key, NANOS_PER_SECOND)?;
        if duration.0.is_zero() {
            return Err(format!("{key} must be greater than zero"));
        }
        Ok(Self(duration))
    }

    pub(crate) fn samples_at(
        self,
        sample_rate: SampleRate,
        key: &str,
    ) -> Result<SampleCount, String> {
        let whole_samples = u128::from(self.0.0.as_secs())
            .checked_mul(u128::from(sample_rate.hertz()))
            .ok_or_else(|| format!("{key} sample count exceeds usize"))?;
        let fractional_product = u128::from(self.0.0.subsec_nanos())
            .checked_mul(u128::from(sample_rate.hertz()))
            .ok_or_else(|| format!("{key} sample count exceeds usize"))?;
        let fractional_samples = fractional_product
            .checked_add(NANOS_PER_SECOND / 2)
            .ok_or_else(|| format!("{key} sample count exceeds usize"))?
            / NANOS_PER_SECOND;
        let samples = whole_samples
            .checked_add(fractional_samples)
            .ok_or_else(|| format!("{key} sample count exceeds usize"))?;
        SampleCount::new(samples, key)
    }
}

/// An addressable count of model samples.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SampleCount(usize);

impl SampleCount {
    fn new(value: u128, key: &str) -> Result<Self, String> {
        let value =
            usize::try_from(value).map_err(|_| format!("{key} sample count exceeds usize"))?;
        if value == 0 {
            return Err(format!("{key} sample count must be greater than zero"));
        }
        Ok(Self(value))
    }

    pub(crate) const fn get(self) -> usize {
        self.0
    }
}

impl ServerLimits {
    pub(crate) fn from_process() -> Result<Self, String> {
        let max_http = process_env("ASR_MAX_CONCURRENCY")?;
        let max_ws = process_env("ASR_MAX_STREAMS")?;
        let body_mb = process_env("ASR_BODY_LIMIT_MB")?;
        let timeout_seconds = process_env("ASR_REQ_TIMEOUT_SEC")?;
        let dedup_default = process_env("ASR_DEDUP")?;
        let dedup_window = process_env("ASR_DEDUP_WINDOW_SEC")?;
        let dedup_threshold = process_env("ASR_DEDUP_THRESHOLD")?;
        let backchannel_max = process_env("ASR_BACKCHANNEL_MAX_MS")?;
        Self::parse(ServerLimitEnvironment {
            max_http: max_http.as_deref(),
            max_ws: max_ws.as_deref(),
            body_mb: body_mb.as_deref(),
            timeout_seconds: timeout_seconds.as_deref(),
            dedup_default: dedup_default.as_deref(),
            dedup_window: dedup_window.as_deref(),
            dedup_threshold: dedup_threshold.as_deref(),
            backchannel_max: backchannel_max.as_deref(),
        })
    }

    pub(crate) fn parse(environment: ServerLimitEnvironment<'_>) -> Result<Self, String> {
        // Parse every raw input in the established environment-key order before runtime work.
        let max_http = required_positive_usize(environment.max_http, "ASR_MAX_CONCURRENCY", 8)?;
        let max_ws = required_positive_usize(environment.max_ws, "ASR_MAX_STREAMS", 32)?;
        let body_mb = required_positive_usize(environment.body_mb, "ASR_BODY_LIMIT_MB", 64)?;
        let timeout_usize =
            required_positive_usize(environment.timeout_seconds, "ASR_REQ_TIMEOUT_SEC", 120)?;
        let dedup_default = parse_optional_bool(environment.dedup_default, "ASR_DEDUP", true)?;
        let dedup_window = match environment.dedup_window {
            Some(value) => PositiveDuration::parse_seconds(value, "ASR_DEDUP_WINDOW_SEC")?,
            None => PositiveDuration::parse_seconds("4", "ASR_DEDUP_WINDOW_SEC")?,
        };
        let dedup_threshold = finite_f32_value(
            environment.dedup_threshold,
            "ASR_DEDUP_THRESHOLD",
            0.99,
            |value| (0.0..=1.0).contains(&value),
        )?;
        let backchannel_max = match environment.backchannel_max {
            Some(value) => NonnegativeDuration::parse(value, "ASR_BACKCHANNEL_MAX_MS", 1_000_000)?,
            None => NonnegativeDuration(Duration::ZERO),
        };

        let body_bytes = body_mb
            .checked_mul(BYTES_PER_MEBIBYTE)
            .ok_or_else(|| "ASR_BODY_LIMIT_MB exceeds the supported body size".to_owned())?;
        let request_body_limit = RequestBodyLimit::new(body_bytes)?;
        let timeout_seconds = u64::try_from(timeout_usize)
            .map_err(|_| "ASR_REQ_TIMEOUT_SEC exceeds the supported duration".to_owned())?;
        let admission =
            ServiceAdmission::new(max_http, max_ws, Duration::from_secs(timeout_seconds))?;
        Ok(Self {
            admission,
            request_body_limit,
            dedup_default,
            dedup_window,
            dedup_threshold,
            backchannel_max,
        })
    }

    pub(crate) fn bind_model(self, sample_rate: SampleRate) -> Result<ModelBoundLimits, String> {
        let dedup_window_samples = self
            .dedup_window
            .samples_at(sample_rate, "ASR_DEDUP_WINDOW_SEC")?;
        Ok(ModelBoundLimits {
            limits: self,
            dedup_window_samples,
        })
    }

    pub(crate) const fn admission(&self) -> &ServiceAdmission {
        &self.admission
    }

    pub(crate) const fn request_body_limit(&self) -> RequestBodyLimit {
        self.request_body_limit
    }

    pub(crate) fn body_megabytes(&self) -> usize {
        self.request_body_limit.bytes() / BYTES_PER_MEBIBYTE
    }
}

fn required_positive_usize(
    value: Option<&str>,
    key: &str,
    default: usize,
) -> Result<usize, String> {
    let Some(value) = value else {
        return Ok(default);
    };
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{key} must be a positive integer, got {value:?}"))?;
    if parsed == 0 {
        return Err(format!("{key} must be a positive integer"));
    }
    Ok(parsed)
}

fn finite_f32_value(
    value: Option<&str>,
    key: &str,
    default: f32,
    accepts: impl FnOnce(f32) -> bool,
) -> Result<f32, String> {
    let Some(value) = value else {
        return Ok(default);
    };
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("{key} must be a finite number, got {value:?}"))?;
    if !parsed.is_finite() || !accepts(parsed) {
        return Err(format!("{key} has an out-of-range value {value:?}"));
    }
    Ok(parsed)
}

pub(crate) fn parse_optional_bool(
    value: Option<&str>,
    key: &str,
    default: bool,
) -> Result<bool, String> {
    match value {
        Some(value) => parse_bool(key, value),
        None => Ok(default),
    }
}

pub(crate) fn parse_bool(key: &str, value: &str) -> Result<bool, String> {
    match value {
        "1" | "true" | "on" => Ok(true),
        "0" | "false" | "off" => Ok(false),
        _ => Err(format!(
            "{key} must be one of 1|0|true|false|on|off, got {value:?}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Device, NonnegativeDuration, PositiveDuration, ServerLimitEnvironment, ServerLimits,
        Startup, StartupEnvironment, parse_bool, parse_optional_bool,
    };
    use gigaam_audio::SampleRate;
    use std::num::NonZeroUsize;
    use tokio::sync::Semaphore;

    fn arguments(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn default_provider_environment() -> StartupEnvironment<'static> {
        #[cfg(feature = "cuda")]
        {
            StartupEnvironment {
                cuda_ctc_assignment_sha256: Some(
                    "0000000000000000000000000000000000000000000000000000000000000000",
                ),
                cuda_rnnt_assignment_sha256: Some(
                    "1111111111111111111111111111111111111111111111111111111111111111",
                ),
                ..StartupEnvironment::default()
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            StartupEnvironment::default()
        }
    }

    #[test]
    fn startup_defaults_to_the_compiled_provider_when_provider_is_absent() {
        let startup = Startup::parse(
            &arguments(&["--ort-dylib", "/runtime/libonnxruntime.so"]),
            default_provider_environment(),
        )
        .expect("absent provider is a documented default");
        assert_eq!(startup.plan.device(), Device::build_default());
        assert!(startup.with_rnnt);
    }

    #[test]
    fn startup_rejects_duplicate_or_conflicting_configuration() {
        let duplicate = Startup::parse(
            &arguments(&[
                "--ort-dylib",
                "/runtime/libonnxruntime.so",
                "--ep",
                "cpu",
                "--ep",
                "cpu",
            ]),
            StartupEnvironment::default(),
        )
        .expect_err("a repeated option must be refused");
        assert!(duplicate.contains("only once"));

        let conflict = Startup::parse(
            &arguments(&["--ort-dylib", "/runtime/libonnxruntime.so", "--ep", "cpu"]),
            StartupEnvironment {
                encoder_ep: Some("cuda"),
                ort_dylib: Some("/runtime/libonnxruntime.so"),
                ..StartupEnvironment::default()
            },
        )
        .expect_err("CLI and environment provider values must agree");
        assert!(conflict.contains("conflicts"));

        let rnnt_conflict = Startup::parse(
            &arguments(&[
                "--ort-dylib",
                "/runtime/libonnxruntime.so",
                "--no-rnnt",
                "--rnnt-fp16",
            ]),
            StartupEnvironment::default(),
        )
        .expect_err("a disabled model cannot select its fp16 graph");
        assert!(rnnt_conflict.contains("conflicts"));
    }

    #[test]
    fn startup_rejects_unknown_options_and_empty_runtime_paths() {
        let unknown = Startup::parse(
            &arguments(&["--cpu", "--ort-dylib", "/runtime/libonnxruntime.so"]),
            StartupEnvironment::default(),
        )
        .expect_err("unknown options must not select a provider");
        assert!(unknown.contains("unknown option"));

        let empty = Startup::parse(
            &arguments(&["--ort-dylib", "/runtime/libonnxruntime.so"]),
            StartupEnvironment {
                encoder_ep: Some("cpu"),
                ort_dylib: Some(""),
                ..StartupEnvironment::default()
            },
        )
        .expect_err("an empty environment path must be refused");
        assert!(empty.contains("must not be empty"));
    }

    #[test]
    fn typed_boolean_values_preserve_the_dedup_default() {
        assert_eq!(parse_optional_bool(None, "ASR_DEDUP", true), Ok(true));
        assert_eq!(parse_bool("ASR_DEDUP", "on"), Ok(true));
        assert_eq!(parse_bool("ASR_DEDUP", "off"), Ok(false));
        assert!(parse_bool("ASR_DEDUP", "yes").is_err());
    }

    #[test]
    fn service_feature_contract_accepts_only_compiled_providers() {
        assert_eq!(Device::parse("cpu"), Ok(Device::Cpu));
        assert!(Device::parse("gpu").is_err());
        assert!(Device::parse("trt").is_err());
    }

    #[test]
    fn decimal_durations_round_to_model_samples_without_float_string_conversion() {
        let sample_rate =
            SampleRate::new(16_000).expect("the documented model sample rate must be valid");
        let half_sample = PositiveDuration::parse_seconds("0.00003125", "duration")
            .expect("a finite positive decimal duration must parse");
        let one_and_a_half_samples = PositiveDuration::parse_seconds("0.00009375", "duration")
            .expect("a finite positive decimal duration must parse");
        assert_eq!(
            half_sample
                .samples_at(sample_rate, "duration")
                .expect("half a sample rounds up")
                .get(),
            1
        );
        assert_eq!(
            one_and_a_half_samples
                .samples_at(sample_rate, "duration")
                .expect("one and a half samples round up")
                .get(),
            2
        );
        let milliseconds = NonnegativeDuration::parse("250", "milliseconds", 1_000_000)
            .expect("milliseconds must parse exactly");
        assert_eq!(milliseconds.as_secs_f32(), 0.25);
        assert!(PositiveDuration::parse_seconds("0", "duration").is_err());
        assert!(PositiveDuration::parse_seconds("NaN", "duration").is_err());
    }

    #[test]
    fn startup_frontend_and_trace_environment_values_remain_exactly_typed() {
        let arguments = arguments(&["--ort-dylib", "/runtime/libonnxruntime.so"]);
        let scalar = Startup::parse(&arguments, default_provider_environment())
            .expect("an absent frontend selects scalar mode");
        assert_eq!(scalar.frontend_mode, gigaam_audio::FrontendMode::Scalar);
        assert!(!scalar.trace_enabled);

        let batched = Startup::parse(
            &arguments,
            StartupEnvironment {
                frontend: Some("batched"),
                trace: Some("request-window"),
                ..default_provider_environment()
            },
        )
        .expect("documented frontend and nonempty trace must parse");
        assert_eq!(batched.frontend_mode, gigaam_audio::FrontendMode::Batched);
        assert!(batched.trace_enabled);

        let empty_trace = Startup::parse(
            &arguments,
            StartupEnvironment {
                trace: Some(""),
                ..default_provider_environment()
            },
        )
        .expect_err("an explicitly empty trace value is invalid");
        assert_eq!(empty_trace, "ASR_TRACE must not be empty when present");
    }

    #[test]
    fn startup_ort_thread_count_environment_value_is_exactly_typed() {
        let arguments = arguments(&["--ort-dylib", "/runtime/libonnxruntime.so"]);
        let absent = Startup::parse(&arguments, default_provider_environment())
            .expect("an absent thread count leaves the library default");
        assert_eq!(absent.plan.intra_threads(), None);

        let positive = Startup::parse(
            &arguments,
            StartupEnvironment {
                ort_intra_threads: Some("4"),
                ..default_provider_environment()
            },
        )
        .expect("a positive documented thread count must parse");
        assert_eq!(
            positive.plan.intra_threads().map(NonZeroUsize::get),
            Some(4)
        );

        let maximum = Startup::parse(
            &arguments,
            StartupEnvironment {
                ort_intra_threads: Some("2147483647"),
                ..default_provider_environment()
            },
        )
        .expect("the documented maximum thread count must parse");
        assert_eq!(
            maximum.plan.intra_threads().map(NonZeroUsize::get),
            Some(2_147_483_647)
        );

        for invalid in [
            "",
            "0",
            "-1",
            "x",
            " 4",
            "+4",
            "2147483648",
            "4294967296",
            "18446744073709551616",
        ] {
            let error = Startup::parse(
                &arguments,
                StartupEnvironment {
                    ort_intra_threads: Some(invalid),
                    ..default_provider_environment()
                },
            )
            .expect_err("an invalid thread count must be refused");
            assert!(
                error.contains("ASR_ORT_THREADS"),
                "refusal must name ASR_ORT_THREADS: {error}"
            );
        }
    }

    #[test]
    fn limits_parse_all_independent_boundaries_with_stable_precedence() {
        let above_semaphore = Semaphore::MAX_PERMITS
            .checked_add(1)
            .expect("the test platform leaves room above semaphore capacity");
        let above_semaphore = above_semaphore.to_string();
        let http = ServerLimits::parse(ServerLimitEnvironment {
            max_http: Some(&above_semaphore),
            ..ServerLimitEnvironment::default()
        })
        .expect_err("HTTP admission above semaphore capacity must refuse");
        assert_eq!(
            http,
            "ASR_MAX_CONCURRENCY exceeds the supported semaphore capacity"
        );
        let ws = ServerLimits::parse(ServerLimitEnvironment {
            max_ws: Some(&above_semaphore),
            ..ServerLimitEnvironment::default()
        })
        .expect_err("WebSocket admission above semaphore capacity must refuse");
        assert_eq!(
            ws,
            "ASR_MAX_STREAMS exceeds the supported semaphore capacity"
        );

        let overflowing_mebibytes = usize::MAX
            .checked_div(1024 * 1024)
            .and_then(|value| value.checked_add(1))
            .expect("the test platform has a first overflowing mebibyte count")
            .to_string();
        let body = ServerLimits::parse(ServerLimitEnvironment {
            body_mb: Some(&overflowing_mebibytes),
            ..ServerLimitEnvironment::default()
        })
        .expect_err("the first overflowing body value must refuse");
        assert_eq!(body, "ASR_BODY_LIMIT_MB exceeds the supported body size");

        let precedence = ServerLimits::parse(ServerLimitEnvironment {
            max_http: Some("0"),
            body_mb: Some(&overflowing_mebibytes),
            ..ServerLimitEnvironment::default()
        })
        .expect_err("earlier independent keys retain deterministic precedence");
        assert_eq!(precedence, "ASR_MAX_CONCURRENCY must be a positive integer");

        let zero_body = ServerLimits::parse(ServerLimitEnvironment {
            body_mb: Some("0"),
            ..ServerLimitEnvironment::default()
        })
        .expect_err("zero mebibytes refuse at the environment boundary");
        assert_eq!(zero_body, "ASR_BODY_LIMIT_MB must be a positive integer");
    }
}
