// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Typed process-environment configuration for the offline CLI.

use crate::grammar::RuntimeRequest;
use gigaam_audio::FrontendMode;
use gigaam_recognition::{
    CudaAssignmentEnvironment, Device, OrtConfig, OrtEnvironment, ProviderPlan,
    RequiredEncoderRoles,
};
use std::path::PathBuf;

/// One fully validated local ORT runtime request.
#[derive(Clone, Debug)]
pub(crate) struct RuntimeConfig {
    plan: ProviderPlan,
    dylib: PathBuf,
}

impl RuntimeConfig {
    pub(crate) fn plan(&self) -> &ProviderPlan {
        &self.plan
    }

    pub(crate) fn dylib(&self) -> &std::path::Path {
        &self.dylib
    }
}

/// Whether the CLI emits synchronous batch-window observations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TraceMode {
    Disabled,
    Enabled,
}

/// Reads the established provider and ORT environment in its historical order.
pub(crate) fn runtime(
    request: &RuntimeRequest,
    required_roles: RequiredEncoderRoles,
) -> Result<RuntimeConfig, String> {
    let encoder_ep = process_env("ASR_ENCODER_EP")?;
    let device = Device::resolve_with_config(request.provider.as_deref(), encoder_ep.as_deref())?;
    let memory_pattern = process_env("ASR_ORT_MEMPATTERN")?;
    let cuda_arena = process_env("ASR_ORT_ARENA")?;
    let intra_threads = process_env("ASR_ORT_THREADS")?;
    let tensorrt_cache = process_env("ASR_TRT_CACHE")?;
    let tensorrt_profile_min = process_env("ASR_TRT_PROFILE_MIN")?;
    let tensorrt_profile_opt = process_env("ASR_TRT_PROFILE_OPT")?;
    let tensorrt_profile_max = process_env("ASR_TRT_PROFILE_MAX")?;
    let cuda_assignment_policy = process_env("ASR_CUDA_ASSIGNMENT_POLICY")?;
    let cuda_ctc_assignment_sha256 = process_env("ASR_CUDA_CTC_ASSIGNMENT_SHA256")?;
    let cuda_rnnt_assignment_sha256 = process_env("ASR_CUDA_RNNT_ASSIGNMENT_SHA256")?;
    let ort = OrtConfig::from_environment(OrtEnvironment {
        memory_pattern: memory_pattern.as_deref(),
        cuda_arena: cuda_arena.as_deref(),
        tensorrt_cache: tensorrt_cache.as_deref(),
        tensorrt_profile_min: tensorrt_profile_min.as_deref(),
        tensorrt_profile_opt: tensorrt_profile_opt.as_deref(),
        tensorrt_profile_max: tensorrt_profile_max.as_deref(),
        intra_threads: intra_threads.as_deref(),
    })?;
    let plan = ProviderPlan::new_with_cuda_assignment(
        device,
        ort,
        required_roles,
        CudaAssignmentEnvironment {
            policy: cuda_assignment_policy.as_deref(),
            ctc_sha256: cuda_ctc_assignment_sha256.as_deref(),
            rnnt_sha256: cuda_rnnt_assignment_sha256.as_deref(),
        },
    )?;
    let ort_dylib = process_env("ORT_DYLIB_PATH")?;
    let dylib = configured_path(
        request.ort_dylib.clone(),
        ort_dylib.as_deref(),
        "--ort-dylib",
        "ORT_DYLIB_PATH",
    )?;
    Ok(RuntimeConfig { plan, dylib })
}

/// Parses the applicable frontend process setting exactly once before package/native work.
pub(crate) fn frontend_mode() -> Result<FrontendMode, String> {
    let value = process_env("ASR_FRONTEND")?;
    frontend_mode_from(value.as_deref())
}

fn frontend_mode_from(value: Option<&str>) -> Result<FrontendMode, String> {
    match value {
        Some(value) => FrontendMode::parse(value).map_err(|error| format!("ASR_FRONTEND: {error}")),
        None => Ok(FrontendMode::Scalar),
    }
}

/// Parses the trace capability only for commands that expose batch-window observations.
pub(crate) fn trace_mode() -> Result<TraceMode, String> {
    let value = process_env("ASR_TRACE")?;
    trace_mode_from(value.as_deref())
}

fn trace_mode_from(value: Option<&str>) -> Result<TraceMode, String> {
    match value {
        None => Ok(TraceMode::Disabled),
        Some("") => Err("ASR_TRACE must not be empty when present".into()),
        Some(_) => Ok(TraceMode::Enabled),
    }
}

fn process_env(key: &str) -> Result<Option<String>, String> {
    match std::env::var_os(key) {
        None => Ok(None),
        Some(value) => value
            .into_string()
            .map(Some)
            .map_err(|_| format!("{key} must contain UTF-8 text")),
    }
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

#[cfg(test)]
mod tests {
    use super::{TraceMode, frontend_mode_from, trace_mode_from};
    use gigaam_audio::FrontendMode;

    #[test]
    fn frontend_and_trace_process_values_are_parsed_at_the_cli_boundary() {
        assert_eq!(frontend_mode_from(None), Ok(FrontendMode::Scalar));
        assert_eq!(
            frontend_mode_from(Some("batched")),
            Ok(FrontendMode::Batched)
        );
        assert!(frontend_mode_from(Some("")).is_err());
        assert!(frontend_mode_from(Some("scalar ")).is_err());
        assert_eq!(trace_mode_from(None), Ok(TraceMode::Disabled));
        assert!(trace_mode_from(Some("")).is_err());
    }
}
