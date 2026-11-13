// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Typed process-environment configuration for the offline CLI.

use crate::grammar::RuntimeRequest;
use gigaam_audio::FrontendMode;
use gigaam_recognition::{
    CudaAssignmentEnvironment, Device, OrtConfig, OrtEnvironment, ProviderPlan,
    RequiredEncoderRoles, RnntPrecisionSelection,
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

pub(crate) fn rnnt_precision(
    cli: Option<RnntPrecisionSelection>,
    legacy_fp16: bool,
) -> Result<Option<RnntPrecisionSelection>, String> {
    let value = process_env("ASR_RNNT_PRECISION")?;
    resolve_rnnt_precision(cli, legacy_fp16, value.as_deref())
}

fn resolve_rnnt_precision(
    cli: Option<RnntPrecisionSelection>,
    legacy_fp16: bool,
    environment: Option<&str>,
) -> Result<Option<RnntPrecisionSelection>, String> {
    let environment = environment
        .map(|value| match value {
            "fp32" => Ok(RnntPrecisionSelection::Fp32),
            "fp16" => Ok(RnntPrecisionSelection::Fp16),
            _ => Err("ASR_RNNT_PRECISION must be fp32 or fp16".to_owned()),
        })
        .transpose()?;
    if legacy_fp16 && (cli.is_some() || environment.is_some()) {
        return Err("--fp16 conflicts with --rnnt-precision and ASR_RNNT_PRECISION".into());
    }
    match (cli, environment) {
        (Some(a), Some(b)) if a != b => {
            Err("--rnnt-precision conflicts with ASR_RNNT_PRECISION".into())
        }
        (Some(value), _) | (_, Some(value)) => Ok(Some(value)),
        (None, None) => Ok(None),
    }
}

pub(crate) fn ctc_precision(
    cli: Option<gigaam_model_package::EncoderPrecision>,
    legacy_fp16: bool,
) -> Result<gigaam_model_package::EncoderPrecision, String> {
    let value = process_env("ASR_CTC_PRECISION")?;
    resolve_ctc_precision(cli, legacy_fp16, value.as_deref())
}

fn resolve_ctc_precision(
    cli: Option<gigaam_model_package::EncoderPrecision>,
    legacy_fp16: bool,
    environment: Option<&str>,
) -> Result<gigaam_model_package::EncoderPrecision, String> {
    use gigaam_model_package::EncoderPrecision;
    let environment = environment
        .map(|value| match value {
            "fp32" => Ok(EncoderPrecision::Fp32),
            "fp16io32" => Ok(EncoderPrecision::Fp16Io32),
            "fp16" => Ok(EncoderPrecision::Fp16),
            _ => Err("ASR_CTC_PRECISION must be fp32, fp16io32 or fp16".to_owned()),
        })
        .transpose()?;
    if legacy_fp16 && (cli.is_some() || environment.is_some()) {
        return Err("--fp16 conflicts with --ctc-precision and ASR_CTC_PRECISION".into());
    }
    match (cli, environment) {
        (Some(a), Some(b)) if a != b => {
            Err("--ctc-precision conflicts with ASR_CTC_PRECISION".into())
        }
        (Some(value), _) | (_, Some(value)) => Ok(value),
        (None, None) if legacy_fp16 => Ok(EncoderPrecision::Fp16Io32),
        (None, None) => Ok(EncoderPrecision::Fp32),
    }
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

/// Validates an optional `--evidence` path once, before runtime or model construction: its
/// parent directory must already exist, and the path itself must not already be a directory.
/// `None` (the option absent) always validates to `None`.
pub(crate) fn evidence_path(path: Option<PathBuf>) -> Result<Option<PathBuf>, String> {
    let Some(path) = path else {
        return Ok(None);
    };
    if path.is_dir() {
        return Err(format!("--evidence {} is a directory", path.display()));
    }
    let parent_exists = match path.parent() {
        None => true,
        Some(parent) if parent.as_os_str().is_empty() => true,
        Some(parent) => parent.is_dir(),
    };
    if !parent_exists {
        return Err(format!(
            "--evidence {} has no existing parent directory",
            path.display()
        ));
    }
    Ok(Some(path))
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
    use super::{TraceMode, evidence_path, frontend_mode_from, trace_mode_from};
    use gigaam_audio::FrontendMode;
    use std::path::PathBuf;

    #[test]
    fn ctc_precision_resolves_all_choices_and_conflicts() {
        use gigaam_model_package::EncoderPrecision;
        let choices = [
            ("fp32", EncoderPrecision::Fp32),
            ("fp16io32", EncoderPrecision::Fp16Io32),
            ("fp16", EncoderPrecision::Fp16),
        ];
        assert_eq!(
            super::resolve_ctc_precision(None, false, None),
            Ok(EncoderPrecision::Fp32)
        );
        assert_eq!(
            super::resolve_ctc_precision(None, true, None),
            Ok(EncoderPrecision::Fp16Io32)
        );
        for (text, choice) in choices {
            assert_eq!(
                super::resolve_ctc_precision(Some(choice), false, None),
                Ok(choice)
            );
            assert_eq!(
                super::resolve_ctc_precision(None, false, Some(text)),
                Ok(choice)
            );
            assert_eq!(
                super::resolve_ctc_precision(Some(choice), false, Some(text)),
                Ok(choice)
            );
            assert!(super::resolve_ctc_precision(Some(choice), true, None).is_err());
            assert!(super::resolve_ctc_precision(None, true, Some(text)).is_err());
            for (other_text, other_choice) in choices {
                if choice != other_choice {
                    assert!(
                        super::resolve_ctc_precision(Some(choice), false, Some(other_text))
                            .is_err()
                    );
                }
            }
        }
        for invalid in ["", "FP16", " fp16", "fp16 ", "other"] {
            assert!(super::resolve_ctc_precision(None, false, Some(invalid)).is_err());
        }
    }

    #[test]
    fn rnnt_precision_resolves_equal_values_and_rejects_conflicts() {
        use gigaam_recognition::RnntPrecisionSelection;
        assert_eq!(super::resolve_rnnt_precision(None, false, None), Ok(None));
        assert_eq!(
            super::resolve_rnnt_precision(Some(RnntPrecisionSelection::Fp16), false, Some("fp16")),
            Ok(Some(RnntPrecisionSelection::Fp16))
        );
        assert!(
            super::resolve_rnnt_precision(Some(RnntPrecisionSelection::Fp32), false, Some("fp16"))
                .is_err()
        );
        assert!(super::resolve_rnnt_precision(None, true, Some("fp16")).is_err());
        assert!(super::resolve_rnnt_precision(None, false, Some("")).is_err());
    }

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

    #[test]
    fn evidence_path_is_absent_by_default_and_validates_a_present_value() {
        assert_eq!(evidence_path(None), Ok(None));

        let directory = std::env::temp_dir().join(format!(
            "gigaam-cli-evidence-path-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory)
            .expect("test evidence-path directory creation must succeed");

        let bare_name = PathBuf::from("evidence.txt");
        assert_eq!(
            evidence_path(Some(bare_name.clone())),
            Ok(Some(bare_name)),
            "a bare filename has an implicit, always-present current-directory parent"
        );

        let inside_existing_directory = directory.join("evidence.txt");
        assert_eq!(
            evidence_path(Some(inside_existing_directory.clone())),
            Ok(Some(inside_existing_directory))
        );

        assert!(
            evidence_path(Some(directory.clone())).is_err(),
            "a path that is itself an existing directory must refuse"
        );

        let missing_parent = directory.join("missing-child").join("evidence.txt");
        assert!(
            evidence_path(Some(missing_parent)).is_err(),
            "a path whose parent directory does not exist must refuse"
        );

        std::fs::remove_dir_all(&directory)
            .expect("test evidence-path directory cleanup must succeed");
    }
}
