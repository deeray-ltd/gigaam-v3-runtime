// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Process-only assembly, startup warmup, listener lifetime, and bounded drain.

use crate::admission::AdmissionState;
use crate::config::{
    ModelBoundLimits, OVERLAP_DURATION, ServerLimits, Startup, WINDOW_DURATION, encoder_precision,
    precision_name,
};
use crate::context::{
    ServiceApplicationParameters, ServiceCapabilities, ServiceCapabilitiesParameters,
    ServicePolicy, ServicePolicyParameters, model_sample_rate,
};
use crate::metrics::Metrics;
use crate::telemetry::PreparedTelemetry;
use axum::Router;
use gigaam_audio::{FrontendProcessor, SampleRate};
use gigaam_model_package::ModelPackage;
use gigaam_recognition::{
    CudaAssignmentEvidence, DirectRecognizer, EncoderRole, ExecutionControl, ExecutionScheduler,
    ExecutionState, ProviderPlan, WindowDecoder, init_runtime,
};
use gigaam_transcription::BatchConfig;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

/// All process inputs which must validate before runtime/model side effects begin.
struct ProcessConfiguration {
    startup: Startup,
    limits: ServerLimits,
}

impl ProcessConfiguration {
    fn from_process() -> Result<Self, String> {
        let startup = Startup::from_process()?;
        let limits = ServerLimits::from_process()?;
        Ok(Self { startup, limits })
    }
}

/// Production-used fail-first continuation boundary for independent process configuration.
fn continue_after_configuration<T>(
    configuration: Result<ProcessConfiguration, String>,
    continuation: impl FnOnce(ProcessConfiguration) -> Result<T, String>,
) -> Result<T, String> {
    continuation(configuration?)
}

/// Test seam for fail-first model-rate-dependent process-limit binding.
#[cfg(test)]
fn continue_after_model_limits<T>(
    limits: ServerLimits,
    sample_rate: SampleRate,
    continuation: impl FnOnce(ModelBoundLimits) -> Result<T, String>,
) -> Result<T, String> {
    continuation(limits.bind_model(sample_rate)?)
}

/// Runs a typed package-open operation before the continuation can initialize native runtime work.
fn continue_after_package_open<P, T>(
    configuration: ProcessConfiguration,
    open_package: impl FnOnce(&Startup) -> Result<P, String>,
    continuation: impl FnOnce(ProcessConfiguration, P) -> Result<T, String>,
) -> Result<T, String> {
    let package = open_package(&configuration.startup)?;
    continuation(configuration, package)
}

/// Binds model-dependent limits and application policy before any native continuation.
enum ModelPolicyFailure {
    ModelLimits(String),
    Policy(String),
    Continuation(String),
}

fn continue_after_model_policy<T, P>(
    limits: ServerLimits,
    sample_rate: SampleRate,
    make_policy: impl FnOnce(&ModelBoundLimits) -> Result<P, String>,
    continuation: impl FnOnce(ModelBoundLimits, P) -> Result<T, String>,
) -> Result<T, ModelPolicyFailure> {
    let model_limits = limits
        .bind_model(sample_rate)
        .map_err(ModelPolicyFailure::ModelLimits)?;
    let policy = make_policy(&model_limits).map_err(ModelPolicyFailure::Policy)?;
    continuation(model_limits, policy).map_err(ModelPolicyFailure::Continuation)
}

/// Opens the typed package before any runtime initialization can create native state.
fn open_model(
    configuration: ProcessConfiguration,
) -> Result<(Startup, ServerLimits, Arc<ModelPackage>), String> {
    continue_after_package_open(
        configuration,
        |startup| {
            ModelPackage::open(Path::new(&startup.model)).map_err(|error| format!("model: {error}"))
        },
        |configuration, pack| {
            let ProcessConfiguration { startup, limits } = configuration;
            Ok((startup, limits, Arc::new(pack)))
        },
    )
}

/// Emits the vocabulary-derived endpointing notice at the service process boundary.
fn emit_ctc_construction_notice(decoder: &DirectRecognizer) {
    if let Some(notice) = decoder.ctc_construction_notice() {
        eprintln!("{}", notice.message());
    }
}

/// Emits the observed placement evidence when the operator explicitly permits an unverified CUDA
/// graph assignment.
fn emit_unverified_cuda_assignment(
    plan: &ProviderPlan,
    role: EncoderRole,
    evidence: Option<&CudaAssignmentEvidence>,
) {
    if let Some(policy) = plan.cuda_assignment_policy()
        && policy.is_allow_unverified()
    {
        let evidence = evidence.expect(
            "allow-unverified CUDA encoder construction must retain graph-assignment evidence",
        );
        eprintln!(
            "# CUDA assignment unverified role={} sha256={} cpu_nodes={} cuda_nodes={}",
            role.as_str(),
            evidence.fingerprint().to_hex(),
            evidence.cpu_assignments(),
            evidence.cuda_assignments()
        );
    }
}

/// Warm the fixed window shape with one scheduled noise decode. A warmup failure prevents startup.
fn warmup(
    scheduler: &ExecutionScheduler,
    frontend: &FrontendProcessor,
    batch_config: BatchConfig,
    name: &str,
) -> Result<(), String> {
    let samples = batch_config.full_window_samples();
    let mel = frontend.log_mel(&vec![0.0_f32; samples])?;
    let control = ExecutionControl::without_deadline();
    match scheduler.window_channel(control.clone()).decode(mel.view()) {
        Ok(_) => match control.complete() {
            ExecutionState::Completed => Ok(()),
            ExecutionState::Cancelled => Err(format!("warmup {name}: execution cancelled")),
            state => Err(format!("warmup {name}: execution ended in {state:?}")),
        },
        Err(error) => {
            control.fail();
            Err(format!("warmup {name}: {error}"))
        }
    }
}

/// Runs the process lifecycle and delegates application assembly to the crate facade.
pub(super) fn run_process(
    assemble: impl FnOnce(
        ServiceApplicationParameters,
        Arc<Metrics>,
        PreparedTelemetry,
    ) -> Result<(Router, Arc<AdmissionState>), String>,
) {
    let (startup, limits, pack) =
        continue_after_configuration(ProcessConfiguration::from_process(), open_model)
            .unwrap_or_else(|error| fatal(&error));
    let sample_rate_config = pack.frontend().sample_rate();
    let sample_rate = model_sample_rate(sample_rate_config).unwrap_or_else(|error| fatal(&error));
    let window_seconds = WINDOW_DURATION.as_secs_f32();
    let overlap_seconds = OVERLAP_DURATION.as_secs_f32();
    let (limits, policy, metrics, telemetry) = match continue_after_model_policy(
        limits,
        sample_rate,
        |model_limits| {
            let metrics = Arc::new(Metrics::new(
                model_limits.limits.admission().max_http(),
                model_limits.limits.admission().max_ws(),
            ));
            let telemetry =
                PreparedTelemetry::prepare(model_limits.limits.admission(), Arc::clone(&metrics))?;
            let policy = ServicePolicy::new(ServicePolicyParameters {
                model_sample_rate: sample_rate,
                window_seconds,
                overlap_seconds,
                dedup_default: model_limits.limits.dedup_default,
                dedup_window_samples: model_limits.dedup_window_samples.get(),
                dedup_threshold: model_limits.limits.dedup_threshold,
                observations: telemetry.observations(startup.trace_enabled),
                backchannel_max_seconds: model_limits.limits.backchannel_max.as_secs_f32(),
            })?;
            Ok((policy, metrics, telemetry))
        },
        |model_limits, (policy, metrics, telemetry)| {
            Ok((model_limits.limits, policy, metrics, telemetry))
        },
    ) {
        Ok(value) => value,
        Err(ModelPolicyFailure::ModelLimits(error))
        | Err(ModelPolicyFailure::Continuation(error)) => fatal(&error),
        Err(ModelPolicyFailure::Policy(error)) => fatal(&format!("service policy: {error}")),
    };

    init_runtime(&startup.ort_dylib).unwrap_or_else(|error| fatal(&error));
    let weights = pack
        .frontend_weights()
        .unwrap_or_else(|error| fatal(&format!("frontend weights: {error}")));
    let frontend = Arc::new(
        FrontendProcessor::new(pack.frontend(), weights, startup.frontend_mode)
            .unwrap_or_else(|error| fatal(&format!("frontend: {error}"))),
    );

    let ctc_decoder =
        DirectRecognizer::ctc(&pack, &startup.plan, encoder_precision(startup.ctc_fp16))
            .unwrap_or_else(|error| fatal(&format!("CTC decoder: {error}")));
    let ctc_precision = precision_name(ctc_decoder.encoder_precision());
    emit_ctc_construction_notice(&ctc_decoder);
    emit_unverified_cuda_assignment(
        &startup.plan,
        EncoderRole::Ctc,
        ctc_decoder.assignment_evidence(),
    );
    let ctc = Arc::new(ExecutionScheduler::spawn(ctc_decoder));
    warmup(&ctc, &frontend, policy.http_batch, "CTC").unwrap_or_else(|error| fatal(&error));
    eprintln!(
        "# CTC worker started ({ctc_precision}, {})",
        startup.plan.device().as_str()
    );

    let rnnt = if startup.with_rnnt {
        let decoder =
            DirectRecognizer::rnnt(&pack, &startup.plan, encoder_precision(startup.rnnt_fp16))
                .unwrap_or_else(|error| fatal(&format!("RNNT decoder: {error}")));
        let precision = precision_name(decoder.encoder_precision());
        emit_unverified_cuda_assignment(
            &startup.plan,
            EncoderRole::Rnnt,
            decoder.assignment_evidence(),
        );
        let scheduler = Arc::new(ExecutionScheduler::spawn(decoder));
        warmup(&scheduler, &frontend, policy.http_batch, "RNNT")
            .unwrap_or_else(|error| fatal(&error));
        eprintln!(
            "# RNNT worker started ({precision}, {})",
            startup.plan.device().as_str()
        );
        Some(scheduler)
    } else {
        None
    };

    eprintln!(
        "# limits: transcribe≤{}, channels≤{}, body≤{}MB, timeout {}s; dedup {}",
        limits.admission().max_http(),
        limits.admission().max_ws(),
        limits.body_megabytes(),
        limits.admission().request_timeout().as_secs(),
        if limits.dedup_default { "on" } else { "off" }
    );

    let capabilities = ServiceCapabilities::new(ServiceCapabilitiesParameters {
        pack,
        frontend,
        ctc,
        rnnt,
        provider: startup.plan.device(),
        intra_threads: startup.plan.intra_threads(),
    })
    .unwrap_or_else(|error| fatal(&format!("service capabilities: {error}")));
    let request_body_limit = limits.request_body_limit();
    let admission = limits.admission;
    let (app, admission) = assemble(
        ServiceApplicationParameters::new(capabilities, policy, admission, request_body_limit),
        metrics,
        telemetry,
    )
    .unwrap_or_else(|error| fatal(&format!("service application: {error}")));

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap_or_else(|error| fatal(&error.to_string()));
    runtime.block_on(async move {
        let address = format!("0.0.0.0:{}", startup.port);
        let listener = tokio::net::TcpListener::bind(&address)
            .await
            .unwrap_or_else(|error| fatal(&format!("bind {address}: {error}")));
        eprintln!("# asr-serve listening on http://{address}  (POST /v1/transcribe, GET /health, /livez, /readyz)");
        let shutdown = async move {
            wait_signal().await.unwrap_or_else(|error| fatal(&error));
            crate::admission::begin_draining_then(admission.as_ref(), || {
                eprintln!("# termination signal — graceful shutdown (draining in-flight work, /readyz -> 503)");
            });
            admission.wait_for_ws_drain(Duration::from_secs(10)).await;
        };
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown)
            .await
            .unwrap_or_else(|error| fatal(&error.to_string()));
        eprintln!("# server stopped");
    });
}

async fn wait_signal() -> Result<(), String> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut terminate = signal(SignalKind::terminate())
            .map_err(|error| format!("install SIGTERM handler: {error}"))?;
        let mut interrupt = signal(SignalKind::interrupt())
            .map_err(|error| format!("install SIGINT handler: {error}"))?;
        tokio::select! {
            _ = terminate.recv() => {},
            _ = interrupt.recv() => {},
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c()
            .await
            .map_err(|error| format!("install Ctrl-C handler: {error}"))
    }
}

fn fatal(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1)
}

#[cfg(test)]
mod tests {
    use super::{
        ModelPolicyFailure, ProcessConfiguration, continue_after_configuration,
        continue_after_model_limits, continue_after_model_policy, continue_after_package_open,
    };
    use crate::config::{ServerLimitEnvironment, ServerLimits, Startup, StartupEnvironment};
    use crate::context::{ServicePolicy, ServicePolicyParameters};
    use gigaam_audio::SampleRate;
    use gigaam_transcription::ObservationMode;
    use std::cell::{Cell, RefCell};

    fn startup() -> Result<Startup, String> {
        Startup::parse(
            &[
                "--ort-dylib".to_owned(),
                "/runtime/libonnxruntime.so".to_owned(),
            ],
            StartupEnvironment {
                encoder_ep: Some("cpu"),
                ..StartupEnvironment::default()
            },
        )
    }

    #[test]
    fn invalid_independent_limits_prevent_the_runtime_model_continuation() {
        let invoked = Cell::new(false);
        let configuration = (|| {
            Ok(ProcessConfiguration {
                startup: startup()?,
                limits: ServerLimits::parse(ServerLimitEnvironment {
                    max_http: Some("0"),
                    ..ServerLimitEnvironment::default()
                })?,
            })
        })();
        let result = continue_after_configuration(configuration, |_| {
            invoked.set(true);
            Err::<(), _>("the ORT continuation must not run".to_owned())
        });
        assert_eq!(
            result,
            Err("ASR_MAX_CONCURRENCY must be a positive integer".to_owned())
        );
        assert!(
            !invoked.get(),
            "invalid process limits must win before ORT/model continuation"
        );
    }

    #[test]
    fn invalid_model_bound_limit_prevents_frontend_session_continuation() {
        let invoked = Cell::new(false);
        let limits = ServerLimits::parse(ServerLimitEnvironment {
            dedup_window: Some("4294967298"),
            ..ServerLimitEnvironment::default()
        })
        .expect("the independent configuration is valid");
        let sample_rate = SampleRate::new(u32::MAX)
            .expect("the largest Audio sample rate is still a validated nonzero rate");
        let result = continue_after_model_limits(limits, sample_rate, |_| {
            invoked.set(true);
            Err::<(), _>("the frontend continuation must not run".to_owned())
        });
        assert_eq!(
            result,
            Err("ASR_DEDUP_WINDOW_SEC sample count exceeds usize".to_owned())
        );
        assert!(
            !invoked.get(),
            "model-bound refusal must win before frontend/session construction"
        );
    }

    fn policy_parameters(
        sample_rate: SampleRate,
        dedup_window_samples: usize,
        window_seconds: f32,
    ) -> ServicePolicyParameters {
        ServicePolicyParameters {
            model_sample_rate: sample_rate,
            window_seconds,
            overlap_seconds: 0.0,
            dedup_default: true,
            dedup_window_samples,
            dedup_threshold: 0.99,
            observations: ObservationMode::disabled(),
            backchannel_max_seconds: 0.0,
        }
    }

    #[test]
    fn typed_package_open_precedes_the_production_native_continuation_boundary() {
        let trace = RefCell::new(Vec::new());
        let configuration = ProcessConfiguration {
            startup: startup().expect("test startup configuration is valid"),
            limits: ServerLimits::parse(ServerLimitEnvironment::default())
                .expect("test process limits are valid"),
        };
        let result = continue_after_package_open(
            configuration,
            |_| {
                trace.borrow_mut().push("package");
                Ok(())
            },
            |_, ()| {
                trace.borrow_mut().push("native");
                Ok(())
            },
        );
        assert_eq!(result, Ok(()));
        assert_eq!(*trace.borrow(), vec!["package", "native"]);
    }

    #[test]
    fn invalid_bound_service_policy_prevents_native_continuation() {
        let native_started = Cell::new(false);
        let sample_rate = SampleRate::new(16)
            .expect("the documented test model rate is a valid Audio sample rate");
        let result = continue_after_model_policy(
            ServerLimits::parse(ServerLimitEnvironment::default())
                .expect("test process limits are valid"),
            sample_rate,
            |_| ServicePolicy::new(policy_parameters(sample_rate, 64, 0.0)),
            |_, _| {
                native_started.set(true);
                Ok(())
            },
        );
        match result {
            Err(ModelPolicyFailure::Policy(error)) => {
                assert_eq!(error, "service window must be finite and greater than zero");
            }
            Ok(()) => panic!("an invalid bound policy must refuse before native continuation"),
            Err(ModelPolicyFailure::ModelLimits(error))
            | Err(ModelPolicyFailure::Continuation(error)) => {
                panic!("unexpected failure origin: {error}")
            }
        }
        assert!(!native_started.get());
    }

    #[test]
    fn model_bound_process_limit_precedes_a_simultaneous_invalid_service_policy() {
        let policy_attempted = Cell::new(false);
        let native_started = Cell::new(false);
        let sample_rate = SampleRate::new(u32::MAX)
            .expect("the largest Audio sample rate is still a validated nonzero rate");
        let limits = ServerLimits::parse(ServerLimitEnvironment {
            dedup_window: Some("4294967298"),
            ..ServerLimitEnvironment::default()
        })
        .expect("the independent configuration is valid");
        let result = continue_after_model_policy(
            limits,
            sample_rate,
            |_| {
                policy_attempted.set(true);
                ServicePolicy::new(policy_parameters(sample_rate, 64, 0.0))
            },
            |_, _| {
                native_started.set(true);
                Ok(())
            },
        );
        match result {
            Err(ModelPolicyFailure::ModelLimits(error)) => {
                assert_eq!(error, "ASR_DEDUP_WINDOW_SEC sample count exceeds usize");
            }
            Ok(()) => panic!("invalid model-bound limits must refuse before native continuation"),
            Err(ModelPolicyFailure::Policy(error))
            | Err(ModelPolicyFailure::Continuation(error)) => {
                panic!("unexpected failure origin: {error}")
            }
        }
        assert!(!policy_attempted.get());
        assert!(!native_started.get());
    }
}
