// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! GigaAM v3 Runtime Service assembly and process entrypoint.

mod admission;
mod config;
mod context;
mod health;
mod http;
mod json;
mod lifecycle;
mod log;
mod metrics;
mod protocol;
mod response;
mod router;
mod stream_response;
mod telemetry;
mod ws;

use admission::AdmissionState;
use axum::Router;
use context::ApplicationContext;
use metrics::{RuntimeGaugeSnapshot, RuntimeGaugeSource};
use std::sync::Arc;
use std::time::Duration;
use telemetry::PreparedTelemetry;

pub use admission::{RequestBodyLimit, ServiceAdmission};
pub use context::{
    ServiceApplicationParameters, ServiceCapabilities, ServiceCapabilitiesParameters,
    ServicePolicy, ServicePolicyParameters,
};

/// Runs the configured production service process and exits directly on fatal startup errors.
pub fn run_process() {
    lifecycle::run_process(|parameters, metrics, telemetry| {
        ServiceApplication::assemble_prepared(parameters, metrics, telemetry)
            .map(ServiceApplication::into_process_parts)
    });
}

/// The facade-owned dynamic gauge reader deliberately samples runtime truth only at scrape time.
struct ServiceRuntimeGaugeSource {
    admission: Arc<AdmissionState>,
    ctc: Arc<gigaam_recognition::ExecutionScheduler>,
    rnnt: Option<Arc<gigaam_recognition::ExecutionScheduler>>,
}

impl RuntimeGaugeSource for ServiceRuntimeGaugeSource {
    fn snapshot(&self) -> RuntimeGaugeSnapshot {
        RuntimeGaugeSnapshot {
            available_http: self.admission.available_http(),
            available_ws: self.admission.available_ws(),
            ctc_pending: self.ctc.pending(),
            rnnt_pending: self.rnnt.as_ref().map(|scheduler| scheduler.pending()),
        }
    }
}

/// A fully validated, independently runnable service application.
pub struct ServiceApplication {
    router: Router,
    admission: Arc<admission::AdmissionState>,
}

/// Fully validated assembly state after telemetry has started but before route ownership exists.
struct StartedAssembly {
    context: Arc<ApplicationContext>,
    admission: Arc<AdmissionState>,
    metrics: Arc<metrics::Metrics>,
    producer: telemetry::TelemetryProducer,
    request_timeout: Duration,
    request_body_limit: RequestBodyLimit,
}

/// Assembly inputs that remain unstarted until every fail-first boundary has accepted them.
struct UnstartedAssembly {
    capabilities: ServiceCapabilities,
    policy: ServicePolicy,
    admission_settings: ServiceAdmission,
    request_body_limit: RequestBodyLimit,
    metrics: Arc<metrics::Metrics>,
    telemetry: PreparedTelemetry,
}

impl ServiceApplication {
    /// Assembles private runtime context and route adapters from cohesive validated inputs.
    pub fn assemble(parameters: ServiceApplicationParameters) -> Result<Self, String> {
        let (capabilities, policy, admission, request_body_limit) =
            parameters.into_assembly_parts();
        let metrics = Arc::new(metrics::Metrics::new(
            admission.max_http(),
            admission.max_ws(),
        ));
        let telemetry = PreparedTelemetry::prepare(&admission, Arc::clone(&metrics))?;
        Self::assemble_parts(
            capabilities,
            policy,
            admission,
            request_body_limit,
            metrics,
            telemetry,
        )
    }

    fn assemble_prepared(
        parameters: ServiceApplicationParameters,
        metrics: Arc<metrics::Metrics>,
        telemetry: PreparedTelemetry,
    ) -> Result<Self, String> {
        let (capabilities, policy, admission, request_body_limit) =
            parameters.into_assembly_parts();
        Self::assemble_parts(
            capabilities,
            policy,
            admission,
            request_body_limit,
            metrics,
            telemetry,
        )
    }

    fn assemble_parts(
        capabilities: ServiceCapabilities,
        policy: ServicePolicy,
        admission_settings: ServiceAdmission,
        request_body_limit: RequestBodyLimit,
        metrics: Arc<metrics::Metrics>,
        telemetry: PreparedTelemetry,
    ) -> Result<Self, String> {
        Self::assemble_parts_with(
            UnstartedAssembly {
                capabilities,
                policy,
                admission_settings,
                request_body_limit,
                metrics,
                telemetry,
            },
            PreparedTelemetry::start,
            Self::assemble_started,
        )
    }

    /// Keeps validation, telemetry launch, and route ownership in their required fail-first order.
    fn assemble_parts_with(
        unstarted: UnstartedAssembly,
        start_telemetry: impl FnOnce(PreparedTelemetry) -> Result<telemetry::TelemetryProducer, String>,
        continue_after_start: impl FnOnce(StartedAssembly) -> Result<Self, String>,
    ) -> Result<Self, String> {
        let UnstartedAssembly {
            capabilities,
            policy,
            admission_settings,
            request_body_limit,
            metrics,
            telemetry,
        } = unstarted;
        let request_timeout = admission_settings.request_timeout();
        let admission = Arc::new(AdmissionState::new(&admission_settings)?);
        let context = Arc::new(ApplicationContext::new(capabilities, policy)?);
        if !telemetry.matches_admission(&admission_settings, admission.as_ref()) {
            return Err("telemetry capacity does not match admission".into());
        }
        let producer = start_telemetry(telemetry)?;
        continue_after_start(StartedAssembly {
            context,
            admission,
            metrics,
            producer,
            request_timeout,
            request_body_limit,
        })
    }

    /// Constructs route state only after every shared assembly boundary has succeeded.
    fn assemble_started(started: StartedAssembly) -> Result<Self, String> {
        let StartedAssembly {
            context,
            admission,
            metrics,
            producer,
            request_timeout,
            request_body_limit,
        } = started;
        let gauges: Arc<dyn RuntimeGaugeSource> = Arc::new(ServiceRuntimeGaugeSource {
            admission: Arc::clone(&admission),
            ctc: Arc::clone(&context.capabilities.ctc),
            rnnt: context.capabilities.rnnt.as_ref().map(Arc::clone),
        });
        let health = health::HealthState::new(
            Arc::clone(&admission),
            Arc::clone(&context.capabilities.ctc),
            context.capabilities.rnnt.as_ref().map(Arc::clone),
            response::HealthProjection::new(
                context.capabilities.provider,
                context.capabilities.rnnt.is_some(),
            ),
        );
        let router = router::router(router::RouterParameters {
            http: http::HttpState::new(http::HttpStateParameters {
                context: Arc::clone(&context),
                admission: Arc::clone(&admission),
                metrics: Arc::clone(&metrics),
                telemetry: producer.clone(),
                request_timeout,
                request_body_limit,
            }),
            ws: ws::WsState::new(
                Arc::clone(&context),
                Arc::clone(&admission),
                Arc::clone(&metrics),
                producer,
            ),
            health,
            metrics: metrics::MetricsState::new(metrics, gauges),
        });
        Ok(Self { router, admission })
    }

    /// Transfers the complete HTTP/WebSocket application to an embedding runtime or test.
    pub fn into_router(self) -> Router {
        self.router
    }

    pub(crate) fn into_process_parts(self) -> (Router, Arc<admission::AdmissionState>) {
        (self.router, self.admission)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{
        PreparedTelemetry, RequestBodyLimit, ServiceAdmission, ServiceApplication,
        ServiceApplicationParameters, ServiceCapabilities, ServiceCapabilitiesParameters,
        ServicePolicy, ServicePolicyParameters,
    };
    use gigaam_audio::{FeatureMatrixView, FrontendMode, FrontendProcessor, SampleRate};
    use gigaam_model_package::ModelPackage;
    use gigaam_recognition::{Decoded, Device, ExecutionScheduler, FrameRate, WindowDecoder};
    use gigaam_transcription::{ObservationMode, WindowTiming, WindowTimingObserver};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    static NEXT_ASSEMBLY_FIXTURE: AtomicU64 = AtomicU64::new(0);

    /// Test-owned package assets provide a fully validated frontend for facade assembly only.
    struct AssemblyPackage {
        root: PathBuf,
    }

    impl AssemblyPackage {
        fn new() -> Result<Self, String> {
            let sequence = NEXT_ASSEMBLY_FIXTURE.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "gigaam-service-assembly-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir(&root)
                .map_err(|error| format!("create service assembly package: {error}"))?;
            let package = Self { root };
            fs::write(
                package.root.join("config.kv"),
                include_str!("../../../model/config.kv"),
            )
            .map_err(|error| format!("write service assembly configuration: {error}"))?;
            Ok(package)
        }

        fn path(&self) -> &Path {
            &self.root
        }

        fn write_f32_asset(
            &self,
            name: &str,
            dimensions: &[usize],
            values: &[f32],
        ) -> Result<(), String> {
            let expected = dimensions.iter().try_fold(1_usize, |total, dimension| {
                total
                    .checked_mul(*dimension)
                    .ok_or_else(|| "test frontend dimensions overflow usize".to_owned())
            })?;
            if values.len() != expected {
                return Err(format!(
                    "test frontend asset {name} has {} values, expected {expected}",
                    values.len()
                ));
            }
            let dimension_count = u32::try_from(dimensions.len())
                .map_err(|_| "test frontend dimension count exceeds u32".to_owned())?;
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&dimension_count.to_le_bytes());
            for dimension in dimensions {
                let dimension = u32::try_from(*dimension)
                    .map_err(|_| "test frontend dimension exceeds u32".to_owned())?;
                bytes.extend_from_slice(&dimension.to_le_bytes());
            }
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            fs::write(self.root.join(name), bytes)
                .map_err(|error| format!("write test frontend asset {name}: {error}"))
        }
    }

    impl Drop for AssemblyPackage {
        fn drop(&mut self) {
            if let Err(error) = fs::remove_dir_all(&self.root) {
                panic!(
                    "test-owned service assembly package cleanup must succeed for {}: {error}",
                    self.root.display()
                );
            }
        }
    }

    pub(crate) struct AssemblyFixture {
        _package: AssemblyPackage,
        pub(crate) pack: Arc<ModelPackage>,
        pub(crate) frontend: Arc<FrontendProcessor>,
    }

    fn test_weight_values(length: usize, asset: &str) -> Result<Vec<f32>, String> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(length)
            .map_err(|_| format!("reserve test frontend {asset} values"))?;
        values.resize(length, 1.0);
        Ok(values)
    }

    pub(crate) fn assembly_fixture() -> Result<AssemblyFixture, String> {
        let package = AssemblyPackage::new()?;
        let pack = Arc::new(
            ModelPackage::open(package.path())
                .map_err(|error| format!("open service assembly package: {error}"))?,
        );
        let definition = pack.frontend();
        let frequency_bins = definition
            .n_fft()
            .checked_div(2)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| "test frontend frequency bins overflow usize".to_owned())?;
        let filterbank_values = frequency_bins
            .checked_mul(definition.n_mels())
            .ok_or_else(|| "test frontend filterbank values overflow usize".to_owned())?;
        let window = test_weight_values(definition.n_fft(), "window")?;
        let filterbank = test_weight_values(filterbank_values, "filterbank")?;
        package.write_f32_asset("stft_window.f32", &[definition.n_fft()], &window)?;
        package.write_f32_asset(
            "mel_fbank.f32",
            &[frequency_bins, definition.n_mels()],
            &filterbank,
        )?;
        let weights = pack
            .frontend_weights()
            .map_err(|error| format!("load service assembly frontend weights: {error}"))?;
        let frontend = Arc::new(FrontendProcessor::new(
            definition,
            weights,
            FrontendMode::Scalar,
        )?);
        Ok(AssemblyFixture {
            _package: package,
            pack,
            frontend,
        })
    }

    struct IdleDecoder;

    impl WindowDecoder for IdleDecoder {
        fn frame_rate(&self) -> FrameRate {
            FrameRate::new(25.0).expect("the fixed assembly decoder frame rate is positive")
        }

        fn decode(&mut self, features: FeatureMatrixView<'_>) -> Result<Decoded, String> {
            Decoded::new(
                Vec::new(),
                vec![false; features.frames()],
                features.frames(),
                0.0,
            )
        }
    }

    struct RecordingWindowObserver {
        records: Mutex<Vec<WindowTiming>>,
    }

    impl RecordingWindowObserver {
        fn count(&self) -> usize {
            match self.records.lock() {
                Ok(records) => records.len(),
                Err(_) => panic!("caller observation records must not be poisoned"),
            }
        }
    }

    impl WindowTimingObserver for RecordingWindowObserver {
        fn observe(&self, observation: WindowTiming) {
            match self.records.lock() {
                Ok(mut records) => records.push(observation),
                Err(_) => panic!("caller observation records must not be poisoned"),
            }
        }
    }

    fn service_policy(
        sample_rate: SampleRate,
        observations: ObservationMode,
    ) -> Result<ServicePolicy, String> {
        ServicePolicy::new(ServicePolicyParameters {
            model_sample_rate: sample_rate,
            window_seconds: 1.0,
            overlap_seconds: 0.0,
            dedup_default: true,
            dedup_window_samples: 1,
            dedup_threshold: 0.99,
            observations,
            backchannel_max_seconds: 0.0,
        })
    }

    fn service_capabilities(fixture: &AssemblyFixture) -> Result<ServiceCapabilities, String> {
        ServiceCapabilities::new(ServiceCapabilitiesParameters {
            pack: Arc::clone(&fixture.pack),
            frontend: Arc::clone(&fixture.frontend),
            ctc: Arc::new(ExecutionScheduler::spawn(IdleDecoder)),
            rnnt: None,
            provider: Device::Cpu,
            intra_threads: None,
        })
    }

    fn valid_parameters(
        fixture: &AssemblyFixture,
        observations: ObservationMode,
    ) -> Result<ServiceApplicationParameters, String> {
        let admission = ServiceAdmission::new(1, 1, Duration::from_secs(1))?;
        Ok(ServiceApplicationParameters::new(
            service_capabilities(fixture)?,
            service_policy(fixture.frontend.sample_rate(), observations)?,
            admission,
            RequestBodyLimit::new(8_192)?,
        ))
    }

    fn pcm16_wav_silence(sample_rate: u32, frames: usize) -> Result<Vec<u8>, String> {
        let data_length = frames
            .checked_mul(2)
            .ok_or_else(|| "test WAV data length overflows usize".to_owned())?;
        let data_length = u32::try_from(data_length)
            .map_err(|_| "test WAV data length exceeds u32".to_owned())?;
        let riff_length = data_length
            .checked_add(36)
            .ok_or_else(|| "test WAV RIFF length overflows u32".to_owned())?;
        let byte_rate = sample_rate
            .checked_mul(2)
            .ok_or_else(|| "test WAV byte rate overflows u32".to_owned())?;
        let reserve = usize::try_from(data_length)
            .map_err(|_| "test WAV data length does not fit usize".to_owned())?
            .checked_add(44)
            .ok_or_else(|| "test WAV buffer length overflows usize".to_owned())?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(reserve)
            .map_err(|_| "reserve test WAV payload".to_owned())?;
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&riff_length.to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&byte_rate.to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_length.to_le_bytes());
        bytes.resize(reserve, 0);
        Ok(bytes)
    }

    #[test]
    fn facade_assembly_refuses_policy_frontend_rate_mismatch_before_routes_exist()
    -> Result<(), String> {
        let fixture = assembly_fixture()?;
        let policy_rate = SampleRate::new(8_000)?;
        assert_ne!(
            fixture.frontend.sample_rate(),
            policy_rate,
            "the assembly fixture must exercise distinct policy and frontend rates"
        );
        let capabilities = ServiceCapabilities::new(ServiceCapabilitiesParameters {
            pack: Arc::clone(&fixture.pack),
            frontend: Arc::clone(&fixture.frontend),
            ctc: Arc::new(ExecutionScheduler::spawn(IdleDecoder)),
            rnnt: None,
            provider: Device::Cpu,
            intra_threads: None,
        })?;
        let policy = ServicePolicy::new(ServicePolicyParameters {
            model_sample_rate: policy_rate,
            window_seconds: 1.0,
            overlap_seconds: 0.0,
            dedup_default: true,
            dedup_window_samples: 1,
            dedup_threshold: 0.99,
            observations: ObservationMode::disabled(),
            backchannel_max_seconds: 0.0,
        })?;
        let admission = ServiceAdmission::new(1, 1, Duration::from_secs(1))?;
        let request_body_limit = RequestBodyLimit::new(1)?;
        let result = ServiceApplication::assemble(ServiceApplicationParameters::new(
            capabilities,
            policy,
            admission,
            request_body_limit,
        ));
        let error = match result {
            Ok(_) => {
                return Err(
                    "a policy/frontend mismatch must refuse before a route application is returned"
                        .into(),
                );
            }
            Err(error) => error,
        };
        assert_eq!(error, "service policy sample rate must match the frontend");
        Ok(())
    }

    #[test]
    fn common_assembly_refusals_stop_before_telemetry_start_or_route_construction()
    -> Result<(), String> {
        use std::sync::atomic::AtomicUsize;

        let fixture = assembly_fixture()?;

        let invalid_context_admission = ServiceAdmission::new(1, 1, Duration::from_secs(1))?;
        let invalid_context_metrics = Arc::new(crate::metrics::Metrics::new(1, 1));
        let invalid_context_telemetry = PreparedTelemetry::prepare(
            &invalid_context_admission,
            Arc::clone(&invalid_context_metrics),
        )?;
        let invalid_context_starts = AtomicUsize::new(0);
        let invalid_context_routes = AtomicUsize::new(0);
        let invalid_context = ServiceApplication::assemble_parts_with(
            super::UnstartedAssembly {
                capabilities: service_capabilities(&fixture)?,
                policy: service_policy(SampleRate::new(8_000)?, ObservationMode::disabled())?,
                admission_settings: invalid_context_admission,
                request_body_limit: RequestBodyLimit::new(8_192)?,
                metrics: invalid_context_metrics,
                telemetry: invalid_context_telemetry,
            },
            |telemetry| {
                invalid_context_starts.fetch_add(1, Ordering::Relaxed);
                telemetry.start()
            },
            |_| {
                invalid_context_routes.fetch_add(1, Ordering::Relaxed);
                Err("route construction must not follow an invalid context".to_owned())
            },
        );
        let invalid_context_error = match invalid_context {
            Ok(_) => return Err("invalid context must refuse common assembly".to_owned()),
            Err(error) => error,
        };
        assert_eq!(
            invalid_context_error,
            "service policy sample rate must match the frontend"
        );
        assert_eq!(invalid_context_starts.load(Ordering::Relaxed), 0);
        assert_eq!(invalid_context_routes.load(Ordering::Relaxed), 0);

        let mismatch_admission = ServiceAdmission::new(1, 1, Duration::from_secs(1))?;
        let mismatched_preparation_admission = ServiceAdmission::new(2, 1, Duration::from_secs(1))?;
        let mismatch_metrics = Arc::new(crate::metrics::Metrics::new(1, 1));
        let mismatched_telemetry = PreparedTelemetry::prepare(
            &mismatched_preparation_admission,
            Arc::clone(&mismatch_metrics),
        )?;
        let mismatch_starts = AtomicUsize::new(0);
        let mismatch_routes = AtomicUsize::new(0);
        let mismatch = ServiceApplication::assemble_parts_with(
            super::UnstartedAssembly {
                capabilities: service_capabilities(&fixture)?,
                policy: service_policy(
                    fixture.frontend.sample_rate(),
                    ObservationMode::disabled(),
                )?,
                admission_settings: mismatch_admission,
                request_body_limit: RequestBodyLimit::new(8_192)?,
                metrics: mismatch_metrics,
                telemetry: mismatched_telemetry,
            },
            |telemetry| {
                mismatch_starts.fetch_add(1, Ordering::Relaxed);
                telemetry.start()
            },
            |_| {
                mismatch_routes.fetch_add(1, Ordering::Relaxed);
                Err("route construction must not follow a telemetry mismatch".to_owned())
            },
        );
        let mismatch_error = match mismatch {
            Ok(_) => return Err("telemetry mismatch must refuse common assembly".to_owned()),
            Err(error) => error,
        };
        assert_eq!(
            mismatch_error,
            "telemetry capacity does not match admission"
        );
        assert_eq!(mismatch_starts.load(Ordering::Relaxed), 0);
        assert_eq!(mismatch_routes.load(Ordering::Relaxed), 0);

        let launch_admission = ServiceAdmission::new(1, 1, Duration::from_secs(1))?;
        let launch_metrics = Arc::new(crate::metrics::Metrics::new(1, 1));
        let launch_telemetry =
            PreparedTelemetry::prepare(&launch_admission, Arc::clone(&launch_metrics))?;
        let launch_starts = AtomicUsize::new(0);
        let launch_routes = AtomicUsize::new(0);
        let launcher_refusal = ServiceApplication::assemble_parts_with(
            super::UnstartedAssembly {
                capabilities: service_capabilities(&fixture)?,
                policy: service_policy(
                    fixture.frontend.sample_rate(),
                    ObservationMode::disabled(),
                )?,
                admission_settings: launch_admission,
                request_body_limit: RequestBodyLimit::new(8_192)?,
                metrics: launch_metrics,
                telemetry: launch_telemetry,
            },
            |telemetry| {
                launch_starts.fetch_add(1, Ordering::Relaxed);
                telemetry.refuse_start_for_test()
            },
            |_| {
                launch_routes.fetch_add(1, Ordering::Relaxed);
                Err("route construction must not follow launcher refusal".to_owned())
            },
        );
        let launcher_error = match launcher_refusal {
            Ok(_) => return Err("launcher refusal must refuse common assembly".to_owned()),
            Err(error) => error,
        };
        assert_eq!(launcher_error, "controlled telemetry launcher refusal");
        assert_eq!(launch_starts.load(Ordering::Relaxed), 1);
        assert_eq!(launch_routes.load(Ordering::Relaxed), 0);
        Ok(())
    }

    #[tokio::test]
    async fn public_assembly_keeps_the_caller_observer_on_a_real_loopback_batch_request()
    -> Result<(), String> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::{TcpListener, TcpStream};

        let fixture = assembly_fixture()?;
        let observer = Arc::new(RecordingWindowObserver {
            records: Mutex::new(Vec::new()),
        });
        let router = ServiceApplication::assemble(valid_parameters(
            &fixture,
            ObservationMode::enabled(observer.clone()),
        )?)?
        .into_router();
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .map_err(|error| format!("bind service loopback listener: {error}"))?;
        let address = listener
            .local_addr()
            .map_err(|error| format!("read service loopback listener address: {error}"))?;
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    shutdown_rx
                        .await
                        .expect("the service loopback test sends one shutdown signal");
                })
                .await
        });

        let body = pcm16_wav_silence(fixture.frontend.sample_rate().hertz(), 640)?;
        let mut client = TcpStream::connect(address)
            .await
            .map_err(|error| format!("connect service loopback client: {error}"))?;
        let request = format!(
            "POST /v1/transcribe?ext=wav HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\nx-request-id: caller-observer\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        client
            .write_all(request.as_bytes())
            .await
            .map_err(|error| format!("write service loopback request headers: {error}"))?;
        client
            .write_all(&body)
            .await
            .map_err(|error| format!("write service loopback request body: {error}"))?;
        let mut response = Vec::new();
        client
            .read_to_end(&mut response)
            .await
            .map_err(|error| format!("read service loopback response: {error}"))?;
        let response = String::from_utf8(response)
            .map_err(|error| format!("service loopback response must be UTF-8: {error}"))?;
        assert!(
            response.starts_with("HTTP/1.1 200"),
            "the public service facade must preserve successful batch response status: {response}"
        );
        assert!(
            response.contains("x-request-id: caller-observer"),
            "the public service facade must preserve the caller request identifier: {response}"
        );
        assert_eq!(
            observer.count(),
            1,
            "the caller observer must receive the successful production batch window"
        );

        shutdown_tx
            .send(())
            .map_err(|_| "service loopback listener shutdown receiver dropped".to_owned())?;
        let stopped = server
            .await
            .map_err(|error| format!("service loopback server task failed: {error}"))?;
        stopped.map_err(|error| format!("service loopback server failed: {error}"))?;
        Ok(())
    }
}
