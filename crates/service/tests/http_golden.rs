// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! End-to-end HTTP acceptance: the service batch route preserves the CTC golden text.
#[path = "../../../tests/support/mod.rs"]
mod external_artifacts;

use gigaam_audio::{FrontendMode, FrontendProcessor, SampleRate};
use gigaam_model_package::{EncoderPrecision, ModelPackage};
use gigaam_recognition::{
    Device, DirectRecognizer, ExecutionScheduler, OrtConfig, ProviderPlan, init_runtime,
};
use gigaam_service::{
    RequestBodyLimit, ServiceAdmission, ServiceApplication, ServiceApplicationParameters,
    ServiceCapabilities, ServiceCapabilitiesParameters, ServicePolicy, ServicePolicyParameters,
};
use gigaam_transcription::ObservationMode;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

fn runtime_path() -> Result<PathBuf, String> {
    let value = std::env::var("ORT_DYLIB_PATH")
        .map_err(|error| format!("ORT_DYLIB_PATH must name the ONNX Runtime library: {error}"))?;
    if value.is_empty() {
        return Err("ORT_DYLIB_PATH must not be empty".into());
    }
    Ok(PathBuf::from(value))
}

fn application(root: &Path) -> ServiceApplication {
    let dylib = runtime_path().expect("the golden suite requires an explicit ORT library path");
    let pack =
        Arc::new(ModelPackage::open(&root.join("model")).expect("the test model pack must open"));
    let model_sample_rate = SampleRate::from_usize(pack.frontend().sample_rate(), "test model")
        .expect("the test model rate must fit Audio's validated rate");
    let dedup_window_samples = pack
        .frontend()
        .sample_rate()
        .checked_mul(4)
        .expect("the test model sample rate supports its four-second deduplication window");
    let policy = ServicePolicy::new(ServicePolicyParameters {
        model_sample_rate,
        window_seconds: 30.0,
        overlap_seconds: 6.0,
        dedup_default: true,
        dedup_window_samples,
        dedup_threshold: 0.99,
        observations: ObservationMode::disabled(),
        backchannel_max_seconds: 0.0,
    })
    .expect("the golden service policy must be valid");
    init_runtime(&dylib).expect("the explicit ORT library must initialize");
    let frontend = Arc::new(
        FrontendProcessor::new(
            pack.frontend(),
            pack.frontend_weights()
                .expect("the test model pack must expose frontend weights"),
            FrontendMode::Scalar,
        )
        .expect("the test model pack frontend must be supported"),
    );
    let plan = ProviderPlan::new(Device::Cpu, OrtConfig::default())
        .expect("the CPU provider plan must be valid");
    let ctc = Arc::new(ExecutionScheduler::spawn(
        DirectRecognizer::ctc(&pack, &plan, EncoderPrecision::Fp32)
            .expect("the CTC decoder must match the test pack"),
    ));
    let capabilities = ServiceCapabilities::new(ServiceCapabilitiesParameters {
        pack,
        frontend,
        ctc,
        rnnt: None,
        provider: Device::Cpu,
        intra_threads: None,
    })
    .expect("the golden capabilities must share one model sample rate");
    let admission = ServiceAdmission::new(1, 1, Duration::from_secs(120))
        .expect("the golden admission policy must be valid");
    let request_body_limit = RequestBodyLimit::new(64 * 1024 * 1024)
        .expect("the golden request body limit must be valid");
    ServiceApplication::assemble(ServiceApplicationParameters::new(
        capabilities,
        policy,
        admission,
        request_body_limit,
    ))
    .expect("the golden service application must assemble")
}

#[tokio::test]
async fn http_short_clip_ctc_text_matches_the_unpadded_golden() {
    let root = external_artifacts::external_artifact_root();
    let audio = std::fs::read(root.join("fixtures/example.wav"))
        .expect("the HTTP golden test WAV must be available");
    let app = application(&root).into_router();
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("the loopback test listener must bind");
    let address = listener
        .local_addr()
        .expect("the loopback listener must have an address");
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                shutdown_rx
                    .await
                    .expect("the test must send the server shutdown signal");
            })
            .await
    });

    let mut invalid_stream = TcpStream::connect(address)
        .await
        .expect("the test client must connect to the loopback listener");
    let invalid_request = format!(
        "POST /v1/transcribe?unknown=value HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\nContent-Length: 1\r\n\r\nx"
    );
    invalid_stream
        .write_all(invalid_request.as_bytes())
        .await
        .expect("the invalid-query request must write");
    let mut invalid_response = Vec::new();
    invalid_stream
        .read_to_end(&mut invalid_response)
        .await
        .expect("the invalid-query response must read");
    let invalid_response =
        String::from_utf8(invalid_response).expect("the HTTP error response must be UTF-8 text");
    assert!(
        invalid_response.starts_with("HTTP/1.1 400"),
        "unknown query parameters must be rejected before decoding: {invalid_response}"
    );

    let mut stream = TcpStream::connect(address)
        .await
        .expect("the test client must connect to the loopback listener");
    let request = format!(
        "POST /v1/transcribe?model=ctc HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\nContent-Length: {}\r\n\r\n",
        audio.len()
    );
    stream
        .write_all(request.as_bytes())
        .await
        .expect("the valid HTTP request headers must write");
    stream
        .write_all(&audio)
        .await
        .expect("the valid HTTP request body must write");
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .await
        .expect("the valid HTTP response must read");
    let shutdown_result = shutdown_tx.send(());
    let server_result = server.await.expect("HTTP server task panicked");
    server_result.expect("HTTP server failed");
    assert!(
        shutdown_result.is_ok(),
        "HTTP server exited before the test could stop it"
    );

    let response = String::from_utf8(response).expect("the HTTP response must be UTF-8 text");
    let (headers, body) = response
        .split_once("\r\n\r\n")
        .expect("HTTP response must include headers and a body");
    assert!(
        headers.starts_with("HTTP/1.1 200"),
        "unexpected response: {response}"
    );
    let text = body
        .rsplit_once("\"text\":\"")
        .and_then(|(_, text)| text.strip_suffix("\"}"))
        .expect("minimal CTC response must end with its text field");
    let golden = std::fs::read_to_string(root.join("fixtures/example.ctc.gold.txt"))
        .expect("the CTC golden text must be available");
    assert_eq!(
        text, golden,
        "HTTP batch CTC text diverged from golden: {body}"
    );
}
