// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Observable production-CLI witnesses for package, policy, and native effect order.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMPORARY_ROOT: AtomicU64 = AtomicU64::new(0);

const PRODUCT_ENVIRONMENT_KEYS: [&str; 14] = [
    "ASR_ENCODER_EP",
    "ASR_ORT_MEMPATTERN",
    "ASR_ORT_ARENA",
    "ASR_ORT_THREADS",
    "ASR_TRT_CACHE",
    "ASR_TRT_PROFILE_MIN",
    "ASR_TRT_PROFILE_OPT",
    "ASR_TRT_PROFILE_MAX",
    "ORT_DYLIB_PATH",
    "ASR_FRONTEND",
    "ASR_TRACE",
    "ASR_CUDA_ASSIGNMENT_POLICY",
    "ASR_CUDA_CTC_ASSIGNMENT_SHA256",
    "ASR_CUDA_RNNT_ASSIGNMENT_SHA256",
];

const DIALOG_RATE_ONE_REFUSAL: &str = "dialog stream configuration: stream step duration at 1 Hz must truncate to at least one sample";
const STREAM_CHUNK_REFUSAL: &str =
    "stream chunk: --chunk-ms must produce at least one model-rate sample";
const WINDOW_SAMPLE_COUNT_REFUSAL: &str =
    "--window sample count: value must be finite and non-negative";
const ALTERNATE_SAMPLE_COUNT_REFUSAL: &str =
    "--alt-window sample count: value must be finite and non-negative";
const INVALID_FRONTEND_REFUSAL: &str =
    "frontend: ASR_FRONTEND: frontend mode must be scalar|batched, got \"invalid\"";

/// The sole cleanup owner for a unique test root across setup and child-process histories.
struct TemporaryRoot {
    root: PathBuf,
}

impl TemporaryRoot {
    fn path(&self) -> &Path {
        &self.root
    }
}

impl Drop for TemporaryRoot {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.root) {
            panic!(
                "test-owned temporary CLI witness cleanup must succeed for {}: {error}",
                self.root.display()
            );
        }
    }
}

/// One test-owned root holds complete schema-valid packages and distinct absent child inputs.
struct CliWitnessRoot {
    _cleanup_root: TemporaryRoot,
    valid_package: PathBuf,
    dialog_rate_one_package: PathBuf,
    benchmark_high_rate_package: PathBuf,
    missing_model: PathBuf,
    absent_audio: PathBuf,
    absent_ort_dylib: PathBuf,
    empty_working_directory: PathBuf,
}

impl CliWitnessRoot {
    fn new() -> Self {
        let sequence = NEXT_TEMPORARY_ROOT.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "gigaam-cli-effect-order-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&root).expect("each test must create a unique temporary CLI witness root");
        let cleanup_root = TemporaryRoot { root };

        let valid_package = cleanup_root.path().join("valid-v1-package");
        fs::create_dir(&valid_package)
            .expect("the test-owned valid V1 package directory must be creatable");
        fs::write(
            valid_package.join("config.kv"),
            asset_free_v1_config(16_000),
        )
        .expect("the test-owned valid V1 package configuration must be writable");

        let dialog_rate_one_package = cleanup_root.path().join("dialog-rate-one-v1-package");
        fs::create_dir(&dialog_rate_one_package)
            .expect("the test-owned Dialog-invalid V1 package directory must be creatable");
        fs::write(
            dialog_rate_one_package.join("config.kv"),
            asset_free_v1_config(1),
        )
        .expect("the test-owned Dialog-invalid V1 package configuration must be writable");

        let benchmark_high_rate_package =
            cleanup_root.path().join("benchmark-high-rate-v1-package");
        fs::create_dir(&benchmark_high_rate_package)
            .expect("the test-owned high-rate benchmark V1 package directory must be creatable");
        fs::write(
            benchmark_high_rate_package.join("config.kv"),
            asset_free_v1_config(4_000_000_000),
        )
        .expect("the test-owned high-rate benchmark V1 package configuration must be writable");

        let missing_model = cleanup_root.path().join("absent-model-package");
        let absent_audio = cleanup_root.path().join("absent-input.wav");
        let absent_ort_dylib = cleanup_root.path().join("absent-ort-dylib");
        let empty_working_directory = cleanup_root.path().join("empty-working-directory");
        fs::create_dir(&empty_working_directory)
            .expect("the test-owned empty working directory must be creatable");
        assert!(
            !missing_model.exists() && !absent_audio.exists() && !absent_ort_dylib.exists(),
            "a fresh witness root must reserve distinct absent model, audio, and ORT paths"
        );

        Self {
            _cleanup_root: cleanup_root,
            valid_package,
            dialog_rate_one_package,
            benchmark_high_rate_package,
            missing_model,
            absent_audio,
            absent_ort_dylib,
            empty_working_directory,
        }
    }

    fn process(&self, workflow: &str) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_asr"));
        for key in PRODUCT_ENVIRONMENT_KEYS {
            command.env_remove(key);
        }
        command.env("ASR_ENCODER_EP", "cpu").arg(workflow);
        command
    }

    fn command(&self, workflow: &str, model: &Path) -> Command {
        let mut command = self.process(workflow);
        command
            .arg(&self.absent_audio)
            .arg("--model")
            .arg(model)
            .arg("--ort-dylib")
            .arg(&self.absent_ort_dylib);
        command
    }

    fn benchmark_command(&self, model: &Path) -> Command {
        let mut command = self.process("bench");
        command
            .arg("--audio")
            .arg(&self.absent_audio)
            .arg("--model")
            .arg(model)
            .arg("--ort-dylib")
            .arg(&self.absent_ort_dylib);
        command
    }

    fn default_model_command(&self) -> Command {
        let mut command = self.process("transcribe");
        command
            .current_dir(&self.empty_working_directory)
            .arg(&self.absent_audio)
            .arg("--ort-dylib")
            .arg(&self.absent_ort_dylib);
        command
    }

    fn valid_package(&self) -> &Path {
        &self.valid_package
    }

    fn dialog_rate_one_package(&self) -> &Path {
        &self.dialog_rate_one_package
    }

    fn benchmark_high_rate_package(&self) -> &Path {
        &self.benchmark_high_rate_package
    }

    fn missing_model(&self) -> &Path {
        &self.missing_model
    }

    fn ort_diagnostic(&self) -> String {
        format!("ORT dylib {}:", self.absent_ort_dylib.display())
    }
}

fn asset_free_v1_config(sample_rate: usize) -> String {
    format!("format_version=1\nsample_rate={sample_rate}\n{V1_CONFIG_BODY}")
}

const V1_CONFIG_BODY: &str = "n_mels=64
hop_length=160
n_fft=320
center=false
log_clamp_min=1e-09
log_clamp_max=1000000000.0
frames_per_sec=25.0
ctc.vocab=vocab_ctc.txt
ctc.blank_id=256
ctc.encoder_fp32=ctc_fp32_encoder.onnx
ctc.input_names=features,feature_lengths
ctc.output_names=log_probs,encoded_lengths
rnnt.vocab=vocab_rnnt.txt
rnnt.blank_id=1024
rnnt.pred_hidden=320
rnnt.max_symbols_per_step=10
rnnt.encoder_fp16io32=rnnt_fp16io32_encoder.onnx
rnnt.decoder_fp16=rnnt_fp16_decoder.onnx
rnnt.joint_fp16=rnnt_fp16_joint.onnx
rnnt.encoder_fp32=rnnt_fp32_encoder.onnx
rnnt.decoder_fp32=rnnt_fp32_decoder.onnx
rnnt.joint_fp32=rnnt_fp32_joint.onnx
ctc.encoder_fp16io32=ctc_fp16io32_encoder.onnx
ctc.out_dim=257
ctc.out_layout=t_d
rnnt.out_dim=768
rnnt.out_layout=d_t
rnnt.input_names=audio_signal,length
rnnt.output_names=encoded,encoded_len
rnnt.decoder_inputs=x,hi,ci
rnnt.decoder_outputs=dec,ho,co
rnnt.joint_inputs=enc,dec
rnnt.joint_outputs=joint
vad.model=silero_vad.onnx";

fn run(mut command: Command) -> Output {
    command.output().expect(
        "the Cargo-provided production asr binary must be executable by its integration test",
    )
}

fn failure_stderr(output: Output, history: &str) -> String {
    let stderr = String::from_utf8(output.stderr)
        .expect("the production CLI's observed stderr diagnostic must be valid UTF-8");
    assert_eq!(
        output.status.code(),
        Some(1),
        "{history} must exit with status 1; stderr: {stderr}"
    );
    assert!(
        output.stdout.is_empty(),
        "{history} must not write stdout; stderr: {stderr}"
    );
    stderr
}

fn assert_exact_failure(output: Output, expected: &str, history: &str) {
    assert_eq!(failure_stderr(output, history), format!("{expected}\n"));
}

fn assert_ort_sentinel(output: Output, witness: &CliWitnessRoot, history: &str) {
    let stderr = failure_stderr(output, history);
    let sentinel = witness.ort_diagnostic();
    assert!(
        stderr.contains(&sentinel),
        "{history} must reach its exact test-owned ORT path diagnostic {sentinel:?}; stderr: {stderr}"
    );
}

fn assert_no_ort_sentinel(stderr: &str, witness: &CliWitnessRoot, history: &str) {
    let sentinel = witness.ort_diagnostic();
    assert!(
        !stderr.contains(&sentinel),
        "{history} must refuse before its ORT path diagnostic {sentinel:?}; stderr: {stderr}"
    );
}

fn assert_missing_package_refusal(output: Output, witness: &CliWitnessRoot, history: &str) {
    let stderr = failure_stderr(output, history);
    let model_open = format!(
        "model: open model package {}:",
        witness.missing_model().display()
    );
    assert!(
        stderr.contains(&model_open),
        "{history} must preserve the model-open refusal class {model_open:?}; stderr: {stderr}"
    );
    assert_no_ort_sentinel(&stderr, witness, history);
}

fn assert_policy_refusal(
    output: Output,
    witness: &CliWitnessRoot,
    policy_class: &str,
    history: &str,
) {
    let stderr = failure_stderr(output, history);
    assert!(
        stderr.contains(policy_class),
        "{history} must preserve policy refusal class {policy_class:?}; stderr: {stderr}"
    );
    assert_no_ort_sentinel(&stderr, witness, history);
}

#[test]
fn transcribe_cli_effect_order() {
    let witness = CliWitnessRoot::new();

    assert_ort_sentinel(
        run(witness.command("transcribe", witness.valid_package())),
        &witness,
        "transcribe valid package",
    );
    assert_missing_package_refusal(
        run(witness.command("transcribe", witness.missing_model())),
        &witness,
        "transcribe missing package",
    );

    let maximum_window = f32::MAX.to_string();
    let mut invalid_policy = witness.command("transcribe", witness.valid_package());
    invalid_policy.arg("--window").arg(maximum_window);
    assert_policy_refusal(
        run(invalid_policy),
        &witness,
        "batch window duration at 16000 Hz sample count: inf cannot be represented as usize",
        "transcribe invalid window",
    );
}

#[test]
fn stream_cli_effect_order_and_checked_chunk_precedence() {
    let witness = CliWitnessRoot::new();

    assert_ort_sentinel(
        run(witness.command("stream", witness.valid_package())),
        &witness,
        "stream valid package",
    );
    assert_missing_package_refusal(
        run(witness.command("stream", witness.missing_model())),
        &witness,
        "stream missing package",
    );

    let maximum_window = f32::MAX.to_string();
    let mut invalid_policy = witness.command("stream", witness.valid_package());
    invalid_policy.arg("--window").arg(maximum_window);
    assert_policy_refusal(
        run(invalid_policy),
        &witness,
        "stream configuration: stream window duration at 16000 Hz sample count: inf cannot be represented as usize",
        "stream invalid window",
    );

    let mut invalid_chunk = witness.command("stream", witness.valid_package());
    invalid_chunk.arg("--chunk-ms").arg(usize::MAX.to_string());
    assert_exact_failure(
        run(invalid_chunk),
        STREAM_CHUNK_REFUSAL,
        "stream checked model-rate chunk overflow",
    );
}

#[test]
fn dialog_cli_effect_order() {
    let witness = CliWitnessRoot::new();

    assert_ort_sentinel(
        run(witness.command("dialog", witness.valid_package())),
        &witness,
        "dialog valid package",
    );
    assert_missing_package_refusal(
        run(witness.command("dialog", witness.missing_model())),
        &witness,
        "dialog missing package",
    );

    let stderr = failure_stderr(
        run(witness.command("dialog", witness.dialog_rate_one_package())),
        "dialog invalid rate-one package",
    );
    assert_eq!(
        stderr,
        format!("{DIALOG_RATE_ONE_REFUSAL}\n"),
        "dialog rate-one package must preserve its exact model-bound stream refusal"
    );
    assert_no_ort_sentinel(&stderr, &witness, "dialog invalid rate-one package");
}

#[test]
fn vad_cli_effect_order() {
    let witness = CliWitnessRoot::new();

    assert_ort_sentinel(
        run(witness.command("vad", witness.valid_package())),
        &witness,
        "vad valid package",
    );
    assert_missing_package_refusal(
        run(witness.command("vad", witness.missing_model())),
        &witness,
        "vad missing package",
    );

    let maximum_minimum_speech = usize::MAX.to_string();
    let mut invalid_policy = witness.command("vad", witness.valid_package());
    invalid_policy
        .arg("--min-speech-ms")
        .arg(maximum_minimum_speech);
    assert_policy_refusal(
        run(invalid_policy),
        &witness,
        "VAD durations: VAD minimum speech duration at 16000 Hz sample count:",
        "vad invalid minimum speech duration",
    );
}

#[test]
fn benchmark_opens_the_package_and_prepares_model_rate_input_before_native_work() {
    let witness = CliWitnessRoot::new();

    assert_missing_package_refusal(
        run(witness.benchmark_command(witness.missing_model())),
        &witness,
        "benchmark missing package",
    );

    let mut primary_overflow = witness.benchmark_command(witness.benchmark_high_rate_package());
    primary_overflow.arg("--window").arg("1e30");
    assert_policy_refusal(
        run(primary_overflow),
        &witness,
        WINDOW_SAMPLE_COUNT_REFUSAL,
        "benchmark primary sample-plan overflow",
    );

    let mut alternate_overflow = witness.benchmark_command(witness.benchmark_high_rate_package());
    alternate_overflow.arg("--alt-window").arg("1e30");
    assert_policy_refusal(
        run(alternate_overflow),
        &witness,
        ALTERNATE_SAMPLE_COUNT_REFUSAL,
        "benchmark alternate sample-plan overflow",
    );

    let output = run(witness.benchmark_command(witness.valid_package()));
    let stderr = failure_stderr(output, "benchmark missing WAV after valid package planning");
    let audio_diagnostic = format!("bench audio {}:", witness.absent_audio.display());
    assert!(
        stderr.contains(&audio_diagnostic),
        "benchmark must load audio before native initialization: {stderr}"
    );
    assert_no_ort_sentinel(
        &stderr,
        &witness,
        "benchmark missing WAV after valid package planning",
    );
}

#[test]
fn default_model_path_stays_current_directory_relative() {
    let witness = CliWitnessRoot::new();
    let stderr = failure_stderr(
        run(witness.default_model_command()),
        "default model path in an empty child working directory",
    );
    assert!(
        stderr.contains("model: open model package model:"),
        "the default model path must stay the literal child-relative model path; stderr: {stderr}"
    );
    assert_no_ort_sentinel(&stderr, &witness, "default model path");
}

#[test]
fn frontend_configuration_fails_before_package_native_and_audio_for_each_applicable_command() {
    let witness = CliWitnessRoot::new();
    for workflow in ["transcribe", "stream", "dialog"] {
        let mut command = witness.command(workflow, witness.missing_model());
        command.env("ASR_FRONTEND", "invalid");
        assert_exact_failure(
            run(command),
            INVALID_FRONTEND_REFUSAL,
            &format!("invalid frontend for {workflow}"),
        );
    }
    let mut benchmark = witness.benchmark_command(witness.missing_model());
    benchmark.env("ASR_FRONTEND", "invalid");
    assert_exact_failure(
        run(benchmark),
        INVALID_FRONTEND_REFUSAL,
        "invalid frontend for benchmark",
    );
}

#[test]
fn vad_ignores_inapplicable_frontend_configuration() {
    let witness = CliWitnessRoot::new();
    let mut command = witness.command("vad", witness.valid_package());
    command.env("ASR_FRONTEND", "invalid");
    assert_ort_sentinel(run(command), &witness, "VAD inapplicable invalid frontend");
}
