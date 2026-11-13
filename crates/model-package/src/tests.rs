// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

use super::{EncoderPrecision, ModelPackage, PackageError};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP_PACK: AtomicU64 = AtomicU64::new(0);

struct TempPack {
    root: PathBuf,
}

impl TempPack {
    fn new(config: &str) -> Self {
        let sequence = NEXT_TEMP_PACK.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "gigaam-model-package-test-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&root).expect("each test must create a unique temporary model package");
        fs::write(root.join("config.kv"), config)
            .expect("temporary model package configuration must be writable");
        Self { root }
    }

    fn write_asset(&self, relative: &str, bytes: &[u8]) {
        fs::write(self.root.join(relative), bytes)
            .expect("temporary selected asset must be writable");
    }

    fn make_directory(&self, relative: &str) {
        fs::create_dir(self.root.join(relative))
            .expect("temporary selected asset directory must be creatable");
    }

    fn path(&self) -> &Path {
        &self.root
    }
}

impl Drop for TempPack {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.root) {
            panic!(
                "test-owned temporary model package cleanup must succeed for {}: {error}",
                self.root.display()
            );
        }
    }
}

fn valid_v1() -> String {
    [
        "format_version=1",
        "sample_rate=16000",
        "n_mels=64",
        "hop_length=160",
        "n_fft=320",
        "center=false",
        "log_clamp_min=1e-09",
        "log_clamp_max=1000000000.0",
        "frames_per_sec=25.0",
        "ctc.vocab=vocab_ctc.txt",
        "ctc.blank_id=256",
        "ctc.encoder_fp32=ctc_fp32_encoder.onnx",
        "ctc.input_names=features,feature_lengths",
        "ctc.output_names=log_probs,encoded_lengths",
        "rnnt.vocab=vocab_rnnt.txt",
        "rnnt.blank_id=1024",
        "rnnt.pred_hidden=320",
        "rnnt.max_symbols_per_step=10",
        "rnnt.encoder_fp16io32=rnnt_fp16io32_encoder.onnx",
        "rnnt.decoder_fp16=rnnt_fp16_decoder.onnx",
        "rnnt.joint_fp16=rnnt_fp16_joint.onnx",
        "rnnt.encoder_fp32=rnnt_fp32_encoder.onnx",
        "rnnt.decoder_fp32=rnnt_fp32_decoder.onnx",
        "rnnt.joint_fp32=rnnt_fp32_joint.onnx",
        "ctc.encoder_fp16io32=ctc_fp16io32_encoder.onnx",
        "ctc.out_dim=257",
        "ctc.out_layout=t_d",
        "rnnt.out_dim=768",
        "rnnt.out_layout=d_t",
        "rnnt.input_names=audio_signal,length",
        "rnnt.output_names=encoded,encoded_len",
        "rnnt.decoder_inputs=x,hi,ci",
        "rnnt.decoder_outputs=dec,ho,co",
        "rnnt.joint_inputs=enc,dec",
        "rnnt.joint_outputs=joint",
        "vad.model=silero_vad.onnx",
    ]
    .join("\n")
}

fn assert_invalid_value(error: PackageError, key: &str) {
    match error {
        PackageError::InvalidValue { key: found, .. } => assert_eq!(found, key),
        other => panic!("expected invalid value for {key}, got {other}"),
    }
}

#[test]
fn valid_v1_has_typed_immutable_projections_without_retained_metadata() {
    let pack = TempPack::new(&valid_v1());
    let opened = ModelPackage::open(pack.path()).expect("valid V1 config must open");

    assert_eq!(opened.schema_version().value(), 1);
    assert_eq!(opened.frontend().sample_rate(), 16_000);
    assert_eq!(opened.frontend().n_mels(), 64);
    assert_eq!(opened.ctc().blank_id(), 256);
    assert_eq!(opened.rnnt().prediction_hidden(), 320);
    assert_eq!(opened.rnnt().max_symbols_per_step(), 10);
}

#[test]
fn retained_fields_are_optional_opaque_and_never_selected() {
    let config = format!(
        "{}\nwin_length=not-a-number\nsubsampling_factor=0\npos_emb_max_len=unused\nctc.encoder_fp16=../legacy.onnx\nrnnt.pred_layers=not-a-number\nrnnt.encoder_fp16=/legacy.onnx\nsource=opaque metadata\nexported=not-a-date",
        valid_v1()
    );
    let pack = TempPack::new(&config);
    ModelPackage::open(pack.path())
        .expect("opaque retained metadata must not acquire V1 runtime semantics");
}

#[test]
fn version_discriminator_precedes_v1_inventory_validation() {
    let pack = TempPack::new("format_version=3\nfuture.only=value");
    match ModelPackage::open(pack.path()) {
        Err(PackageError::UnsupportedFormatVersion { value }) => assert_eq!(value, "3"),
        Err(other) => panic!("expected unsupported discriminator refusal, got {other}"),
        Ok(_) => panic!("unsupported configuration must not be accepted"),
    }
}

#[test]
fn format_version_refusal_classes_are_distinct() {
    let absent = TempPack::new("sample_rate=16000");
    assert!(matches!(
        ModelPackage::open(absent.path()),
        Err(PackageError::MissingKey {
            key: "format_version"
        })
    ));

    for value in ["x", "01"] {
        let pack = TempPack::new(&format!("format_version={value}"));
        let error = ModelPackage::open(pack.path())
            .expect_err("non-canonical format version must refuse as an invalid value");
        assert_invalid_value(error, "format_version");
    }

    for value in ["3", "256"] {
        let unsupported = TempPack::new(&format!("format_version={value}\nfuture.only=value"));
        assert!(matches!(
            ModelPackage::open(unsupported.path()),
            Err(PackageError::UnsupportedFormatVersion { value: found }) if found == value
        ));
    }
}

#[test]
fn unknown_v1_key_carries_its_real_source_line() {
    let config = format!("{}\nunknown.future=value", valid_v1());
    let pack = TempPack::new(&config);
    match ModelPackage::open(pack.path()) {
        Err(PackageError::UnknownKey { line, key }) => {
            assert_eq!(key, "unknown.future");
            assert_eq!(line, 37);
        }
        Err(other) => panic!("expected unknown-key refusal, got {other}"),
        Ok(_) => panic!("unknown V1 key must refuse"),
    }
}

#[test]
fn malformed_duplicate_missing_and_boolean_values_refuse() {
    let malformed = TempPack::new("format_version=1\nsample_rate");
    assert!(matches!(
        ModelPackage::open(malformed.path()),
        Err(PackageError::MalformedLine { line: 2, .. })
    ));

    let duplicate = TempPack::new(&format!("{}\nsample_rate=16000", valid_v1()));
    assert!(matches!(
        ModelPackage::open(duplicate.path()),
        Err(PackageError::DuplicateKey { key, .. }) if key == "sample_rate"
    ));

    let missing = TempPack::new(&valid_v1().replace("\nvad.model=silero_vad.onnx", ""));
    assert!(matches!(
        ModelPackage::open(missing.path()),
        Err(PackageError::MissingKey { key: "vad.model" })
    ));

    let invalid_bool = TempPack::new(&valid_v1().replace("center=false", "center=False"));
    let error =
        ModelPackage::open(invalid_bool.path()).expect_err("non-canonical boolean must refuse");
    assert_invalid_value(error, "center");
}

#[test]
fn invalid_active_asset_path_and_zero_max_symbols_refuse() {
    for path in ["../vocab", "/absolute/vocab", "./vocab"] {
        let unsafe_path = TempPack::new(
            &valid_v1().replace("ctc.vocab=vocab_ctc.txt", &format!("ctc.vocab={path}")),
        );
        assert!(matches!(
            ModelPackage::open(unsafe_path.path()),
            Err(PackageError::UnsafeAssetPath {
                key: "ctc.vocab",
                ..
            })
        ));
    }

    let zero_max = TempPack::new(&valid_v1().replace(
        "rnnt.max_symbols_per_step=10",
        "rnnt.max_symbols_per_step=0",
    ));
    assert!(matches!(
        ModelPackage::open(zero_max.path()),
        Err(PackageError::Compatibility {
            field: "rnnt.max_symbols_per_step",
            ..
        })
    ));
}

#[test]
fn invalid_typed_scalar_refuses_before_any_asset_selection() {
    let pack = TempPack::new(&valid_v1().replace("sample_rate=16000", "sample_rate=16k"));
    let error = ModelPackage::open(pack.path())
        .expect_err("non-decimal sample rate must refuse before asset selection");
    assert_invalid_value(error, "sample_rate");
}

#[test]
fn encoder_tensor_contract_requires_exactly_two_inputs_and_outputs_before_asset_selection() {
    for (expected_key, expected, replacement) in [
        (
            "ctc.input_names",
            "ctc.input_names=features,feature_lengths",
            "ctc.input_names=features",
        ),
        (
            "ctc.output_names",
            "ctc.output_names=log_probs,encoded_lengths",
            "ctc.output_names=log_probs,encoded_lengths,extra",
        ),
        (
            "rnnt.input_names",
            "rnnt.input_names=audio_signal,length",
            "rnnt.input_names=audio_signal",
        ),
        (
            "rnnt.output_names",
            "rnnt.output_names=encoded,encoded_len",
            "rnnt.output_names=encoded,encoded_len,extra",
        ),
    ] {
        let pack = TempPack::new(&valid_v1().replace(expected, replacement));
        let error = ModelPackage::open(pack.path()).expect_err(
            "a non-exact encoder tensor arity must refuse before any asset is selected",
        );
        assert_invalid_value(error, expected_key);
    }
}

#[test]
fn selected_assets_are_checked_without_selecting_other_precisions_or_rnnt() {
    let pack = TempPack::new(&valid_v1());
    let opened = ModelPackage::open(pack.path()).expect("V1 definition must open before assets");

    pack.make_directory("ctc_fp32_encoder.onnx");
    assert!(matches!(
        opened.ctc_encoder(EncoderPrecision::Fp32),
        Err(PackageError::AssetNotRegularFile {
            key: "ctc.encoder_fp32",
            ..
        })
    ));

    fs::remove_dir(pack.path().join("ctc_fp32_encoder.onnx"))
        .expect("temporary non-regular selected asset must be removable");
    pack.write_asset("ctc_fp32_encoder.onnx", b"graph");
    opened
        .ctc_encoder(EncoderPrecision::Fp32)
        .expect("selected CTC fp32 graph must validate as a regular file");
    let selected = opened
        .ctc_encoder(EncoderPrecision::Fp32)
        .expect("selected CTC fp32 graph must expose its typed tensor contract");
    let tensor_contract = selected.tensor_contract();
    assert_eq!(tensor_contract.data_input(), "features");
    assert_eq!(tensor_contract.length_input(), "feature_lengths");
    assert_eq!(tensor_contract.data_output(), "log_probs");
    assert_eq!(tensor_contract.length_output(), "encoded_lengths");
    assert_eq!(selected.precision(), EncoderPrecision::Fp32);
    assert_eq!(
        selected.artifact().path(),
        fs::canonicalize(pack.path().join("ctc_fp32_encoder.onnx"))
            .expect("selected asset must have a canonical target")
    );
    assert_eq!(selected.output_dimension(), 257);
    assert!(opened.ctc_encoder(EncoderPrecision::Fp16Io32).is_err());
    assert!(opened.rnnt_assets(EncoderPrecision::Fp32).is_err());
}

#[cfg(unix)]
#[test]
fn selected_asset_rejects_an_intermediate_symlink_escape() {
    use std::os::unix::fs::symlink;

    let outside = TempPack::new("format_version=1");
    outside.write_asset("escaped.onnx", b"outside package");

    let pack = TempPack::new(&valid_v1().replace(
        "ctc.encoder_fp32=ctc_fp32_encoder.onnx",
        "ctc.encoder_fp32=graphs/escaped.onnx",
    ));
    symlink(outside.path(), pack.path().join("graphs"))
        .expect("test setup must create an intermediate directory symlink");

    let opened = ModelPackage::open(pack.path()).expect("V1 definition must open before assets");
    assert!(matches!(
        opened.ctc_encoder(EncoderPrecision::Fp32),
        Err(PackageError::UnsafeAssetPath {
            key: "ctc.encoder_fp32",
            reason: "canonical target escapes the model package root",
            ..
        })
    ));
}

#[test]
fn malformed_selected_f32_asset_refuses_after_regular_file_validation() {
    let pack = TempPack::new(&valid_v1());
    pack.write_asset("stft_window.f32", b"\x01\x00");
    pack.write_asset("mel_fbank.f32", b"\x01\x00");
    let opened = ModelPackage::open(pack.path()).expect("valid V1 config must open");
    assert!(matches!(
        opened.frontend_weights(),
        Err(PackageError::InvalidAssetData {
            key: "frontend.window",
            ..
        })
    ));
}

#[test]
fn v2_selects_native_fp16_and_format() {
    let config = format!(
        "{}\nartifact_format=trt_ep_context\nctc.encoder_fp16=ctc-half.onnx\nrnnt.encoder_fp16=rnnt-half.onnx",
        valid_v1().replace("format_version=1", "format_version=2")
    );
    let pack = TempPack::new(&config);
    std::fs::write(pack.path().join("rnnt-half.onnx"), b"graph").expect("write selected graph");
    assert!(
        ModelPackage::open(pack.path())
            .expect("V2 must open")
            .ctc_encoder(EncoderPrecision::Fp16)
            .is_err()
    );
    pack.write_asset("ctc-half.onnx", b"native");
    pack.write_asset("ctc_fp32_encoder.onnx", b"fp32");
    pack.write_asset("ctc_fp16io32_encoder.onnx", b"io32");
    let opened = ModelPackage::open(pack.path()).expect("V2 must open");
    assert_eq!(
        opened.artifact_format(),
        crate::ArtifactFormat::TrtEpContext
    );
    for (precision, path) in [
        (EncoderPrecision::Fp32, "ctc_fp32_encoder.onnx"),
        (EncoderPrecision::Fp16Io32, "ctc_fp16io32_encoder.onnx"),
        (EncoderPrecision::Fp16, "ctc-half.onnx"),
    ] {
        let selected = opened
            .ctc_encoder(precision)
            .expect("selected CTC asset exists");
        assert_eq!(selected.precision(), precision);
        assert!(selected.artifact().path().ends_with(path));
    }
    let selected = opened
        .rnnt_encoder(crate::EncoderPrecision::Fp16)
        .expect("select native FP16");
    assert!(selected.artifact().path().ends_with("rnnt-half.onnx"));
    let v1 = TempPack::new(&valid_v1());
    let opened_v1 = ModelPackage::open(v1.path()).expect("V1 must open");
    assert_eq!(opened_v1.artifact_format(), crate::ArtifactFormat::Onnx);
    assert!(opened_v1.ctc_encoder(EncoderPrecision::Fp16).is_err());
    assert!(
        opened_v1
            .rnnt_encoder(crate::EncoderPrecision::Fp16)
            .is_err()
    );
}

fn valid_v3(variant: &str, sentencepiece: bool, rnnt: bool) -> String {
    let mut lines = vec![
        "format_version=3".to_string(),
        format!("variant={variant}"),
        "artifact_format=onnx".into(),
        "sample_rate=16000".into(),
        "n_mels=64".into(),
        "hop_length=160".into(),
        "n_fft=320".into(),
        "win_length=320".into(),
        "center=false".into(),
        "log_clamp_min=0.000000001".into(),
        "log_clamp_max=1000000000".into(),
        "frames_per_sec=25".into(),
        "subsampling_factor=4".into(),
        "frontend.window=window.f32".into(),
        "frontend.filterbank=filter.f32".into(),
        format!(
            "tokenizer.kind={}",
            if sentencepiece {
                "sentencepiece"
            } else {
                "characters"
            }
        ),
        "tokenizer.vocab=vocab.txt".into(),
        "tokenizer.vocab_size=2".into(),
        "tokenizer.blank_id=2".into(),
    ];
    if sentencepiece {
        lines.push("tokenizer.model=tokenizer.model".into());
    }
    let family = if rnnt { "rnnt" } else { "ctc" };
    lines.extend([
        format!("{family}.encoder_fp32=encoder.onnx"),
        format!("{family}.input_names=x,length"),
        format!("{family}.output_names=y,length"),
        format!("{family}.out_dim={}", if rnnt { 7 } else { 3 }),
        format!("{family}.out_layout=t_d"),
    ]);
    if rnnt {
        lines.extend(
            [
                "rnnt.decoder_fp32=decoder.onnx",
                "rnnt.joint_fp32=joint.onnx",
                "rnnt.pred_hidden=5",
                "rnnt.pred_layers=2",
                "rnnt.max_symbols_per_step=3",
                "rnnt.decoder_inputs=x,h,c",
                "rnnt.decoder_outputs=y,h,c",
                "rnnt.joint_inputs=e,d",
                "rnnt.joint_outputs=y",
            ]
            .into_iter()
            .map(String::from),
        );
    }
    lines.join("\n")
}

#[test]
fn v3_variants_admit_one_family_and_selected_assets() {
    use super::{ModelFamily, TokenizerKind, VariantId, VariantPackage};
    for (name, sp, rnnt, id) in [
        ("v3_ctc", false, false, VariantId::V3Ctc),
        ("v3_rnnt", false, true, VariantId::V3Rnnt),
        ("v3_e2e_ctc", true, false, VariantId::V3E2eCtc),
        ("v3_e2e_rnnt", true, true, VariantId::V3E2eRnnt),
    ] {
        let pack = TempPack::new(&valid_v3(name, sp, rnnt));
        let package =
            VariantPackage::open(pack.path()).expect("valid V3 metadata opens without assets");
        assert_eq!(package.variant(), id);
        assert_eq!(
            id.family(),
            if rnnt {
                ModelFamily::Rnnt
            } else {
                ModelFamily::Ctc
            }
        );
        assert_eq!(
            package.tokenizer().kind(),
            if sp {
                TokenizerKind::SentencePiece
            } else {
                TokenizerKind::Characters
            }
        );
        assert!(package.encoder(EncoderPrecision::Fp32).is_err());
        pack.write_asset("encoder.onnx", b"graph");
        assert!(package.encoder(EncoderPrecision::Fp32).is_ok());
        assert!(package.encoder(EncoderPrecision::Fp16).is_err());
        pack.write_asset("vocab.txt", b" \na\n");
        assert_eq!(
            package.vocabulary().expect("selected vocabulary is valid"),
            [" ", "a"]
        );
        pack.write_asset("vocab.txt", b" \na\nextra\n");
        assert!(package.vocabulary().is_err());
        if sp {
            assert!(package.tokenizer_model().is_err());
        } else {
            assert!(
                package
                    .tokenizer_model()
                    .expect("character tokenizer has no model")
                    .is_none()
            );
        }
        assert!(ModelPackage::open(pack.path()).is_err());
    }
    let legacy = TempPack::new(&valid_v1());
    assert!(super::VariantPackage::open(legacy.path()).is_err());
}

#[test]
fn v3_refuses_inconsistent_or_open_metadata() {
    use super::VariantPackage;
    let base = valid_v3("v3_ctc", false, false);
    for invalid in [
        base.replace("tokenizer.blank_id=2", "tokenizer.blank_id=1"),
        base.replace("ctc.out_dim=3", "ctc.out_dim=4"),
        base.replace("tokenizer.kind=characters", "tokenizer.kind=sentencepiece"),
        format!("{base}\nunknown=value"),
        format!("{base}\nvariant=v3_ctc"),
        format!("{base}\nrnnt.encoder_fp32=x"),
        format!("{base}\ntokenizer.model=x"),
        base.replace("hop_length=160", "hop_length=0"),
        base.replace(
            "tokenizer.vocab_size=2",
            "tokenizer.vocab_size=999999999999999999999999999999999999999",
        ),
        base.replace("ctc.out_layout=t_d", "ctc.out_layout=invalid"),
        base.replace("ctc.input_names=x,length", "ctc.input_names=x,x"),
        base.replace("ctc.encoder_fp32=encoder.onnx", ""),
        format!("{base}\nmalformed"),
        base.replace("frames_per_sec=25", "frames_per_sec=26"),
        base.replace(
            "ctc.encoder_fp32=encoder.onnx",
            "ctc.encoder_fp32=../escape",
        ),
    ] {
        let pack = TempPack::new(&invalid);
        assert!(VariantPackage::open(pack.path()).is_err());
    }
    let rate = base
        .replace("subsampling_factor=4", "subsampling_factor=3")
        .replace("frames_per_sec=25", "frames_per_sec=33.33333206176758");
    assert!(VariantPackage::open(TempPack::new(&rate).path()).is_ok());
    let wrong = rate.replace("33.33333206176758", "33.33334");
    assert!(VariantPackage::open(TempPack::new(&wrong).path()).is_err());
}

#[test]
fn v3_rnnt_precision_requires_complete_declared_heads() {
    use super::VariantPackage;
    let base = valid_v3("v3_rnnt", false, true);
    let partial = format!("{base}\nrnnt.decoder_fp16=decoder16.onnx");
    assert!(VariantPackage::open(TempPack::new(&partial).path()).is_err());
    let complete =
        format!("{partial}\nrnnt.joint_fp16=joint16.onnx\nrnnt.encoder_fp16=encoder16.onnx");
    let pack = TempPack::new(&complete);
    let package =
        VariantPackage::open(pack.path()).expect("complete declarations need no assets at open");
    pack.write_asset("vocab.txt", b" \na\n");
    pack.write_asset("decoder16.onnx", b"decoder");
    pack.write_asset("joint16.onnx", b"joint");
    assert!(package.rnnt_assets(EncoderPrecision::Fp16).is_ok());
    assert!(package.encoder(EncoderPrecision::Fp16).is_err());
    pack.write_asset("vocab.txt", b"wrong\n");
    assert!(package.rnnt_assets(EncoderPrecision::Fp16).is_err());
}

#[cfg(unix)]
#[test]
fn v3_selected_asset_cannot_escape_through_symlink() {
    use std::os::unix::fs::symlink;
    let pack = TempPack::new(&valid_v3("v3_ctc", false, false));
    symlink(
        pack.path()
            .parent()
            .expect("temporary package has a parent"),
        pack.path().join("encoder.onnx"),
    )
    .expect("test symlink must be creatable");
    let package =
        super::VariantPackage::open(pack.path()).expect("metadata does not select an asset");
    assert!(package.encoder(EncoderPrecision::Fp32).is_err());
}
