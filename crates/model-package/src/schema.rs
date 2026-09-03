// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

use crate::assets::RelativeAsset;
use crate::definition::{
    CtcDefinition, EncoderTensorContract, FrontendDefinition, OutputLayout, PackageDefinition,
    RetainedMetadata, RnntDefinition, SchemaVersion, VadDefinition,
};
use crate::error::PackageError;
use std::collections::{BTreeMap, BTreeSet};

const V1_REQUIRED_KEYS: &[&str] = &[
    "format_version",
    "sample_rate",
    "n_mels",
    "hop_length",
    "n_fft",
    "center",
    "log_clamp_min",
    "log_clamp_max",
    "frames_per_sec",
    "ctc.vocab",
    "ctc.blank_id",
    "ctc.encoder_fp32",
    "ctc.input_names",
    "ctc.output_names",
    "rnnt.vocab",
    "rnnt.blank_id",
    "rnnt.pred_hidden",
    "rnnt.max_symbols_per_step",
    "rnnt.encoder_fp16io32",
    "rnnt.decoder_fp16",
    "rnnt.joint_fp16",
    "rnnt.encoder_fp32",
    "rnnt.decoder_fp32",
    "rnnt.joint_fp32",
    "ctc.encoder_fp16io32",
    "ctc.out_dim",
    "ctc.out_layout",
    "rnnt.out_dim",
    "rnnt.out_layout",
    "rnnt.input_names",
    "rnnt.output_names",
    "rnnt.decoder_inputs",
    "rnnt.decoder_outputs",
    "rnnt.joint_inputs",
    "rnnt.joint_outputs",
    "vad.model",
];

const V1_RETAINED_KEYS: &[&str] = &[
    "win_length",
    "subsampling_factor",
    "pos_emb_max_len",
    "ctc.encoder_fp16",
    "rnnt.pred_layers",
    "rnnt.encoder_fp16",
    "source",
    "exported",
];

pub(crate) fn parse(text: &str) -> Result<PackageDefinition, PackageError> {
    let mut values = BTreeMap::new();
    let mut line_numbers = BTreeMap::new();
    for (index, raw_line) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (raw_key, raw_value) = line.split_once('=').ok_or(PackageError::MalformedLine {
            line: line_number,
            reason: "expected key=value",
        })?;
        let key = raw_key.trim();
        let value = raw_value.trim();
        if key.is_empty() {
            return Err(PackageError::MalformedLine {
                line: line_number,
                reason: "key is empty",
            });
        }
        if value.is_empty() {
            return Err(PackageError::MalformedLine {
                line: line_number,
                reason: "value is empty",
            });
        }
        if values.contains_key(key) {
            return Err(PackageError::DuplicateKey {
                line: line_number,
                key: key.to_string(),
            });
        }
        values.insert(key.to_string(), value.to_string());
        line_numbers.insert(key.to_string(), line_number);
    }
    let version = values
        .get("format_version")
        .cloned()
        .ok_or(PackageError::MissingKey {
            key: "format_version",
        })?;
    if !version.bytes().all(|byte| byte.is_ascii_digit())
        || (version.len() > 1 && version.starts_with('0'))
    {
        return Err(PackageError::InvalidValue {
            key: "format_version",
            value: version,
            reason: "expected a canonical unsigned integer",
        });
    }
    if version != SchemaVersion::V1.value().to_string() {
        return Err(PackageError::UnsupportedFormatVersion { value: version });
    }
    for key in values.keys() {
        if !V1_REQUIRED_KEYS.contains(&key.as_str()) && !V1_RETAINED_KEYS.contains(&key.as_str()) {
            let line = line_numbers
                .get(key)
                .copied()
                .ok_or(PackageError::Compatibility {
                    field: "config.kv",
                    reason: "parsed key has no source line",
                })?;
            return Err(PackageError::UnknownKey {
                line,
                key: key.clone(),
            });
        }
    }
    for &key in V1_REQUIRED_KEYS {
        if !values.contains_key(key) {
            return Err(PackageError::MissingKey { key });
        }
    }

    let _ = take(&mut values, "format_version")?;

    let frontend = FrontendDefinition {
        sample_rate: parse_usize(&mut values, "sample_rate")?,
        n_mels: parse_usize(&mut values, "n_mels")?,
        hop_length: parse_usize(&mut values, "hop_length")?,
        n_fft: parse_usize(&mut values, "n_fft")?,
        center: parse_bool(&mut values, "center")?,
        log_clamp_min: parse_f64(&mut values, "log_clamp_min")?,
        log_clamp_max: parse_f64(&mut values, "log_clamp_max")?,
        frames_per_second: parse_f64(&mut values, "frames_per_sec")?,
        window: RelativeAsset::fixed("frontend.window", "stft_window.f32"),
        filterbank: RelativeAsset::fixed("frontend.filterbank", "mel_fbank.f32"),
    };
    let ctc = CtcDefinition {
        vocabulary: parse_asset(&mut values, "ctc.vocab")?,
        blank_id: parse_usize(&mut values, "ctc.blank_id")?,
        encoder_fp16io32: parse_asset(&mut values, "ctc.encoder_fp16io32")?,
        encoder_fp32: parse_asset(&mut values, "ctc.encoder_fp32")?,
        tensor_contract: parse_encoder_tensor_contract(
            &mut values,
            "ctc.input_names",
            "ctc.output_names",
        )?,
        output_dimension: parse_usize(&mut values, "ctc.out_dim")?,
        output_layout: parse_layout(&mut values, "ctc.out_layout")?,
    };
    let rnnt = RnntDefinition {
        vocabulary: parse_asset(&mut values, "rnnt.vocab")?,
        blank_id: parse_usize(&mut values, "rnnt.blank_id")?,
        prediction_hidden: parse_usize(&mut values, "rnnt.pred_hidden")?,
        max_symbols_per_step: parse_usize(&mut values, "rnnt.max_symbols_per_step")?,
        encoder_fp16io32: parse_asset(&mut values, "rnnt.encoder_fp16io32")?,
        decoder_fp16: parse_asset(&mut values, "rnnt.decoder_fp16")?,
        joint_fp16: parse_asset(&mut values, "rnnt.joint_fp16")?,
        encoder_fp32: parse_asset(&mut values, "rnnt.encoder_fp32")?,
        decoder_fp32: parse_asset(&mut values, "rnnt.decoder_fp32")?,
        joint_fp32: parse_asset(&mut values, "rnnt.joint_fp32")?,
        output_dimension: parse_usize(&mut values, "rnnt.out_dim")?,
        output_layout: parse_layout(&mut values, "rnnt.out_layout")?,
        encoder_tensor_contract: parse_encoder_tensor_contract(
            &mut values,
            "rnnt.input_names",
            "rnnt.output_names",
        )?,
        decoder_input_names: parse_names_array(&mut values, "rnnt.decoder_inputs")?,
        decoder_output_names: parse_names_array(&mut values, "rnnt.decoder_outputs")?,
        joint_input_names: parse_names_array(&mut values, "rnnt.joint_inputs")?,
        joint_output_name: parse_single_name(&mut values, "rnnt.joint_outputs")?,
    };
    let vad = VadDefinition {
        model: parse_asset(&mut values, "vad.model")?,
    };
    let retained = RetainedMetadata {
        win_length: take_optional(&mut values, "win_length"),
        subsampling_factor: take_optional(&mut values, "subsampling_factor"),
        pos_emb_max_len: take_optional(&mut values, "pos_emb_max_len"),
        ctc_encoder_fp16: take_optional(&mut values, "ctc.encoder_fp16"),
        rnnt_pred_layers: take_optional(&mut values, "rnnt.pred_layers"),
        rnnt_encoder_fp16: take_optional(&mut values, "rnnt.encoder_fp16"),
        source: take_optional(&mut values, "source"),
        exported: take_optional(&mut values, "exported"),
    };
    if !values.is_empty() {
        return Err(PackageError::Compatibility {
            field: "config.kv",
            reason: "schema did not consume a declared V1 key",
        });
    }
    Ok(PackageDefinition::new(
        SchemaVersion::V1,
        frontend,
        ctc,
        rnnt,
        vad,
        retained,
    ))
}

fn take(values: &mut BTreeMap<String, String>, key: &'static str) -> Result<String, PackageError> {
    values.remove(key).ok_or(PackageError::MissingKey { key })
}

fn parse_usize(
    values: &mut BTreeMap<String, String>,
    key: &'static str,
) -> Result<usize, PackageError> {
    let value = take(values, key)?;
    if !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(PackageError::InvalidValue {
            key,
            value,
            reason: "expected an unsigned decimal integer",
        });
    }
    value
        .parse::<usize>()
        .map_err(|_| PackageError::InvalidValue {
            key,
            value,
            reason: "does not fit usize",
        })
}

fn parse_f64(
    values: &mut BTreeMap<String, String>,
    key: &'static str,
) -> Result<f64, PackageError> {
    let value = take(values, key)?;
    let parsed = value
        .parse::<f64>()
        .map_err(|_| PackageError::InvalidValue {
            key,
            value: value.clone(),
            reason: "expected a finite decimal number",
        })?;
    if !parsed.is_finite() {
        return Err(PackageError::InvalidValue {
            key,
            value,
            reason: "expected a finite decimal number",
        });
    }
    Ok(parsed)
}

fn parse_bool(
    values: &mut BTreeMap<String, String>,
    key: &'static str,
) -> Result<bool, PackageError> {
    let value = take(values, key)?;
    match value.as_str() {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(PackageError::InvalidValue {
            key,
            value,
            reason: "expected exactly true or false",
        }),
    }
}

fn parse_asset(
    values: &mut BTreeMap<String, String>,
    key: &'static str,
) -> Result<RelativeAsset, PackageError> {
    RelativeAsset::parse(key, take(values, key)?)
}

fn take_optional(values: &mut BTreeMap<String, String>, key: &'static str) -> Option<String> {
    values.remove(key)
}

fn parse_names(
    values: &mut BTreeMap<String, String>,
    key: &'static str,
) -> Result<Vec<String>, PackageError> {
    let value = take(values, key)?;
    parse_name_value(key, value)
}

fn parse_name_value(key: &'static str, value: String) -> Result<Vec<String>, PackageError> {
    let mut names = Vec::new();
    let mut unique = BTreeSet::new();
    for raw_name in value.split(',') {
        let name = raw_name.trim();
        if name.is_empty() {
            return Err(PackageError::InvalidValue {
                key,
                value,
                reason: "contains an empty tensor name",
            });
        }
        if !unique.insert(name) {
            return Err(PackageError::InvalidValue {
                key,
                value,
                reason: "contains a duplicate tensor name",
            });
        }
        names.push(name.to_string());
    }
    Ok(names)
}

fn parse_names_array<const N: usize>(
    values: &mut BTreeMap<String, String>,
    key: &'static str,
) -> Result<[String; N], PackageError> {
    let value = take(values, key)?;
    let names = parse_name_value(key, value.clone())?;
    names.try_into().map_err(|_| PackageError::InvalidValue {
        key,
        value,
        reason: "does not contain the required number of tensor names",
    })
}

fn parse_encoder_tensor_contract(
    values: &mut BTreeMap<String, String>,
    input_key: &'static str,
    output_key: &'static str,
) -> Result<EncoderTensorContract, PackageError> {
    let inputs = parse_names_array(values, input_key)?;
    let outputs = parse_names_array(values, output_key)?;
    Ok(EncoderTensorContract::new(inputs, outputs))
}

fn parse_single_name(
    values: &mut BTreeMap<String, String>,
    key: &'static str,
) -> Result<String, PackageError> {
    let names = parse_names(values, key)?;
    match <Vec<String> as TryInto<[String; 1]>>::try_into(names) {
        Ok([name]) => Ok(name),
        Err(_) => Err(PackageError::InvalidValue {
            key,
            value: "comma-separated names".to_string(),
            reason: "requires exactly one tensor name",
        }),
    }
}

fn parse_layout(
    values: &mut BTreeMap<String, String>,
    key: &'static str,
) -> Result<OutputLayout, PackageError> {
    let value = take(values, key)?;
    match value.as_str() {
        "t_d" => Ok(OutputLayout::TimeThenDimension),
        "d_t" => Ok(OutputLayout::DimensionThenTime),
        _ => Err(PackageError::InvalidValue {
            key,
            value,
            reason: "expected t_d or d_t",
        }),
    }
}
