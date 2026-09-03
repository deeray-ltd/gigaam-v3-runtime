// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Versioned, typed model-package definitions and selected asset access.
//!
//! `config.kv` is the authoritative runtime input. The package owns its schema,
//! asset-path safety checks, compatibility checks, and typed immutable projections.
//! Callers cannot select arbitrary configuration keys or resolve arbitrary package
//! paths.

mod assets;
mod compatibility;
mod definition;
mod error;
mod schema;

use std::fs;
use std::path::{Path, PathBuf};

pub use assets::ValidatedArtifact;
pub use definition::{
    CtcDefinition, EncoderArtifact, EncoderPrecision, EncoderTensorContract, FrontendDefinition,
    FrontendWeights, OutputLayout, RnntAssets, RnntDefinition, SchemaVersion, VadDefinition,
};
pub use error::PackageError;

use assets::validate_regular_asset;
use definition::PackageDefinition;

/// Immutable definition of one opened model package.
#[derive(Debug, Clone)]
pub struct ModelPackage {
    root: PathBuf,
    definition: PackageDefinition,
}

impl ModelPackage {
    /// Parses and validates the complete V1 configuration without selecting model assets.
    /// Selected asset methods validate a regular file immediately before their caller creates
    /// a session, so disabled RNN-T and unused precision artifacts remain optional.
    pub fn open(root: &Path) -> Result<Self, PackageError> {
        let root = fs::canonicalize(root)
            .map_err(|source| PackageError::io("open model package", root.to_path_buf(), source))?;
        let metadata = fs::metadata(&root)
            .map_err(|source| PackageError::io("inspect model package", root.clone(), source))?;
        if !metadata.is_dir() {
            return Err(PackageError::Compatibility {
                field: "model package",
                reason: "expected a directory",
            });
        }
        let config_path = root.join("config.kv");
        let text = fs::read_to_string(&config_path)
            .map_err(|source| PackageError::io("read config.kv", config_path, source))?;
        let definition = schema::parse(&text)?;
        compatibility::validate(&definition)?;
        Ok(Self { root, definition })
    }

    pub fn schema_version(&self) -> SchemaVersion {
        self.definition.schema_version()
    }

    pub fn frontend(&self) -> &FrontendDefinition {
        self.definition.frontend()
    }

    pub fn ctc(&self) -> &CtcDefinition {
        self.definition.ctc()
    }

    pub fn rnnt(&self) -> &RnntDefinition {
        self.definition.rnnt()
    }

    pub fn vad(&self) -> &VadDefinition {
        self.definition.vad()
    }

    /// Loads the frontend's declared window and filter-bank assets after validating that
    /// both are regular files within this package.
    pub fn frontend_weights(&self) -> Result<FrontendWeights, PackageError> {
        let frontend = self.frontend();
        let window = validate_regular_asset(&self.root, frontend.window_asset())?;
        let filterbank = validate_regular_asset(&self.root, frontend.filterbank_asset())?;
        let (window_dimensions, window_values) = assets::read_f32(&window)?;
        let (filterbank_dimensions, filterbank_values) = assets::read_f32(&filterbank)?;
        Ok(FrontendWeights::new(
            window_dimensions,
            window_values,
            filterbank_dimensions,
            filterbank_values,
        ))
    }

    /// Selects a CTC encoder graph and validates only that selected graph.
    pub fn ctc_encoder(
        &self,
        precision: EncoderPrecision,
    ) -> Result<EncoderArtifact, PackageError> {
        let ctc = self.ctc();
        let artifact = validate_regular_asset(&self.root, ctc.encoder_asset(precision))?;
        Ok(EncoderArtifact::new(
            artifact,
            ctc.tensor_contract().clone(),
            precision,
            ctc.output_dimension(),
            ctc.output_layout(),
        ))
    }

    /// Selects an RNN-T encoder graph and validates only that selected graph.
    pub fn rnnt_encoder(
        &self,
        precision: EncoderPrecision,
    ) -> Result<EncoderArtifact, PackageError> {
        let rnnt = self.rnnt();
        let artifact = validate_regular_asset(&self.root, rnnt.encoder_asset(precision))?;
        Ok(EncoderArtifact::new(
            artifact,
            rnnt.encoder_tensor_contract().clone(),
            precision,
            rnnt.output_dimension(),
            rnnt.output_layout(),
        ))
    }

    /// Loads the CTC vocabulary after validating the selected required vocabulary asset.
    pub fn ctc_vocabulary(&self) -> Result<Vec<String>, PackageError> {
        let artifact = validate_regular_asset(&self.root, self.ctc().vocabulary_asset())?;
        assets::read_vocabulary(&artifact)
    }

    /// Selects RNN-T decoder, joint, and vocabulary artifacts for one precision. This method
    /// is not called when RNN-T is disabled, and does not validate the other precision.
    pub fn rnnt_assets(&self, precision: EncoderPrecision) -> Result<RnntAssets, PackageError> {
        let rnnt = self.rnnt();
        let decoder = validate_regular_asset(&self.root, rnnt.decoder_asset(precision))?;
        let joint = validate_regular_asset(&self.root, rnnt.joint_asset(precision))?;
        let vocabulary = validate_regular_asset(&self.root, rnnt.vocabulary_asset())?;
        Ok(RnntAssets::new(decoder, joint, vocabulary))
    }

    /// Selects the VAD artifact only for callers that enable VAD.
    pub fn vad_model(&self) -> Result<ValidatedArtifact, PackageError> {
        validate_regular_asset(&self.root, self.vad().model_asset())
    }
}

#[cfg(test)]
mod tests;
