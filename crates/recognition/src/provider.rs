// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Exact execution-provider parsing, configuration, and resolved-plan construction.

use std::num::NonZeroUsize;
use std::path::PathBuf;

/// One logical CUDA encoder role for role-specific graph-assignment observation.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum EncoderRole {
    Ctc,
    Rnnt,
}

impl EncoderRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ctc => "ctc",
            Self::Rnnt => "rnnt",
        }
    }
}

/// The exact encoder roles a process will construct after package validation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RequiredEncoderRoles {
    ctc: bool,
    rnnt: bool,
}

impl RequiredEncoderRoles {
    pub const fn none() -> Self {
        Self {
            ctc: false,
            rnnt: false,
        }
    }

    pub const fn ctc() -> Self {
        Self {
            ctc: true,
            rnnt: false,
        }
    }

    pub const fn rnnt() -> Self {
        Self {
            ctc: false,
            rnnt: true,
        }
    }

    pub const fn ctc_and_rnnt() -> Self {
        Self {
            ctc: true,
            rnnt: true,
        }
    }

    pub const fn contains(self, role: EncoderRole) -> bool {
        match role {
            EncoderRole::Ctc => self.ctc,
            EncoderRole::Rnnt => self.rnnt,
        }
    }
}

/// A validated SHA-256 fingerprint of one canonical ONNX Runtime assignment multiset.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaAssignmentFingerprint([u8; 32]);

impl CudaAssignmentFingerprint {
    pub fn parse(value: &str) -> Result<Self, String> {
        if value.len() != 64 {
            return Err(
                "CUDA assignment SHA-256 must contain exactly 64 hexadecimal characters".into(),
            );
        }
        let mut bytes = [0_u8; 32];
        for (index, output) in bytes.iter_mut().enumerate() {
            let start = index
                .checked_mul(2)
                .ok_or_else(|| "CUDA assignment SHA-256 index overflows".to_owned())?;
            let high = value
                .as_bytes()
                .get(start)
                .copied()
                .ok_or_else(|| "CUDA assignment SHA-256 is truncated".to_owned())?;
            let low = value
                .as_bytes()
                .get(start + 1)
                .copied()
                .ok_or_else(|| "CUDA assignment SHA-256 is truncated".to_owned())?;
            let high = hexadecimal_nibble(high)?;
            let low = hexadecimal_nibble(low)?;
            *output = high
                .checked_mul(16)
                .and_then(|value| value.checked_add(low))
                .ok_or_else(|| "CUDA assignment SHA-256 byte overflows".to_owned())?;
        }
        Ok(Self(bytes))
    }

    pub fn to_hex(&self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    pub(crate) const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

fn hexadecimal_nibble(byte: u8) -> Result<u8, String> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err("CUDA assignment SHA-256 must contain only hexadecimal characters".into()),
    }
}

/// Raw CUDA-assignment inputs supplied by one process boundary before package or ORT work.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaAssignmentEnvironment<'a> {
    pub policy: Option<&'a str>,
    pub ctc_sha256: Option<&'a str>,
    pub rnnt_sha256: Option<&'a str>,
}

impl CudaAssignmentEnvironment<'_> {
    const fn is_configured(self) -> bool {
        self.policy.is_some() || self.ctc_sha256.is_some() || self.rnnt_sha256.is_some()
    }
}

/// CUDA assignment verification selected before package or ONNX Runtime initialization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CudaAssignmentPolicy {
    Verified {
        required_roles: RequiredEncoderRoles,
        ctc: Option<CudaAssignmentFingerprint>,
        rnnt: Option<CudaAssignmentFingerprint>,
    },
    AllowUnverified {
        required_roles: RequiredEncoderRoles,
    },
}

impl CudaAssignmentPolicy {
    pub fn from_environment(
        device: Device,
        required_roles: RequiredEncoderRoles,
        environment: CudaAssignmentEnvironment<'_>,
    ) -> Result<Option<Self>, String> {
        if device != Device::Cuda {
            if environment.is_configured() {
                return Err(
                    "ASR_CUDA_ASSIGNMENT_* settings require the selected cuda provider".into(),
                );
            }
            return Ok(None);
        }

        let ctc = environment
            .ctc_sha256
            .map(CudaAssignmentFingerprint::parse)
            .transpose()
            .map_err(|error| format!("ASR_CUDA_CTC_ASSIGNMENT_SHA256: {error}"))?;
        let rnnt = environment
            .rnnt_sha256
            .map(CudaAssignmentFingerprint::parse)
            .transpose()
            .map_err(|error| format!("ASR_CUDA_RNNT_ASSIGNMENT_SHA256: {error}"))?;
        match environment.policy {
            None | Some("verified") => {
                if required_roles.contains(EncoderRole::Ctc) && ctc.is_none() {
                    return Err(
                        "ASR_CUDA_CTC_ASSIGNMENT_SHA256 is required for verified CUDA CTC composition"
                            .into(),
                    );
                }
                if required_roles.contains(EncoderRole::Rnnt) && rnnt.is_none() {
                    return Err(
                        "ASR_CUDA_RNNT_ASSIGNMENT_SHA256 is required for verified CUDA RNN-T composition"
                            .into(),
                    );
                }
                Ok(Some(Self::Verified {
                    required_roles,
                    ctc,
                    rnnt,
                }))
            }
            Some("allow-unverified") => {
                if ctc.is_some() || rnnt.is_some() {
                    return Err(
                        "ASR_CUDA_ASSIGNMENT_POLICY=allow-unverified conflicts with CUDA assignment SHA-256 settings"
                            .into(),
                    );
                }
                Ok(Some(Self::AllowUnverified { required_roles }))
            }
            Some(value) => Err(format!(
                "ASR_CUDA_ASSIGNMENT_POLICY must be verified or allow-unverified, got {value:?}"
            )),
        }
    }

    pub const fn is_allow_unverified(&self) -> bool {
        matches!(self, Self::AllowUnverified { .. })
    }

    /// Admits one declared CUDA encoder role before native session construction.
    fn admit_role(&self, role: EncoderRole) -> Result<(), String> {
        let required_roles = match self {
            Self::Verified { required_roles, .. } | Self::AllowUnverified { required_roles } => {
                *required_roles
            }
        };
        if !required_roles.contains(role) {
            return Err(format!(
                "CUDA {} encoder session is not declared by this process composition",
                role.as_str()
            ));
        }
        if let Self::Verified { ctc, rnnt, .. } = self {
            match role {
                EncoderRole::Ctc => ctc.as_ref(),
                EncoderRole::Rnnt => rnnt.as_ref(),
            }
            .ok_or_else(|| {
                format!(
                    "CUDA {} encoder session has no declared assignment fingerprint",
                    role.as_str()
                )
            })?;
        }
        Ok(())
    }

    fn verify_role(
        &self,
        role: EncoderRole,
        observed: &CudaAssignmentFingerprint,
    ) -> Result<(), String> {
        self.admit_role(role)?;
        match self {
            Self::Verified { ctc, rnnt, .. } => {
                let expected = match role {
                    EncoderRole::Ctc => ctc.as_ref(),
                    EncoderRole::Rnnt => rnnt.as_ref(),
                }
                .ok_or_else(|| {
                    "CUDA role admission must retain its verified fingerprint".to_owned()
                })?;
                if expected != observed {
                    return Err(format!(
                        "CUDA {} assignment fingerprint mismatch: expected {}, observed {}",
                        role.as_str(),
                        expected.to_hex(),
                        observed.to_hex()
                    ));
                }
                Ok(())
            }
            Self::AllowUnverified { .. } => Ok(()),
        }
    }
}

/// Execution provider selected for an encoder session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Device {
    Cuda,
    Cpu,
    /// TensorRT requires a CUDA-capable runtime but never enables CUDA as a second encoder provider.
    Tensorrt,
}

/// Whether ONNX Runtime may use its memory-pattern optimizer.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MemoryPattern {
    #[default]
    Enabled,
    Disabled,
}

/// CUDA arena growth policy accepted by this runtime.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CudaArena {
    #[default]
    Default,
    SameAsRequested,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TensorRtShape {
    name: String,
    dimensions: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TensorRtShapeSet(pub(crate) Vec<TensorRtShape>);

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TensorRtProfile {
    pub(crate) min: TensorRtShapeSet,
    pub(crate) opt: TensorRtShapeSet,
    pub(crate) max: TensorRtShapeSet,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct TensorRtConfig {
    pub(crate) cache_dir: Option<PathBuf>,
    pub(crate) profile: Option<TensorRtProfile>,
}

impl TensorRtConfig {
    fn is_configured(&self) -> bool {
        self.cache_dir.is_some() || self.profile.is_some()
    }
}

/// Typed native provider settings supplied by a process boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OrtConfig {
    pub(crate) memory_pattern: MemoryPattern,
    pub(crate) cuda_arena: CudaArena,
    pub(crate) tensorrt: TensorRtConfig,
    pub(crate) intra_threads: Option<NonZeroUsize>,
}

/// Borrowed raw values that a process adapter has already read from its environment or arguments.
#[derive(Clone, Copy, Debug, Default)]
pub struct OrtEnvironment<'a> {
    pub memory_pattern: Option<&'a str>,
    pub cuda_arena: Option<&'a str>,
    pub tensorrt_cache: Option<&'a str>,
    pub tensorrt_profile_min: Option<&'a str>,
    pub tensorrt_profile_opt: Option<&'a str>,
    pub tensorrt_profile_max: Option<&'a str>,
    pub intra_threads: Option<&'a str>,
}

/// One validated encoder-provider decision. Native session constructors accept this value instead
/// of a loose device/configuration pair.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderPlan {
    device: Device,
    pub(crate) config: OrtConfig,
    cuda_assignment_policy: Option<CudaAssignmentPolicy>,
}

impl ProviderPlan {
    pub fn new(device: Device, config: OrtConfig) -> Result<Self, String> {
        Self::new_with_cuda_assignment(
            device,
            config,
            RequiredEncoderRoles::none(),
            CudaAssignmentEnvironment::default(),
        )
    }

    pub fn new_with_cuda_assignment(
        device: Device,
        config: OrtConfig,
        required_roles: RequiredEncoderRoles,
        assignment_environment: CudaAssignmentEnvironment<'_>,
    ) -> Result<Self, String> {
        device.require_compiled_feature()?;
        config.validate_for(device)?;
        let cuda_assignment_policy =
            CudaAssignmentPolicy::from_environment(device, required_roles, assignment_environment)?;
        Ok(Self {
            device,
            config,
            cuda_assignment_policy,
        })
    }

    pub const fn device(&self) -> Device {
        self.device
    }

    pub const fn cuda_assignment_policy(&self) -> Option<&CudaAssignmentPolicy> {
        self.cuda_assignment_policy.as_ref()
    }

    /// The intra-op thread count applied to every ONNX Runtime session this plan constructs,
    /// or `None` to leave the library default in place.
    pub const fn intra_threads(&self) -> Option<NonZeroUsize> {
        self.config.intra_threads
    }

    /// Refuses an undeclared CUDA role before constructing a native session.
    pub(crate) fn preflight_cuda_role(&self, role: EncoderRole) -> Result<(), String> {
        match (self.device, self.cuda_assignment_policy()) {
            (Device::Cuda, Some(policy)) => policy.admit_role(role),
            (Device::Cuda, None) => {
                Err("CUDA encoder sessions require a CUDA assignment policy".into())
            }
            (Device::Cpu | Device::Tensorrt, _) => {
                Err("CUDA encoder role preflight requires the selected cuda provider".into())
            }
        }
    }

    pub(crate) fn verify_cuda_assignment(
        &self,
        role: EncoderRole,
        observed: &CudaAssignmentFingerprint,
    ) -> Result<(), String> {
        match self.cuda_assignment_policy() {
            Some(policy) => policy.verify_role(role, observed),
            None => Err("CUDA encoder sessions require a CUDA assignment policy".into()),
        }
    }
}

impl OrtConfig {
    /// Parses exact provider settings after the application boundary has supplied raw values.
    pub fn from_environment(values: OrtEnvironment<'_>) -> Result<Self, String> {
        let memory_pattern = match values.memory_pattern {
            None | Some("1") => MemoryPattern::Enabled,
            Some("0") => MemoryPattern::Disabled,
            Some(value) => {
                return Err(format!("ASR_ORT_MEMPATTERN must be 0 or 1, got {value:?}"));
            }
        };
        let cuda_arena = match values.cuda_arena {
            None | Some("default") => CudaArena::Default,
            Some("same") => CudaArena::SameAsRequested,
            Some(value) => {
                return Err(format!(
                    "ASR_ORT_ARENA must be default or same, got {value:?}"
                ));
            }
        };
        let cache_dir = match values.tensorrt_cache {
            None => None,
            Some("") => return Err("ASR_TRT_CACHE must name an existing directory".into()),
            Some(value) => {
                let path = PathBuf::from(value);
                let metadata = std::fs::metadata(&path)
                    .map_err(|error| format!("ASR_TRT_CACHE {}: {error}", path.display()))?;
                if !metadata.is_dir() {
                    return Err(format!(
                        "ASR_TRT_CACHE {} is not a directory",
                        path.display()
                    ));
                }
                Some(path)
            }
        };
        let profile = TensorRtProfile::from_environment(
            values.tensorrt_profile_min,
            values.tensorrt_profile_opt,
            values.tensorrt_profile_max,
        )?;
        let intra_threads = match values.intra_threads {
            None => None,
            Some(value) => {
                let invalid = || {
                    format!(
                        "ASR_ORT_THREADS must be a positive integer of at most 2147483647, \
                         got {value:?}"
                    )
                };
                if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
                    return Err(invalid());
                }
                let parsed = value.parse::<i32>().map_err(|_| invalid())?;
                let parsed = usize::try_from(parsed).map_err(|_| invalid())?;
                Some(NonZeroUsize::new(parsed).ok_or_else(invalid)?)
            }
        };
        Ok(Self {
            memory_pattern,
            cuda_arena,
            tensorrt: TensorRtConfig { cache_dir, profile },
            intra_threads,
        })
    }

    fn validate_for(&self, device: Device) -> Result<(), String> {
        if self.cuda_arena != CudaArena::Default && device != Device::Cuda {
            return Err("ASR_ORT_ARENA requires the selected cuda provider".into());
        }
        if self.tensorrt.is_configured() && device != Device::Tensorrt {
            return Err("ASR_TRT_* settings require the selected tensorrt provider".into());
        }
        Ok(())
    }
}

impl TensorRtProfile {
    fn from_environment(
        min: Option<&str>,
        opt: Option<&str>,
        max: Option<&str>,
    ) -> Result<Option<Self>, String> {
        match (min, opt, max) {
            (None, None, None) => Ok(None),
            (Some(min), Some(opt), Some(max)) => {
                let profile = Self {
                    min: TensorRtShapeSet::parse(min, "ASR_TRT_PROFILE_MIN")?,
                    opt: TensorRtShapeSet::parse(opt, "ASR_TRT_PROFILE_OPT")?,
                    max: TensorRtShapeSet::parse(max, "ASR_TRT_PROFILE_MAX")?,
                };
                profile.validate_order()?;
                Ok(Some(profile))
            }
            _ => Err(
                "ASR_TRT_PROFILE_MIN, ASR_TRT_PROFILE_OPT, and ASR_TRT_PROFILE_MAX must be set together"
                    .into(),
            ),
        }
    }

    fn validate_order(&self) -> Result<(), String> {
        for ((min, opt), max) in self.min.0.iter().zip(&self.opt.0).zip(&self.max.0) {
            if min.name != opt.name
                || min.name != max.name
                || min.dimensions.len() != opt.dimensions.len()
                || min.dimensions.len() != max.dimensions.len()
            {
                return Err(
                    "ASR_TRT_PROFILE_* must name the same inputs with the same rank and order"
                        .into(),
                );
            }
            for ((min_dimension, opt_dimension), max_dimension) in min
                .dimensions
                .iter()
                .zip(&opt.dimensions)
                .zip(&max.dimensions)
            {
                if min_dimension > opt_dimension || opt_dimension > max_dimension {
                    return Err(format!(
                        "ASR_TRT_PROFILE_* dimensions for {} must satisfy min <= opt <= max",
                        min.name
                    ));
                }
            }
        }
        if self.min.0.len() != self.opt.0.len() || self.min.0.len() != self.max.0.len() {
            return Err(
                "ASR_TRT_PROFILE_* must name the same inputs with the same rank and order".into(),
            );
        }
        Ok(())
    }
}

impl TensorRtShapeSet {
    fn parse(value: &str, key: &str) -> Result<Self, String> {
        if value.is_empty() {
            return Err(format!("{key} must not be empty"));
        }
        let mut shapes = Vec::new();
        for item in value.split(',') {
            let (name, dimensions) = item
                .split_once(':')
                .ok_or_else(|| format!("{key} item {item:?} must be input:d0xd1[...]"))?;
            if !valid_tensor_name(name) {
                return Err(format!("{key} input name {name:?} is invalid"));
            }
            if shapes
                .iter()
                .any(|shape: &TensorRtShape| shape.name == name)
            {
                return Err(format!("{key} repeats input {name:?}"));
            }
            let mut parsed_dimensions = Vec::new();
            for dimension in dimensions.split('x') {
                let parsed = dimension.parse::<usize>().map_err(|_| {
                    format!("{key} dimension {dimension:?} for {name} must be a positive integer")
                })?;
                if parsed == 0 {
                    return Err(format!(
                        "{key} dimension for {name} must be a positive integer"
                    ));
                }
                parsed_dimensions.push(parsed);
            }
            if parsed_dimensions.is_empty() {
                return Err(format!("{key} input {name:?} must have dimensions"));
            }
            shapes.push(TensorRtShape {
                name: name.to_owned(),
                dimensions: parsed_dimensions,
            });
        }
        Ok(Self(shapes))
    }

    #[cfg(feature = "tensorrt")]
    pub(crate) fn ort_value(&self) -> String {
        self.0
            .iter()
            .map(|shape| {
                let dimensions = shape
                    .dimensions
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join("x");
                format!("{}:{dimensions}", shape.name)
            })
            .collect::<Vec<_>>()
            .join(",")
    }
}

fn valid_tensor_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(first) if first.is_ascii_alphabetic() || first == '_' => {}
        _ => return false,
    }
    chars.all(|character| character.is_ascii_alphanumeric() || character == '_')
}

impl Device {
    /// Resolves one exact provider from independently supplied command and environment values.
    pub fn resolve_with_config(
        ep_cli: Option<&str>,
        env_ep: Option<&str>,
    ) -> Result<Device, String> {
        match (ep_cli, env_ep) {
            (Some(cli), Some(env)) if cli != env => Err(format!(
                "--ep ({cli}) conflicts with ASR_ENCODER_EP ({env}); configure one exact provider"
            )),
            (Some(value), _) | (None, Some(value)) => Self::parse(value),
            (None, None) => Ok(Self::build_default()),
        }
    }

    /// Parses exactly one public provider spelling.
    pub fn parse(value: &str) -> Result<Device, String> {
        match value {
            "cuda" => {
                #[cfg(feature = "cuda")]
                {
                    Ok(Self::Cuda)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err("cuda EP is unavailable in this build; rebuild with --features cuda".into())
                }
            }
            "cpu" => Ok(Self::Cpu),
            "tensorrt" => {
                #[cfg(feature = "tensorrt")]
                {
                    Ok(Self::Tensorrt)
                }
                #[cfg(not(feature = "tensorrt"))]
                {
                    Err("tensorrt EP is unavailable in this build; rebuild with --features tensorrt".into())
                }
            }
            "" => Err("ASR_ENCODER_EP/--ep must be one of cpu|cuda|tensorrt, not empty".into()),
            other => Err(format!(
                "ASR_ENCODER_EP/--ep: unknown provider \"{other}\" (expected cpu|cuda|tensorrt)"
            )),
        }
    }

    /// The provider selected only when neither process source supplied a value.
    pub const fn build_default() -> Device {
        #[cfg(feature = "cuda")]
        {
            Self::Cuda
        }
        #[cfg(not(feature = "cuda"))]
        {
            Self::Cpu
        }
    }

    /// Stable provider text used by process adapters.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Tensorrt => "tensorrt",
        }
    }

    fn require_compiled_feature(self) -> Result<(), String> {
        match self {
            Self::Cpu => Ok(()),
            Self::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    Ok(())
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err("cuda EP is unavailable in this build; rebuild with --features cuda".into())
                }
            }
            Self::Tensorrt => {
                #[cfg(feature = "tensorrt")]
                {
                    Ok(())
                }
                #[cfg(not(feature = "tensorrt"))]
                {
                    Err("tensorrt EP is unavailable in this build; rebuild with --features tensorrt".into())
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CudaArena, CudaAssignmentEnvironment, CudaAssignmentFingerprint, CudaAssignmentPolicy,
        Device, EncoderRole, MemoryPattern, OrtConfig, OrtEnvironment, ProviderPlan,
        RequiredEncoderRoles,
    };

    const ZERO_FINGERPRINT: &str =
        "0000000000000000000000000000000000000000000000000000000000000000";
    const ONE_FINGERPRINT: &str =
        "1111111111111111111111111111111111111111111111111111111111111111";

    fn plan(device: Device, config: OrtConfig) -> Result<ProviderPlan, String> {
        ProviderPlan::new(device, config)
    }

    #[test]
    fn conflicting_provider_sources_are_refused() {
        assert_eq!(
            Device::resolve_with_config(Some("cpu"), Some("cuda")),
            Err(
                "--ep (cpu) conflicts with ASR_ENCODER_EP (cuda); configure one exact provider"
                    .into()
            )
        );
    }

    #[test]
    fn cpu_provider_plan_is_available_in_every_build() {
        let resolved = Device::resolve_with_config(Some("cpu"), None)
            .expect("the CPU provider is always compiled");
        let resolved_plan = plan(resolved, OrtConfig::default())
            .expect("the CPU provider needs no accelerator configuration");
        assert_eq!(resolved_plan.device(), Device::Cpu);
    }

    #[test]
    fn provider_aliases_and_empty_values_are_refused() {
        for value in ["", "gpu", "trt", "CUDA", " cpu"] {
            assert!(Device::parse(value).is_err(), "{value:?} must be refused");
        }
    }

    #[test]
    fn verified_cuda_policy_requires_each_declared_role_before_native_work() {
        let missing_ctc = CudaAssignmentPolicy::from_environment(
            Device::Cuda,
            RequiredEncoderRoles::ctc(),
            CudaAssignmentEnvironment::default(),
        );
        assert!(missing_ctc.is_err());

        let missing_rnnt = CudaAssignmentPolicy::from_environment(
            Device::Cuda,
            RequiredEncoderRoles::ctc_and_rnnt(),
            CudaAssignmentEnvironment {
                ctc_sha256: Some(ZERO_FINGERPRINT),
                ..CudaAssignmentEnvironment::default()
            },
        );
        assert!(missing_rnnt.is_err());
    }

    #[test]
    fn verified_cuda_policy_rejects_undeclared_roles_and_ignores_extra_valid_fingerprints()
    -> Result<(), String> {
        let policy = CudaAssignmentPolicy::from_environment(
            Device::Cuda,
            RequiredEncoderRoles::ctc(),
            CudaAssignmentEnvironment {
                ctc_sha256: Some(ZERO_FINGERPRINT),
                rnnt_sha256: Some(ONE_FINGERPRINT),
                ..CudaAssignmentEnvironment::default()
            },
        )?
        .ok_or_else(|| "CUDA must construct an assignment policy".to_owned())?;
        let observed = CudaAssignmentFingerprint::parse(ZERO_FINGERPRINT)?;
        policy.admit_role(EncoderRole::Ctc)?;
        assert!(policy.admit_role(EncoderRole::Rnnt).is_err());
        policy.verify_role(EncoderRole::Ctc, &observed)?;
        assert!(policy.verify_role(EncoderRole::Rnnt, &observed).is_err());
        Ok(())
    }

    #[test]
    fn allow_unverified_is_explicit_and_conflicts_with_expected_fingerprints() -> Result<(), String>
    {
        let policy = CudaAssignmentPolicy::from_environment(
            Device::Cuda,
            RequiredEncoderRoles::rnnt(),
            CudaAssignmentEnvironment {
                policy: Some("allow-unverified"),
                ..CudaAssignmentEnvironment::default()
            },
        )?
        .ok_or_else(|| "CUDA must construct an assignment policy".to_owned())?;
        assert!(policy.is_allow_unverified());
        policy.verify_role(
            EncoderRole::Rnnt,
            &CudaAssignmentFingerprint::parse(ONE_FINGERPRINT)?,
        )?;

        let conflict = CudaAssignmentPolicy::from_environment(
            Device::Cuda,
            RequiredEncoderRoles::ctc(),
            CudaAssignmentEnvironment {
                policy: Some("allow-unverified"),
                ctc_sha256: Some(ZERO_FINGERPRINT),
                ..CudaAssignmentEnvironment::default()
            },
        );
        assert!(conflict.is_err());
        Ok(())
    }

    #[test]
    fn assignment_environment_requires_cuda_and_rejects_invalid_values() {
        for device in [Device::Cpu, Device::Tensorrt] {
            assert!(
                CudaAssignmentPolicy::from_environment(
                    device,
                    RequiredEncoderRoles::none(),
                    CudaAssignmentEnvironment {
                        policy: Some("verified"),
                        ..CudaAssignmentEnvironment::default()
                    },
                )
                .is_err()
            );
        }
        for environment in [
            CudaAssignmentEnvironment {
                policy: Some(""),
                ..CudaAssignmentEnvironment::default()
            },
            CudaAssignmentEnvironment {
                ctc_sha256: Some("not-a-digest"),
                ..CudaAssignmentEnvironment::default()
            },
            CudaAssignmentEnvironment {
                policy: Some("permissive"),
                ..CudaAssignmentEnvironment::default()
            },
        ] {
            assert!(
                CudaAssignmentPolicy::from_environment(
                    Device::Cuda,
                    RequiredEncoderRoles::none(),
                    environment,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn provider_configuration_is_typed_and_coherent_before_session_creation() {
        let config = OrtConfig::from_environment(OrtEnvironment {
            memory_pattern: Some("0"),
            cuda_arena: Some("same"),
            ..OrtEnvironment::default()
        })
        .expect("exact ORT settings must produce typed configuration");
        assert_eq!(config.memory_pattern, MemoryPattern::Disabled);
        assert_eq!(config.cuda_arena, CudaArena::SameAsRequested);
        assert!(plan(Device::Cpu, config.clone()).is_err());
        assert!(
            OrtConfig::from_environment(OrtEnvironment {
                memory_pattern: Some("off"),
                ..OrtEnvironment::default()
            })
            .is_err()
        );
        assert!(
            OrtConfig::from_environment(OrtEnvironment {
                tensorrt_profile_min: Some("features:1x64x99"),
                ..OrtEnvironment::default()
            })
            .is_err()
        );
        assert!(
            OrtConfig::from_environment(OrtEnvironment {
                tensorrt_profile_min: Some("features:1x64x100,feature_lengths:1"),
                tensorrt_profile_opt: Some("features:1x64x99,feature_lengths:1"),
                tensorrt_profile_max: Some("features:1x64x2999,feature_lengths:1"),
                ..OrtEnvironment::default()
            })
            .is_err()
        );
    }

    #[test]
    fn intra_thread_count_environment_value_is_typed_and_validated() {
        let absent = OrtConfig::from_environment(OrtEnvironment::default())
            .expect("an absent thread count leaves the library default");
        assert_eq!(absent.intra_threads, None);

        let positive = OrtConfig::from_environment(OrtEnvironment {
            intra_threads: Some("4"),
            ..OrtEnvironment::default()
        })
        .expect("a positive documented thread count must parse");
        assert_eq!(positive.intra_threads.map(|threads| threads.get()), Some(4));

        let maximum = OrtConfig::from_environment(OrtEnvironment {
            intra_threads: Some("2147483647"),
            ..OrtEnvironment::default()
        })
        .expect("the documented maximum thread count must parse");
        assert_eq!(
            maximum.intra_threads.map(|threads| threads.get()),
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
            assert!(
                OrtConfig::from_environment(OrtEnvironment {
                    intra_threads: Some(invalid),
                    ..OrtEnvironment::default()
                })
                .is_err(),
                "{invalid:?} must be refused"
            );
        }
    }

    #[test]
    fn tensor_rt_settings_require_a_tensor_rt_plan() {
        let config = OrtConfig::from_environment(OrtEnvironment {
            tensorrt_cache: Some("."),
            tensorrt_profile_min: Some("features:1x64x99,feature_lengths:1"),
            tensorrt_profile_opt: Some("features:1x64x2999,feature_lengths:1"),
            tensorrt_profile_max: Some("features:1x64x6000,feature_lengths:1"),
            ..OrtEnvironment::default()
        })
        .expect("valid TensorRT settings must parse before a session is opened");
        assert!(plan(Device::Cuda, config.clone()).is_err());
        #[cfg(feature = "tensorrt")]
        assert!(plan(Device::Tensorrt, config).is_ok());
        #[cfg(not(feature = "tensorrt"))]
        assert!(plan(Device::Tensorrt, config).is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn direct_cuda_plan_requires_the_cuda_feature_and_accepts_it_here() {
        let resolved_plan = plan(Device::Cuda, OrtConfig::default())
            .expect("the CUDA feature makes direct CUDA plans valid");
        assert_eq!(resolved_plan.device(), Device::Cuda);
        assert!(resolved_plan.preflight_cuda_role(EncoderRole::Ctc).is_err());
        assert_eq!(Device::resolve_with_config(None, None), Ok(Device::Cuda));
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn direct_cuda_plan_refuses_without_the_cuda_feature() {
        assert!(plan(Device::Cuda, OrtConfig::default()).is_err());
        assert_eq!(Device::resolve_with_config(None, None), Ok(Device::Cpu));
    }

    #[cfg(feature = "tensorrt")]
    #[test]
    fn direct_tensor_rt_plan_requires_the_tensor_rt_feature_and_accepts_it_here() {
        let resolved_plan = plan(Device::Tensorrt, OrtConfig::default())
            .expect("the TensorRT feature makes direct TensorRT plans valid");
        assert_eq!(resolved_plan.device(), Device::Tensorrt);
    }

    #[cfg(not(feature = "tensorrt"))]
    #[test]
    fn direct_tensor_rt_plan_refuses_without_the_tensor_rt_feature() {
        assert!(plan(Device::Tensorrt, OrtConfig::default()).is_err());
    }
}
