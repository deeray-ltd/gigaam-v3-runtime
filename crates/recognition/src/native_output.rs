// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Validation of f32 tensors extracted from Recognition native sessions.

/// A semantic role for one f32 output crossing a native Recognition boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NativeOutputRole {
    Encoder,
    RnntPredictionDecoder,
    RnntPredictionHidden,
    RnntPredictionCell,
    RnntJoint,
    Vad,
}

impl NativeOutputRole {
    const fn label(self) -> &'static str {
        match self {
            Self::Encoder => "encoder",
            Self::RnntPredictionDecoder => "RNN-T prediction decoder",
            Self::RnntPredictionHidden => "RNN-T prediction hidden state",
            Self::RnntPredictionCell => "RNN-T prediction cell state",
            Self::RnntJoint => "RNN-T joint",
            Self::Vad => "VAD",
        }
    }
}

/// The cardinality that a native output role may carry after its shape is validated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExpectedCardinality {
    /// The supported role layout supplies the complete cardinality.
    ShapeDerived,
    /// The role has an independently declared number of f32 values.
    Exact(usize),
}

/// Validates one extracted f32 output before it can enter Recognition state or an algorithm.
pub(crate) fn validate_f32_output(
    role: NativeOutputRole,
    shape: &[i64],
    values: &[f32],
    expected: ExpectedCardinality,
) -> Result<(), String> {
    let mut cardinality = 1_usize;
    for (axis, &dimension) in shape.iter().enumerate() {
        let dimension = usize::try_from(dimension).map_err(|_| {
            format!(
                "native {} output dimension at axis {axis} is invalid",
                role.label()
            )
        })?;
        cardinality = cardinality.checked_mul(dimension).ok_or_else(|| {
            format!(
                "native {} output shape product overflows usize",
                role.label()
            )
        })?;
    }

    if let ExpectedCardinality::Exact(expected) = expected
        && cardinality != expected
    {
        return Err(format!(
            "native {} output shape has {cardinality} values, expected {expected}",
            role.label()
        ));
    }
    if values.len() != cardinality {
        return Err(format!(
            "native {} output storage has {} values, expected {cardinality}",
            role.label(),
            values.len()
        ));
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(format!(
            "native {} output value at index {index} is not finite",
            role.label()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ExpectedCardinality, NativeOutputRole, validate_f32_output};

    const ROLES: [NativeOutputRole; 6] = [
        NativeOutputRole::Encoder,
        NativeOutputRole::RnntPredictionDecoder,
        NativeOutputRole::RnntPredictionHidden,
        NativeOutputRole::RnntPredictionCell,
        NativeOutputRole::RnntJoint,
        NativeOutputRole::Vad,
    ];

    #[test]
    fn finite_native_controls_preserve_all_role_cardinalities() {
        for role in ROLES {
            assert!(
                validate_f32_output(
                    role,
                    &[1, 3],
                    &[-1.0, 0.0, 1.0],
                    ExpectedCardinality::Exact(3),
                )
                .is_ok(),
                "finite {role:?} control must remain valid"
            );
        }
        assert!(
            validate_f32_output(
                NativeOutputRole::Encoder,
                &[1, 0, 4],
                &[],
                ExpectedCardinality::ShapeDerived,
            )
            .is_ok()
        );
    }

    #[test]
    fn native_validator_refuses_invalid_shape_and_storage_cardinality() {
        assert!(
            validate_f32_output(
                NativeOutputRole::Encoder,
                &[1, -1, 4],
                &[],
                ExpectedCardinality::ShapeDerived,
            )
            .is_err()
        );
        assert!(
            validate_f32_output(
                NativeOutputRole::RnntJoint,
                &[i64::MAX, i64::MAX],
                &[],
                ExpectedCardinality::ShapeDerived,
            )
            .is_err()
        );
        assert!(
            validate_f32_output(
                NativeOutputRole::Vad,
                &[2],
                &[0.2],
                ExpectedCardinality::Exact(2),
            )
            .is_err()
        );
        assert!(
            validate_f32_output(
                NativeOutputRole::RnntPredictionDecoder,
                &[2],
                &[0.2, 0.8],
                ExpectedCardinality::Exact(3),
            )
            .is_err()
        );
    }

    #[test]
    fn native_validator_refuses_every_nonfinite_value_for_every_role() {
        for role in ROLES {
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                for index in 0..3 {
                    let mut values = [0.1, 0.2, 0.3];
                    values[index] = invalid;
                    assert!(
                        validate_f32_output(role, &[3], &values, ExpectedCardinality::Exact(3),)
                            .is_err(),
                        "{role:?} must refuse a non-finite value at index {index}"
                    );
                }
            }
        }
    }
}
