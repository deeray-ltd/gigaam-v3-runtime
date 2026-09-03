// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

use crate::definition::PackageDefinition;
use crate::error::PackageError;

pub(crate) fn validate(definition: &PackageDefinition) -> Result<(), PackageError> {
    let frontend = definition.frontend();
    for (field, value) in [
        ("sample_rate", frontend.sample_rate),
        ("n_mels", frontend.n_mels),
        ("hop_length", frontend.hop_length),
        ("n_fft", frontend.n_fft),
    ] {
        if value == 0 {
            return Err(PackageError::Compatibility {
                field,
                reason: "must be non-zero",
            });
        }
    }
    if frontend.n_fft < 2 {
        return Err(PackageError::Compatibility {
            field: "n_fft",
            reason: "must describe at least one real FFT bin pair",
        });
    }
    if !frontend.log_clamp_min.is_finite()
        || !frontend.log_clamp_max.is_finite()
        || frontend.log_clamp_min <= 0.0
        || frontend.log_clamp_max < frontend.log_clamp_min
    {
        return Err(PackageError::Compatibility {
            field: "log_clamp_min/log_clamp_max",
            reason: "must be finite, positive, and ordered",
        });
    }
    if !frontend.frames_per_second.is_finite() || frontend.frames_per_second <= 0.0 {
        return Err(PackageError::Compatibility {
            field: "frames_per_sec",
            reason: "must be finite and positive",
        });
    }

    let ctc = definition.ctc();
    if ctc.output_dimension == 0 || ctc.blank_id >= ctc.output_dimension {
        return Err(PackageError::Compatibility {
            field: "ctc.blank_id/ctc.out_dim",
            reason: "blank identifier must fit the declared CTC output dimension",
        });
    }

    let rnnt = definition.rnnt();
    if rnnt.prediction_hidden == 0 || rnnt.output_dimension == 0 {
        return Err(PackageError::Compatibility {
            field: "rnnt.pred_hidden/rnnt.out_dim",
            reason: "must be non-zero",
        });
    }
    if rnnt.max_symbols_per_step == 0 {
        return Err(PackageError::Compatibility {
            field: "rnnt.max_symbols_per_step",
            reason: "zero would prevent all non-blank decoding",
        });
    }
    if !definition.retained().values_are_nonempty() {
        return Err(PackageError::Compatibility {
            field: "retained V1 metadata",
            reason: "present metadata values must be non-empty",
        });
    }
    Ok(())
}
