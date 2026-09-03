// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

use crate::contracts::ChannelAudioView;
use gigaam_primitives::usize_to_f64;

/// Pearson correlation for two validated, equally sized nonempty channels.
///
/// A constant channel has no variance; returning zero gives callers bounded evidence that it is
/// not positively correlated without silently changing the frame alignment. A final tolerance-
/// bounded projection only compensates for f64 roundoff around the mathematical [-1, 1] limit.
pub fn channel_correlation(
    left: ChannelAudioView<'_>,
    right: ChannelAudioView<'_>,
) -> Result<f64, String> {
    if left.is_empty() || right.is_empty() {
        return Err("channel correlation requires nonempty channels".into());
    }
    if left.len() != right.len() {
        return Err("channel correlation requires equal channel lengths".into());
    }

    let count = usize_to_f64(left.len());
    let left_mean = left
        .samples()
        .iter()
        .map(|sample| f64::from(*sample))
        .sum::<f64>()
        / count;
    let right_mean = right
        .samples()
        .iter()
        .map(|sample| f64::from(*sample))
        .sum::<f64>()
        / count;
    let (mut numerator, mut left_energy, mut right_energy) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (left_sample, right_sample) in left.samples().iter().zip(right.samples()) {
        let left_delta = f64::from(*left_sample) - left_mean;
        let right_delta = f64::from(*right_sample) - right_mean;
        numerator += left_delta * right_delta;
        left_energy += left_delta * left_delta;
        right_energy += right_delta * right_delta;
    }
    let denominator = (left_energy * right_energy).sqrt();
    if denominator == 0.0 {
        return Ok(0.0);
    }
    let value = numerator / denominator;
    const ROUNDING_TOLERANCE: f64 = 1e-12;
    if !value.is_finite() || value < -1.0 - ROUNDING_TOLERANCE || value > 1.0 + ROUNDING_TOLERANCE {
        return Err("channel correlation is outside its bounded finite range".into());
    }
    Ok(value.clamp(-1.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::channel_correlation;
    use crate::ChannelAudio;

    #[test]
    fn correlation_refuses_misaligned_or_empty_channels() -> Result<(), String> {
        let empty = ChannelAudio::new(Vec::new())?;
        let one = ChannelAudio::new(vec![0.0])?;
        assert!(channel_correlation(empty.view(), one.view()).is_err());
        let two = ChannelAudio::new(vec![0.0, 1.0])?;
        assert!(channel_correlation(one.view(), two.view()).is_err());
        Ok(())
    }
}
