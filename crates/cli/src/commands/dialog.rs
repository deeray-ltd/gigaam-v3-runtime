// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Offline multi-channel dialogue command adapter.

use crate::composition;
use crate::configuration;
use crate::grammar::DialogInvocation;
use crate::projection;
use gigaam_audio::load;
use gigaam_recognition::RequiredEncoderRoles;
use gigaam_transcription::{
    BackchannelDuration, BackchannelPolicy, OfflineDialogSetup, OfflineDialogSetupInput,
    StreamConfig, TurnGap, transcribe_offline_dialog,
};

/// Executes offline dialogue reconstruction using direct channel-local recognition composition.
pub(crate) fn run(invocation: DialogInvocation) -> Result<(), String> {
    let runtime = configuration::runtime(&invocation.runtime, RequiredEncoderRoles::ctc())
        .map_err(|error| format!("runtime configuration: {error}"))?;
    let turn_gap = TurnGap::new(invocation.turn_gap_seconds)
        .map_err(|error| format!("--turn-gap: {error}"))?;
    let backchannel_policy = backchannel_policy(invocation.backchannel_maximum_milliseconds)?;
    let frontend_mode =
        configuration::frontend_mode().map_err(|error| format!("frontend: {error}"))?;

    let package = composition::open_package(&invocation.model)?;
    let sample_rate = composition::package_sample_rate(&package)?;
    let stream_config = StreamConfig::checked_default(sample_rate)
        .map_err(|error| format!("dialog stream configuration: {error}"))?
        .with_endpoint_source(invocation.endpoint)
        .map_err(|error| format!("dialog stream configuration: {error}"))?;
    let setup = OfflineDialogSetup::new(OfflineDialogSetupInput {
        channel_policy: invocation.channel_policy,
        stream_config,
        turn_gap,
        backchannel_policy,
    })
    .map_err(|error| format!("dialog setup: {error}"))?;
    composition::initialize_runtime(&runtime)?;
    let frontend = composition::frontend_for_package(&package, frontend_mode)
        .map_err(|error| format!("frontend: {error}"))?;
    let loaded = load(&invocation.input)?;
    let audio = composition::resample_to(loaded, frontend.sample_rate())
        .map_err(|error| format!("resample: {error}"))?;
    let channels = audio.channels().len();
    let mut factory = composition::DialogStreamFactory::new(&package, &runtime, frontend);
    let result = transcribe_offline_dialog(setup, &mut factory, audio.channels())
        .map_err(|failure| projection::dialog_failure_message(&failure))?;
    projection::dialog(&result, channels);
    Ok(())
}

/// Builds the complete backchannel choice from the validated dialog scalar.
fn backchannel_policy(milliseconds: f32) -> Result<BackchannelPolicy, String> {
    match milliseconds > 0.0 {
        true => Ok(BackchannelPolicy::MarkShorterThan(
            BackchannelDuration::new(milliseconds / 1_000.0)
                .map_err(|error| format!("--backchannel-max-ms: {error}"))?,
        )),
        false => Ok(BackchannelPolicy::Disabled),
    }
}

#[cfg(test)]
mod tests {
    use super::backchannel_policy;
    use gigaam_transcription::BackchannelPolicy;

    #[test]
    fn zero_backchannel_limit_preserves_disabled_dialog_policy() {
        assert_eq!(backchannel_policy(0.0), Ok(BackchannelPolicy::Disabled));
    }
}
