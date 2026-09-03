// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Exhaustive dispatch from a validated invocation to one offline command adapter.

mod bench;
mod dialog;
mod stream;
mod transcribe;
mod vad;

use crate::grammar::Invocation;
use crate::projection::CliFailure;

/// Executes exactly one already validated CLI command.
pub(crate) fn dispatch(invocation: Invocation) -> Result<(), CliFailure> {
    let result = match invocation {
        Invocation::Transcribe(invocation) => transcribe::run(invocation),
        Invocation::Bench(invocation) => bench::run(invocation),
        Invocation::Stream(invocation) => stream::run(invocation),
        Invocation::Vad(invocation) => vad::run(invocation),
        Invocation::Dialog(invocation) => dialog::run(invocation),
    };
    result.map_err(CliFailure::Command)
}
