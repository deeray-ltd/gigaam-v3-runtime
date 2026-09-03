// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Binary-only offline command-line entrypoint for GigaAM v3 Runtime.

mod commands;
mod composition;
mod configuration;
mod grammar;
mod numeric;
mod projection;

/// Acquires OS arguments, parses one invocation, dispatches it, and projects one terminal result.
fn main() {
    let result = grammar::parse(std::env::args_os().skip(1))
        .map_err(projection::CliFailure::from)
        .and_then(commands::dispatch);
    projection::terminal(result);
}
