// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Frontend microbenchmark. The process adapter reads `ASR_FRONTEND` once and passes the typed
//! mode to Audio; the library itself never reads process configuration.

use gigaam_audio::{FrontendMode, FrontendProcessor};
use gigaam_model_package::ModelPackage;
use gigaam_primitives::{usize_to_f32, usize_to_f64};
use std::fmt::Display;
use std::path::Path;
use std::time::Instant;

const BENCH_SECONDS: usize = 30;
const BENCH_SAMPLE_RATE: usize = 16_000;
const BENCH_ITERATIONS: usize = 40;

fn exit_on_error<T, E: Display>(result: Result<T, E>, context: &str) -> T {
    match result {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{context}: {error}");
            std::process::exit(1);
        }
    }
}

fn frontend_mode_from_process() -> Result<FrontendMode, String> {
    match std::env::var_os("ASR_FRONTEND") {
        None => Ok(FrontendMode::Scalar),
        Some(value) => {
            let value = value
                .into_string()
                .map_err(|_| "ASR_FRONTEND must contain UTF-8 text".to_owned())?;
            if value.is_empty() {
                return Err("ASR_FRONTEND must not be empty when present".into());
            }
            FrontendMode::parse(&value).map_err(|error| format!("ASR_FRONTEND: {error}"))
        }
    }
}

fn main() {
    let model = match std::env::args().nth(1) {
        Some(value) => value,
        None => "model".to_owned(),
    };
    let pack = exit_on_error(
        ModelPackage::open(Path::new(&model)),
        "benchmark model pack",
    );
    let weights = exit_on_error(pack.frontend_weights(), "benchmark frontend weights");
    let frontend = exit_on_error(
        FrontendProcessor::new(
            pack.frontend(),
            weights,
            exit_on_error(frontend_mode_from_process(), "benchmark frontend mode"),
        ),
        "benchmark frontend",
    );
    let frames = BENCH_SECONDS
        .checked_mul(BENCH_SAMPLE_RATE)
        .expect("fixed benchmark constants must fit usize");
    let waveform: Vec<f32> = (0..frames)
        .map(|index| (usize_to_f32((index * 2_654_435_761_usize) % 20_011) / 10_005.0) - 1.0)
        .collect();
    let warmup = exit_on_error(frontend.log_mel(&waveform), "benchmark frontend warmup");
    let mut best = f64::INFINITY;
    let mut sum = 0.0_f64;
    for _ in 0..BENCH_ITERATIONS {
        let started = Instant::now();
        let features = exit_on_error(frontend.log_mel(&waveform), "benchmark frontend run");
        let elapsed = started.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(features.values());
        best = best.min(elapsed);
        sum += elapsed;
    }
    eprintln!(
        "mode={:?} frames={} best={:.2}ms mean={:.2}ms (n={})",
        frontend.mode(),
        warmup.frames(),
        best,
        sum / usize_to_f64(BENCH_ITERATIONS),
        BENCH_ITERATIONS
    );
}
