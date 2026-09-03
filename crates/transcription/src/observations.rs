// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Process-independent per-window observation contracts.

use std::sync::Arc;

/// Immutable observation emitted after a successful recognition window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowTiming {
    offset_sec: f32,
    frames: usize,
    encoder_seconds: f64,
}

impl WindowTiming {
    pub(crate) fn new(
        offset_sec: f32,
        frames: usize,
        encoder_seconds: f64,
    ) -> Result<Self, String> {
        if !offset_sec.is_finite() || offset_sec < 0.0 {
            return Err("window observation offset must be finite and nonnegative".into());
        }
        if !encoder_seconds.is_finite() || encoder_seconds < 0.0 {
            return Err(
                "window observation encoder duration must be finite and nonnegative".into(),
            );
        }
        Ok(Self {
            offset_sec,
            frames,
            encoder_seconds,
        })
    }

    pub const fn offset_sec(&self) -> f32 {
        self.offset_sec
    }

    pub const fn frames(&self) -> usize {
        self.frames
    }

    pub const fn encoder_seconds(&self) -> f64 {
        self.encoder_seconds
    }
}

/// A process adapter for successful-window observations.
pub trait WindowTimingObserver: Send + Sync {
    fn observe(&self, observation: WindowTiming);
}

/// A typed enabled/disabled observation mode selected by a process adapter.
#[derive(Clone)]
pub enum ObservationMode {
    Disabled,
    Enabled(Arc<dyn WindowTimingObserver>),
}

impl ObservationMode {
    pub fn disabled() -> Self {
        Self::Disabled
    }

    pub fn enabled(observer: Arc<dyn WindowTimingObserver>) -> Self {
        Self::Enabled(observer)
    }

    pub(crate) fn emit(&self, observation: WindowTiming) {
        match self {
            Self::Disabled => {}
            Self::Enabled(observer) => observer.observe(observation),
        }
    }
}
