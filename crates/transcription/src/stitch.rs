// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Time-window construction and overlap stitching.

use crate::contracts::TranscriptWord;
use crate::text::normalize_word;
use gigaam_primitives::{trunc_f32_to_usize, usize_to_f32};

/// A validated half-open time window in seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Window {
    start: f32,
    end: f32,
}

impl Window {
    /// Constructs a finite, nonnegative, ordered time window.
    pub fn new(start: f32, end: f32) -> Result<Self, String> {
        validate_time(start, "window start")?;
        validate_time(end, "window end")?;
        if end < start {
            return Err("window end must not precede its start".into());
        }
        Ok(Self { start, end })
    }

    pub const fn start(self) -> f32 {
        self.start
    }

    pub const fn end(self) -> f32 {
        self.end
    }

    /// Returns the nonnegative duration of the window.
    pub const fn duration(self) -> f32 {
        self.end - self.start
    }
}

/// One decoded window whose words use absolute application timestamps.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowWords {
    window: Window,
    words: Vec<TranscriptWord>,
}

impl WindowWords {
    /// Associates a validated decoded word line with its decoded time window.
    pub fn new(window: Window, words: Vec<TranscriptWord>) -> Self {
        Self { window, words }
    }

    pub const fn window(&self) -> Window {
        self.window
    }

    pub const fn start(&self) -> f32 {
        self.window.start()
    }

    pub const fn end(&self) -> f32 {
        self.window.end()
    }

    pub fn words(&self) -> &[TranscriptWord] {
        &self.words
    }

    pub fn into_words(self) -> Vec<TranscriptWord> {
        self.words
    }
}

/// The pair of cuts that removes a duplicated overlap between consecutive word windows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Seam {
    previous_count: usize,
    current_skip: usize,
}

impl Seam {
    pub const fn previous_count(self) -> usize {
        self.previous_count
    }

    pub const fn current_skip(self) -> usize {
        self.current_skip
    }
}

/// Produces overlapping windows that cover a finite nonnegative duration.
pub fn windows(
    total_seconds: f32,
    window_seconds: f32,
    overlap_seconds: f32,
) -> Result<Vec<Window>, String> {
    windows_bucketed(
        total_seconds,
        window_seconds,
        overlap_seconds,
        window_seconds,
    )
}

/// Produces overlapping windows whose final window is widened backward to a bucket multiple.
///
/// The final window never exceeds `window_seconds`; it uses real left context rather than padding.
pub fn windows_bucketed(
    total_seconds: f32,
    window_seconds: f32,
    overlap_seconds: f32,
    bucket_seconds: f32,
) -> Result<Vec<Window>, String> {
    validate_geometry(
        total_seconds,
        window_seconds,
        overlap_seconds,
        bucket_seconds,
    )?;

    let mut result = Vec::new();
    let mut start = 0.0_f32;
    loop {
        let end = (start + window_seconds).min(total_seconds);
        if end >= total_seconds {
            let length = (total_seconds - start).max(0.0);
            let desired = ((length / bucket_seconds).ceil() * bucket_seconds).min(window_seconds);
            result.push(Window::new(
                (total_seconds - desired).max(0.0),
                total_seconds,
            )?);
            return Ok(result);
        }
        result.push(Window::new(start, end)?);
        start += window_seconds - overlap_seconds;
    }
}

/// Lists every window duration that bucketed construction can produce for shape warmup.
pub fn window_shapes(window_seconds: f32, bucket_seconds: f32) -> Result<Vec<f32>, String> {
    validate_time(window_seconds, "window duration")?;
    if window_seconds <= 0.0 {
        return Err("window duration must be positive".into());
    }
    validate_time(bucket_seconds, "window bucket duration")?;
    if bucket_seconds <= 0.0 {
        return Err("window bucket duration must be positive".into());
    }

    let count = trunc_f32_to_usize((window_seconds / bucket_seconds).round())
        .map_err(|error| format!("window bucket count: {error}"))?;
    let mut shapes: Vec<f32> = (1..=count)
        .map(|index| usize_to_f32(index) * bucket_seconds)
        .collect();
    if shapes
        .last()
        .is_none_or(|last| (*last - window_seconds).abs() > 1e-6)
    {
        shapes.push(window_seconds);
    }
    Ok(shapes)
}

/// Finds the overlap seam between the preceding word line and a current decoded window.
///
/// A matching normalized word near the overlap midpoint is preferred. If no matching pair exists,
/// the midpoint defines the two cuts.
pub fn seam(
    previous: &[TranscriptWord],
    previous_end: f32,
    current: &WindowWords,
    tolerance_seconds: f32,
) -> Result<Seam, String> {
    validate_time(previous_end, "previous window end")?;
    validate_time(tolerance_seconds, "stitch tolerance")?;

    let midpoint = (current.start() + previous_end) / 2.0;
    let mut best: Option<(f32, usize, usize)> = None;
    for (previous_index, previous_word) in previous.iter().enumerate() {
        if previous_word.start() < current.start() {
            continue;
        }
        for (current_index, current_word) in current.words().iter().enumerate() {
            if current_word.start() > previous_end {
                break;
            }
            let offsets_match =
                (previous_word.start() - current_word.start()).abs() <= tolerance_seconds;
            let text_matches =
                normalize_word(previous_word.text()) == normalize_word(current_word.text());
            if offsets_match && text_matches {
                let distance = (previous_word.start() - midpoint).abs();
                if best.is_none_or(|(best_distance, _, _)| distance < best_distance) {
                    best = Some((distance, previous_index, current_index));
                }
            }
        }
    }

    match best {
        Some((_, previous_count, current_skip)) => Ok(Seam {
            previous_count,
            current_skip,
        }),
        None => Ok(Seam {
            previous_count: previous
                .iter()
                .filter(|word| word.start() < midpoint)
                .count(),
            current_skip: current
                .words()
                .iter()
                .filter(|word| word.start() < midpoint)
                .count(),
        }),
    }
}

/// Stitches consecutive decoded windows through their overlap seams.
pub fn stitch_aligned(
    windows: &[WindowWords],
    tolerance_seconds: f32,
) -> Result<Vec<TranscriptWord>, String> {
    validate_time(tolerance_seconds, "stitch tolerance")?;
    let Some(first) = windows.first() else {
        return Ok(Vec::new());
    };

    let mut result = Vec::new();
    let mut previous = first.words().to_vec();
    let mut previous_end = first.end();
    for current in &windows[1..] {
        let seam = seam(&previous, previous_end, current, tolerance_seconds)?;
        result.extend(previous[..seam.previous_count()].iter().cloned());
        previous = current.words()[seam.current_skip()..].to_vec();
        previous_end = current.end();
    }
    result.extend(previous);
    Ok(result)
}

/// Joins a word line with one ASCII space between consecutive words.
pub fn words_to_text(words: &[TranscriptWord]) -> String {
    crate::contracts::words_to_text(words)
}

fn validate_geometry(
    total_seconds: f32,
    window_seconds: f32,
    overlap_seconds: f32,
    bucket_seconds: f32,
) -> Result<(), String> {
    validate_time(total_seconds, "input duration")?;
    validate_time(window_seconds, "window duration")?;
    validate_time(overlap_seconds, "window overlap duration")?;
    validate_time(bucket_seconds, "window bucket duration")?;
    if window_seconds <= 0.0 {
        return Err("window duration must be positive".into());
    }
    if overlap_seconds >= window_seconds {
        return Err("window overlap duration must be shorter than the window".into());
    }
    if bucket_seconds <= 0.0 {
        return Err("window bucket duration must be positive".into());
    }
    Ok(())
}

fn validate_time(value: f32, name: &str) -> Result<(), String> {
    if !value.is_finite() || value < 0.0 {
        return Err(format!("{name} must be finite and nonnegative"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        Seam, Window, WindowWords, seam, stitch_aligned, window_shapes, windows, windows_bucketed,
        words_to_text,
    };
    use crate::contracts::TranscriptWord;

    fn word(text: &str, start: f32) -> TranscriptWord {
        TranscriptWord::new(text.into(), start, start + 0.3)
            .expect("test word timestamps are finite and ordered")
    }

    fn window(start: f32, end: f32, words: Vec<TranscriptWord>) -> WindowWords {
        WindowWords::new(
            Window::new(start, end).expect("test window bounds are finite and ordered"),
            words,
        )
    }

    #[test]
    fn windows_cover_the_input_and_preserve_overlap() {
        let window_list = windows(71.25, 30.0, 6.0)
            .expect("positive window geometry produces a covering sequence");
        assert_eq!(
            window_list,
            vec![
                Window::new(0.0, 30.0).expect("valid expected window"),
                Window::new(24.0, 54.0).expect("valid expected window"),
                Window::new(41.25, 71.25).expect("valid expected window"),
            ]
        );
        assert_eq!(
            windows(10.3, 30.0, 6.0).expect("short input remains one window"),
            vec![Window::new(0.0, 10.3).expect("valid expected short window")]
        );
        assert_eq!(
            windows_bucketed(71.25, 30.0, 6.0, 1.0).expect("bucketed geometry is valid")[2],
            Window::new(47.25, 71.25).expect("valid expected bucketed window")
        );
        assert_eq!(
            window_shapes(30.0, 1.0)
                .expect("positive warmup geometry is valid")
                .len(),
            30
        );
    }

    #[test]
    fn matching_overlap_word_selects_nearest_seam() {
        let previous = window(
            0.0,
            10.0,
            vec![
                word("one", 1.0),
                word("two,", 5.0),
                word("three", 7.0),
                word("four", 9.5),
            ],
        );
        let current = window(
            4.0,
            14.0,
            vec![
                word("two", 5.02),
                word("three.", 7.01),
                word("four", 9.4),
                word("five", 12.0),
            ],
        );
        assert_eq!(
            seam(previous.words(), previous.end(), &current, 0.25)
                .expect("finite tolerance permits seam selection"),
            Seam {
                previous_count: 2,
                current_skip: 1,
            }
        );
        let output = stitch_aligned(&[previous, current], 0.25)
            .expect("finite tolerance permits aligned stitching");
        assert_eq!(words_to_text(&output), "one two, three. four five");
    }

    #[test]
    fn seam_falls_back_to_overlap_midpoint_without_matching_word() {
        let previous = window(
            0.0,
            10.0,
            vec![word("one", 1.0), word("two", 6.0), word("three", 8.0)],
        );
        let current = window(
            4.0,
            14.0,
            vec![word("five", 6.5), word("six", 7.5), word("seven", 12.0)],
        );
        assert_eq!(
            seam(previous.words(), previous.end(), &current, 0.25)
                .expect("finite tolerance permits seam selection"),
            Seam {
                previous_count: 2,
                current_skip: 1,
            }
        );
    }

    #[test]
    fn invalid_window_geometry_refuses() {
        assert!(windows(1.0, 1.0, 1.0).is_err());
        assert!(windows(1.0, f32::NAN, 0.0).is_err());
        assert!(window_shapes(1.0, 0.0).is_err());
    }
}
