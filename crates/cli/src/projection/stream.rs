// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Client patch application, event display, and stream-simulator statistics projection.

use super::precision_label;
use crate::grammar::StreamEventOutput;
use crate::numeric::{
    count_to_f32, count_to_f64, duration_to_f32, percentile, rounded_f32_to_usize,
};
use gigaam_model_package::EncoderPrecision;
use gigaam_transcription::{
    FinalReason, StreamConfig, StreamEvent, StreamLockPolicy, StreamWord, WordStability,
    normalize_word, word_count, word_edits,
};

/// Stateful client-visible projection for one stream simulator invocation.
pub(crate) struct StreamProjection {
    display_events: StreamEventOutput,
    patch_count: usize,
    append_count: usize,
    revision_count: usize,
    cosmetic_count: usize,
    final_count: usize,
    endpoint_count: usize,
    revision_depths: Vec<f32>,
    replaced_words: Vec<usize>,
    start_latencies: Vec<f32>,
    end_latencies: Vec<f32>,
    first_word_latency: Option<f32>,
    stable_changed: usize,
}

trait StreamPatchWord {
    fn text(&self) -> &str;
    fn start(&self) -> f32;
    fn end(&self) -> f32;
    fn stability(&self) -> WordStability;
}

impl StreamPatchWord for StreamWord {
    fn text(&self) -> &str {
        StreamWord::text(self)
    }

    fn start(&self) -> f32 {
        StreamWord::start(self)
    }

    fn end(&self) -> f32 {
        StreamWord::end(self)
    }

    fn stability(&self) -> WordStability {
        StreamWord::stability(self)
    }
}

impl StreamProjection {
    /// Creates one independent client patch observer for a stream invocation.
    pub(crate) fn new(display_events: StreamEventOutput) -> Self {
        Self {
            display_events,
            patch_count: 0,
            append_count: 0,
            revision_count: 0,
            cosmetic_count: 0,
            final_count: 0,
            endpoint_count: 0,
            revision_depths: Vec::new(),
            replaced_words: Vec::new(),
            start_latencies: Vec::new(),
            end_latencies: Vec::new(),
            first_word_latency: None,
            stable_changed: 0,
        }
    }

    /// Applies immutable lower-layer events to the exact client-visible word line.
    pub(crate) fn apply(
        &mut self,
        events: Vec<StreamEvent>,
        client: &mut Vec<StreamWord>,
    ) -> Result<(), String> {
        for event in events {
            if self.display_events == StreamEventOutput::Shown {
                eprintln!("{event:?}");
            }
            match event {
                StreamEvent::Words(words_event) => self.apply_words(
                    words_event.at(),
                    words_event.revise_from(),
                    words_event.words(),
                    client,
                )?,
                StreamEvent::Stable(frontier) => {
                    let prefix = client.get_mut(..frontier.upto()).ok_or_else(|| {
                        "stream stable frontier exceeds client transcript".to_owned()
                    })?;
                    for word in prefix {
                        *word = word.clone().with_stability(WordStability::Stable);
                    }
                }
                StreamEvent::Final(final_event) => {
                    let prefix = client.get_mut(..final_event.upto()).ok_or_else(|| {
                        "stream final frontier exceeds client transcript".to_owned()
                    })?;
                    for word in prefix {
                        *word = word.clone().with_stability(WordStability::Stable);
                    }
                    self.record_final(final_event.reason())?;
                }
            }
        }
        Ok(())
    }

    fn apply_words(
        &mut self,
        at: f32,
        revise_from: usize,
        words: &[StreamWord],
        client: &mut Vec<StreamWord>,
    ) -> Result<(), String> {
        self.record_words_patch(at, revise_from, client.as_slice(), words)?;
        client.truncate(revise_from);
        client.extend_from_slice(words);
        Ok(())
    }

    fn record_words_patch<Word>(
        &mut self,
        at: f32,
        revise_from: usize,
        client: &[Word],
        words: &[Word],
    ) -> Result<(), String>
    where
        Word: StreamPatchWord,
    {
        if revise_from > client.len() {
            return Err("stream words event revises beyond client transcript".into());
        }
        increment(&mut self.patch_count, "stream patch count")?;
        if revise_from < client.len() {
            let revised_tail = &client[revise_from..];
            let shared = revised_tail.len().min(words.len());
            let first_difference = (0..shared).find(|index| {
                normalize_word(revised_tail[*index].text()) != normalize_word(words[*index].text())
            });
            if revised_tail.len() == words.len() && first_difference.is_none() {
                increment(&mut self.cosmetic_count, "stream cosmetic patch count")?;
            } else {
                increment(&mut self.revision_count, "stream word-revision count")?;
                let first_difference = match first_difference {
                    Some(index) => index,
                    None => shared,
                };
                let tail_length = client
                    .len()
                    .checked_sub(revise_from)
                    .ok_or_else(|| "stream client revision tail underflows".to_owned())?;
                let tail_index = first_difference.min(
                    tail_length
                        .checked_sub(1)
                        .ok_or_else(|| "stream revision tail must be nonempty".to_owned())?,
                );
                let client_index = revise_from
                    .checked_add(tail_index)
                    .ok_or_else(|| "stream client revision index overflows".to_owned())?;
                let revised = client
                    .get(client_index)
                    .ok_or_else(|| "stream client revision index exceeds transcript".to_owned())?;
                self.revision_depths.push(at - revised.start());
                self.replaced_words.push(
                    client
                        .len()
                        .checked_sub(revise_from)
                        .and_then(|count| count.checked_sub(first_difference))
                        .ok_or_else(|| "stream replaced-word count underflows".to_owned())?,
                );
                if revised.stability() == WordStability::Stable {
                    increment(
                        &mut self.stable_changed,
                        "stream stable-word revision count",
                    )?;
                }
            }
        } else {
            increment(&mut self.append_count, "stream append patch count")?;
        }
        for (index, word) in words.iter().enumerate() {
            let destination = revise_from
                .checked_add(index)
                .ok_or_else(|| "stream word destination index overflows".to_owned())?;
            if destination >= client.len() {
                self.start_latencies.push(at - word.start());
                self.end_latencies.push(at - word.end());
                if self.first_word_latency.is_none() {
                    self.first_word_latency = Some(at - word.start());
                }
            }
        }
        Ok(())
    }

    fn record_final(&mut self, reason: FinalReason) -> Result<(), String> {
        increment(&mut self.final_count, "stream final-event count")?;
        if reason == FinalReason::Endpoint {
            increment(&mut self.endpoint_count, "stream endpoint-event count")?;
        }
        Ok(())
    }

    /// Emits the final word line and every established stream-simulator statistic.
    pub(crate) fn finish(&self, summary: StreamSummary<'_>) -> Result<(), String> {
        let text = client_text(summary.client.iter().map(StreamWord::text));
        println!("{text}");
        self.render_statistics(
            StreamStatistics {
                text: &text,
                batch_text: summary.batch_text,
                step_seconds: summary.config.step_sec(),
                horizon_seconds: summary.config.horizon_sec(),
                lock_policy: summary.config.lock_policy(),
                chunk_milliseconds: summary.chunk_milliseconds,
                precision: summary.precision,
                decoder_seconds: summary.decoder_seconds,
                encoder_seconds: summary.encoder_seconds,
                decodes: summary.decodes,
                audio_seconds: summary.audio_seconds,
                wall_seconds: summary.wall_seconds,
                reference: summary.reference,
            },
            |line| eprintln!("{line}"),
        )?;
        Ok(())
    }

    fn render_statistics(
        &self,
        summary: StreamStatistics<'_>,
        mut emit: impl FnMut(String),
    ) -> Result<(), String> {
        let batch_edits = word_edits(summary.batch_text, summary.text);
        emit(format!(
            "# stream: {:.1} s, chunk {} ms, step {:.0} ms, H={} s, lock={}, {}",
            summary.audio_seconds,
            summary.chunk_milliseconds,
            summary.step_seconds * 1_000.0,
            summary.horizon_seconds,
            summary.lock_policy == StreamLockPolicy::CommitStable,
            precision_label(summary.precision)
        ));
        let decoded = summary.decodes.max(1);
        emit(format!(
            "# decodes {} | mean {:.1} ms (encoder {:.1}) | total {:.2} s, RTF {:.4} | wall {:.2} s",
            summary.decodes,
            summary.decoder_seconds * 1_000.0 / count_to_f64(decoded),
            summary.encoder_seconds * 1_000.0 / count_to_f64(decoded),
            summary.decoder_seconds,
            duration_to_f32(summary.decoder_seconds, "stream decode time")? / summary.audio_seconds,
            summary.wall_seconds
        ));
        emit(format!(
            "# events: patches {} (appends {}, word revisions {}, including before stable word {}; cosmetic {}), final {} (endpoint {})",
            self.patch_count,
            self.append_count,
            self.revision_count,
            self.stable_changed,
            self.cosmetic_count,
            self.final_count,
            self.endpoint_count
        ));
        let batch_words = word_count(summary.batch_text);
        emit(format!(
            "# final vs batch: words {}, edits {}, WER {:.2} %",
            batch_words,
            batch_edits,
            100.0 * count_to_f32(batch_edits) / count_to_f32(batch_words.max(1))
        ));
        if let Some(reference) = summary.reference {
            let batch_reference_edits = word_edits(reference, summary.batch_text);
            let stream_reference_edits = word_edits(reference, summary.text);
            let reference_words = word_count(reference).max(1);
            emit(format!(
                "# vs reference: batch WER {:.2} % ({}/{}), stream WER {:.2} % ({}/{})",
                100.0 * count_to_f32(batch_reference_edits) / count_to_f32(reference_words),
                batch_reference_edits,
                reference_words,
                100.0 * count_to_f32(stream_reference_edits) / count_to_f32(reference_words),
                stream_reference_edits,
                reference_words
            ));
        }
        let mut start_latencies = self.start_latencies.clone();
        let mut start_latencies_ninety = self.start_latencies.clone();
        let mut end_latencies = self.end_latencies.clone();
        let mut end_latencies_ninety = self.end_latencies.clone();
        let first_word_latency = self.first_word_latency.iter().copied().sum::<f32>();
        emit(format!(
            "# word-appearance latency: from start p50 {:.2} p90 {:.2} s | from end p50 {:.2} p90 {:.2} s | first word {:.2} s after its start",
            percentile(&mut start_latencies, 0.5)?,
            percentile(&mut start_latencies_ninety, 0.9)?,
            percentile(&mut end_latencies, 0.5)?,
            percentile(&mut end_latencies_ninety, 0.9)?,
            first_word_latency
        ));
        let mut depths = self.revision_depths.clone();
        let mut depths_ninety = self.revision_depths.clone();
        let median_depth = percentile(&mut depths, 0.5)?;
        let ninety_depth = percentile(&mut depths_ninety, 0.9)?;
        let revision_depth_maximum = match depths.last() {
            Some(value) => *value,
            None => 0.0,
        };
        let share = |threshold: f32| {
            100.0
                * count_to_f32(
                    self.revision_depths
                        .iter()
                        .filter(|value| **value > threshold)
                        .count(),
                )
                / count_to_f32(self.revision_depths.len().max(1))
        };
        let mut replaced: Vec<f32> = self
            .replaced_words
            .iter()
            .map(|value| count_to_f32(*value))
            .collect();
        let median_replaced = rounded_f32_to_usize(
            percentile(&mut replaced, 0.5)?,
            "words-per-patch percentile",
        )?;
        let maximum_replaced = match self.replaced_words.iter().max() {
            Some(value) => *value,
            None => 0,
        };
        emit(format!(
            "# revision depth (at − start of revised word): p50 {:.2} p90 {:.2} max {:.2} s | share >1 s {:.0} %, >2 s {:.0} %, >3 s {:.0} %, >4 s {:.0} %, >5 s {:.0} %, >6 s {:.0} % | words per patch p50 {} max {}",
            median_depth,
            ninety_depth,
            revision_depth_maximum,
            share(1.0),
            share(2.0),
            share(3.0),
            share(4.0),
            share(5.0),
            share(6.0),
            median_replaced,
            maximum_replaced
        ));
        Ok(())
    }
}

struct StreamStatistics<'a> {
    text: &'a str,
    batch_text: &'a str,
    step_seconds: f32,
    horizon_seconds: f32,
    lock_policy: StreamLockPolicy,
    chunk_milliseconds: usize,
    precision: EncoderPrecision,
    decoder_seconds: f64,
    encoder_seconds: f64,
    decodes: usize,
    audio_seconds: f32,
    wall_seconds: f32,
    reference: Option<&'a str>,
}

/// Immutable stream execution values that become visible only in final projection.
pub(crate) struct StreamSummary<'a> {
    pub(crate) client: &'a [StreamWord],
    pub(crate) batch_text: &'a str,
    pub(crate) config: &'a StreamConfig,
    pub(crate) chunk_milliseconds: usize,
    pub(crate) precision: EncoderPrecision,
    pub(crate) decoder_seconds: f64,
    pub(crate) encoder_seconds: f64,
    pub(crate) decodes: usize,
    pub(crate) audio_seconds: f32,
    pub(crate) wall_seconds: f32,
    pub(crate) reference: Option<&'a str>,
}

fn increment(value: &mut usize, context: &str) -> Result<(), String> {
    *value = value
        .checked_add(1)
        .ok_or_else(|| format!("{context} overflows usize"))?;
    Ok(())
}

fn client_text<'a>(words: impl Iterator<Item = &'a str>) -> String {
    words.collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::{StreamPatchWord, StreamProjection, StreamStatistics, client_text};
    use crate::grammar::StreamEventOutput;
    use gigaam_model_package::EncoderPrecision;
    use gigaam_transcription::{FinalReason, StreamLockPolicy, WordStability};

    #[derive(Clone, Copy)]
    struct WordFact {
        text: &'static str,
        start: f32,
        end: f32,
        stability: WordStability,
        readable: bool,
    }

    impl WordFact {
        fn assert_readable(&self) {
            assert!(
                self.readable,
                "append patch must not inspect prior client words"
            );
        }
    }

    impl StreamPatchWord for WordFact {
        fn text(&self) -> &str {
            self.assert_readable();
            self.text
        }

        fn start(&self) -> f32 {
            self.assert_readable();
            self.start
        }

        fn end(&self) -> f32 {
            self.assert_readable();
            self.end
        }

        fn stability(&self) -> WordStability {
            self.assert_readable();
            self.stability
        }
    }

    fn word(text: &'static str, start: f32, end: f32, stability: WordStability) -> WordFact {
        WordFact {
            text,
            start,
            end,
            stability,
            readable: true,
        }
    }

    fn unread_prior_word() -> WordFact {
        WordFact {
            text: "",
            start: 0.0,
            end: 0.0,
            stability: WordStability::Stable,
            readable: false,
        }
    }

    #[test]
    fn append_patch_does_not_inspect_prior_client_words() -> Result<(), String> {
        let mut projection = StreamProjection::new(StreamEventOutput::Hidden);
        let prior = [unread_prior_word(); 4_096];
        let incoming = [word("next", 2.0, 2.25, WordStability::Revisable)];

        projection.record_words_patch(2.5, prior.len(), &prior, &incoming)?;

        assert_eq!(projection.patch_count, 1);
        assert_eq!(projection.append_count, 1);
        assert_eq!(projection.start_latencies, vec![0.5]);
        assert_eq!(projection.end_latencies, vec![0.25]);
        assert_eq!(projection.first_word_latency, Some(0.5));
        Ok(())
    }

    #[test]
    fn stream_patch_statistics_preserve_the_client_visible_output() -> Result<(), String> {
        let mut projection = StreamProjection::new(StreamEventOutput::Hidden);
        let empty: [WordFact; 0] = [];
        let first = [word("Hello", 0.0, 0.4, WordStability::Revisable)];
        projection.record_words_patch(0.5, 0, &empty, &first)?;

        let client_after_first = [word("Hello", 0.0, 0.4, WordStability::Stable)];
        let second = [word("world", 0.5, 1.0, WordStability::Revisable)];
        projection.record_words_patch(1.2, 1, &client_after_first, &second)?;

        let client_after_second = [
            word("Hello", 0.0, 0.4, WordStability::Stable),
            word("world", 0.5, 1.0, WordStability::Revisable),
        ];
        let replacement = [word("there", 0.5, 1.0, WordStability::Revisable)];
        projection.record_words_patch(2.0, 1, &client_after_second, &replacement)?;

        let client_after_revision = [
            word("Hello", 0.0, 0.4, WordStability::Stable),
            word("there", 0.5, 1.0, WordStability::Revisable),
        ];
        let cosmetic = [
            word("hello", 0.0, 0.4, WordStability::Stable),
            word("there", 0.5, 1.0, WordStability::Revisable),
        ];
        projection.record_words_patch(2.5, 0, &client_after_revision, &cosmetic)?;

        let stable_revision = [
            word("hi", 0.0, 0.4, WordStability::Revisable),
            word("there", 0.5, 1.0, WordStability::Revisable),
        ];
        projection.record_words_patch(3.0, 0, &cosmetic, &stable_revision)?;
        assert_eq!(projection.stable_changed, 1);
        projection.record_final(FinalReason::Endpoint)?;

        let text = client_text(stable_revision.iter().map(|word| word.text));
        assert_eq!(text, "hi there");
        let mut lines = Vec::new();
        projection.render_statistics(
            StreamStatistics {
                text: &text,
                batch_text: "hi world",
                step_seconds: 0.5,
                horizon_seconds: 4.0,
                lock_policy: StreamLockPolicy::CommitStable,
                chunk_milliseconds: 100,
                precision: EncoderPrecision::Fp16Io32,
                decoder_seconds: 0.4,
                encoder_seconds: 0.2,
                decodes: 2,
                audio_seconds: 2.0,
                wall_seconds: 0.7,
                reference: Some("hi there"),
            },
            |line| lines.push(line),
        )?;
        assert_eq!(
            lines,
            vec![
                String::from("# stream: 2.0 s, chunk 100 ms, step 500 ms, H=4 s, lock=true, fp16"),
                String::from(
                    "# decodes 2 | mean 200.0 ms (encoder 100.0) | total 0.40 s, RTF 0.2000 | wall 0.70 s"
                ),
                String::from(
                    "# events: patches 5 (appends 2, word revisions 2, including before stable word 1; cosmetic 1), final 1 (endpoint 1)"
                ),
                String::from("# final vs batch: words 2, edits 1, WER 50.00 %"),
                String::from("# vs reference: batch WER 50.00 % (1/2), stream WER 0.00 % (0/2)"),
                String::from(
                    "# word-appearance latency: from start p50 0.70 p90 0.70 s | from end p50 0.20 p90 0.20 s | first word 0.50 s after its start"
                ),
                String::from(
                    "# revision depth (at − start of revised word): p50 3.00 p90 3.00 max 3.00 s | share >1 s 100 %, >2 s 50 %, >3 s 0 %, >4 s 0 %, >5 s 0 %, >6 s 0 % | words per patch p50 2 max 2"
                ),
            ]
        );
        Ok(())
    }
}
