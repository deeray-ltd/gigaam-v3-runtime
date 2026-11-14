// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Bounded time/symbol-layer RNN-T prefix search.

use crate::rnnt::RnntFrameSource;
use crate::{ExecutionControl, ScoreDetail, Token};
use std::cmp::Ordering;

/// Beam transitions own predictor states independently of the native sessions executing them.
/// For a given complete token prefix, predictor state and distributions must be equivalent
/// regardless of the alignment that produced it; only then may search merge that prefix.
pub trait RnntBeamTransition<T = f32> {
    type State;
    fn start(&mut self, blank: usize) -> Result<Self::State, String>;
    fn prepare_frame(&mut self, frame: &[T]) -> Result<(), String>;
    /// Returns finite logits for the entire vocabulary, including blank.
    fn distribution(&mut self, frame: &[T], state: &Self::State) -> Result<Vec<f32>, String>;
    /// Returns an owned successor without changing its parent.
    fn advance(&mut self, token: usize, state: &Self::State) -> Result<Self::State, String>;
}

/// Numeric encoder values supported by the beam search.
pub trait RnntBeamFrameValue {
    fn is_finite_value(&self) -> bool;
}
impl RnntBeamFrameValue for f32 {
    fn is_finite_value(&self) -> bool {
        self.is_finite()
    }
}
impl RnntBeamFrameValue for half::f16 {
    fn is_finite_value(&self) -> bool {
        self.is_finite()
    }
}

/// Explicit beam and model dimensions.
#[derive(Clone, Copy, Debug)]
pub struct RnntBeamOptions {
    pub vocabulary: usize,
    pub blank: usize,
    pub width: usize,
    pub max_symbols_per_step: usize,
    pub detail: ScoreDetail,
}

/// Representative token occurrences and per-frame token absence.
#[derive(Debug, PartialEq)]
pub struct RnntBeamOutput {
    tokens: Vec<Token>,
    silence: Vec<bool>,
    #[cfg(test)]
    mass: f64,
}
impl RnntBeamOutput {
    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }
    pub fn silence(&self) -> &[bool] {
        &self.silence
    }
    /// Consumes the search result without copying its token or silence buffers.
    pub fn into_parts(self) -> (Vec<Token>, Vec<bool>) {
        (self.tokens, self.silence)
    }
}

#[derive(Clone)]
struct Occurrence {
    frame: usize,
    token: usize,
    score: f64,
}
struct History {
    previous: Option<usize>,
    occurrence: Occurrence,
    len: usize,
}
fn length(history: Option<usize>, arena: &[History]) -> usize {
    history.map_or(0, |index| arena[index].len)
}
// Reused initialized buffers: comparisons never allocate and only inspect live prefixes.
#[derive(Default)]
struct ComparisonBuffers {
    left: Vec<(usize, usize)>,
    right: Vec<(usize, usize)>,
    #[cfg(test)]
    visits: usize,
}
impl ComparisonBuffers {
    fn prepare(&mut self, needed: usize) -> Result<(), String> {
        for buffer in [&mut self.left, &mut self.right] {
            if buffer.len() < needed {
                buffer
                    .try_reserve_exact(needed - buffer.len())
                    .map_err(|_| "RNN-T comparison allocation failed")?;
                buffer.resize(needed, (0, 0));
            }
        }
        Ok(())
    }

    fn history_cmp(
        &mut self,
        a: Option<usize>,
        b: Option<usize>,
        arena: &[History],
        frames: bool,
    ) -> Ordering {
        if a == b {
            return Ordering::Equal;
        }
        let al = length(a, arena);
        let bl = length(b, arena);
        let visits = fill_history(a, arena, &mut self.left[..al])
            + fill_history(b, arena, &mut self.right[..bl]);
        #[cfg(test)]
        {
            self.visits += visits;
        }
        #[cfg(not(test))]
        let _ = visits;
        if frames {
            self.left[..al].cmp(&self.right[..bl])
        } else {
            self.left[..al]
                .iter()
                .map(|key| key.1)
                .cmp(self.right[..bl].iter().map(|key| key.1))
        }
    }

    fn candidate_cmp(
        &mut self,
        a: &Candidate,
        b: &Candidate,
        active: &[Hypothesis],
        arena: &[History],
    ) -> Ordering {
        if a.parent == b.parent {
            return a.token.cmp(&b.token);
        }
        let ah = active[a.parent].history;
        let bh = active[b.parent].history;
        let al = length(ah, arena);
        let bl = length(bh, arena);
        let visits = fill_history(ah, arena, &mut self.left[..al])
            + fill_history(bh, arena, &mut self.right[..bl]);
        #[cfg(test)]
        {
            self.visits += visits;
        }
        #[cfg(not(test))]
        let _ = visits;
        self.left[al] = (0, a.token);
        self.right[bl] = (0, b.token);
        self.left[..=al]
            .iter()
            .map(|key| key.1)
            .cmp(self.right[..=bl].iter().map(|key| key.1))
    }
}

// Each predecessor is visited once. Returning the visit count lets semantic tests check that
// bound without depending on wall time or the private representation of a sorting algorithm.
fn fill_history(
    mut history: Option<usize>,
    arena: &[History],
    output: &mut [(usize, usize)],
) -> usize {
    let mut visits = 0;
    while let Some(index) = history {
        let node = &arena[index];
        output[node.len - 1] = (node.occurrence.frame, node.occurrence.token);
        history = node.previous;
        visits += 1;
    }
    visits
}
struct Hypothesis {
    state: usize,
    mass: f64,
    path: f64,
    history: Option<usize>,
}
struct Candidate {
    parent: usize,
    token: usize,
    mass: f64,
    path: f64,
    score: f64,
}
fn finite_add(a: f64, b: f64) -> Result<f64, String> {
    let value = a + b;
    if value.is_finite() {
        Ok(value)
    } else {
        Err("RNN-T beam score is not finite".into())
    }
}
fn log_add(a: f64, b: f64) -> Result<f64, String> {
    let hi = a.max(b);
    let value = hi + (a.min(b) - hi).exp().ln_1p();
    if value.is_finite() {
        Ok(value)
    } else {
        Err("RNN-T beam mass is not finite".into())
    }
}
fn row(logits: Vec<f32>, vocabulary: usize) -> Result<Vec<f64>, String> {
    if logits.len() != vocabulary || logits.iter().any(|v| !v.is_finite()) {
        return Err("invalid RNN-T joint row".into());
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits
        .iter()
        .map(|v| (f64::from(*v) - f64::from(maximum)).exp())
        .sum();
    let normalizer = sum.ln();
    let mut result = Vec::new();
    result
        .try_reserve_exact(vocabulary)
        .map_err(|_| "RNN-T row allocation failed")?;
    for value in logits {
        result.push(f64::from(value) - f64::from(maximum) - normalizer);
    }
    if result.iter().any(|v| !v.is_finite()) {
        return Err("invalid normalized RNN-T row".into());
    }
    Ok(result)
}
fn release_states<S>(states: &mut [Option<S>], active: &[Hypothesis], done: &[Hypothesis]) {
    for (index, state) in states.iter_mut().enumerate() {
        if !active.iter().chain(done).any(|h| h.state == index) {
            *state = None;
        }
    }
}
fn insert_state<S>(states: &mut Vec<Option<S>>, state: S) -> Result<usize, String> {
    if let Some(index) = states.iter().position(Option::is_none) {
        states[index] = Some(state);
        return Ok(index);
    }
    states
        .try_reserve(1)
        .map_err(|_| "RNN-T state allocation failed")?;
    let index = states.len();
    states.push(Some(state));
    Ok(index)
}

/// Searches the frame source, retaining at most the selected width at each layer.
pub fn decode<T: RnntBeamFrameValue, R: RnntBeamTransition<T>, F: RnntFrameSource<T>>(
    transition: &mut R,
    frames: &F,
    options: RnntBeamOptions,
    control: &ExecutionControl,
) -> Result<RnntBeamOutput, String> {
    let RnntBeamOptions {
        vocabulary,
        blank,
        width,
        max_symbols_per_step: cap,
        detail,
    } = options;
    let count = frames.output_frames();
    if vocabulary == 0 || blank >= vocabulary || width == 0 {
        return Err("invalid RNN-T beam options".into());
    }
    let scratch = width
        .checked_mul(vocabulary)
        .ok_or("RNN-T candidates overflow")?;
    let completed = cap
        .checked_add(1)
        .and_then(|n| n.checked_mul(width))
        .ok_or("RNN-T completed candidates overflow")?;
    count
        .checked_mul(cap)
        .and_then(|n| n.checked_mul(width))
        .and_then(|n| n.checked_add(1))
        .ok_or("RNN-T history or prediction count overflow")?;
    count
        .checked_mul(completed)
        .ok_or("RNN-T joint call count overflow")?;
    let mut silence = Vec::new();
    silence
        .try_reserve_exact(count)
        .map_err(|_| "RNN-T silence allocation failed")?;
    if count == 0 {
        control.checkpoint()?;
        return Ok(RnntBeamOutput {
            tokens: Vec::new(),
            silence,
            #[cfg(test)]
            mass: 0.0,
        });
    }
    control.checkpoint()?;
    let state = transition.start(blank)?;
    control.checkpoint()?;
    let mut states = Vec::new();
    let initial = insert_state(&mut states, state)?;
    let mut arena: Vec<History> = Vec::new();
    let mut comparisons = ComparisonBuffers::default();
    let mut active = Vec::new();
    active
        .try_reserve_exact(width)
        .map_err(|_| "RNN-T active allocation failed")?;
    active.push(Hypothesis {
        state: initial,
        mass: 0.0,
        path: 0.0,
        history: None,
    });
    for frame_index in 0..count {
        control.checkpoint()?;
        let frame = frames.frame(frame_index);
        if frame.iter().any(|value| !value.is_finite_value()) {
            return Err("invalid RNN-T encoder frame".into());
        }
        transition.prepare_frame(frame)?;
        control.checkpoint()?;
        let mut done: Vec<Hypothesis> = Vec::new();
        done.try_reserve_exact(completed)
            .map_err(|_| "RNN-T completed allocation failed")?;
        for layer in 0..=cap {
            let max_history = active
                .iter()
                .chain(&done)
                .map(|h| length(h.history, &arena))
                .fold(0, usize::max);
            comparisons.prepare(max_history)?;
            control.checkpoint()?;
            let mut candidates = Vec::new();
            candidates
                .try_reserve_exact(scratch)
                .map_err(|_| "RNN-T candidate allocation failed")?;
            for (parent, hypothesis) in active.iter().enumerate() {
                control.checkpoint()?;
                let logits = transition.distribution(
                    frame,
                    states[hypothesis.state]
                        .as_ref()
                        .ok_or("missing RNN-T state")?,
                )?;
                control.checkpoint()?;
                let probabilities = row(logits, vocabulary)?;
                let mass = finite_add(hypothesis.mass, probabilities[blank])?;
                let path = finite_add(hypothesis.path, probabilities[blank])?;
                if let Some(existing) = done.iter_mut().find(|x| {
                    comparisons.history_cmp(x.history, hypothesis.history, &arena, false)
                        == Ordering::Equal
                }) {
                    existing.mass = log_add(existing.mass, mass)?;
                    if path > existing.path
                        || (path == existing.path
                            && comparisons.history_cmp(
                                hypothesis.history,
                                existing.history,
                                &arena,
                                true,
                            ) == Ordering::Less)
                    {
                        existing.path = path;
                        existing.history = hypothesis.history;
                        existing.state = hypothesis.state;
                    }
                } else {
                    done.push(Hypothesis {
                        state: hypothesis.state,
                        mass,
                        path,
                        history: hypothesis.history,
                    });
                }
                if layer < cap {
                    for (token, score) in probabilities.iter().copied().enumerate() {
                        control.checkpoint()?;
                        if token != blank {
                            candidates.push(Candidate {
                                parent,
                                token,
                                mass: finite_add(hypothesis.mass, score)?,
                                path: finite_add(hypothesis.path, score)?,
                                score,
                            });
                        }
                    }
                }
            }
            if layer == cap || candidates.is_empty() {
                break;
            }
            // Active full prefixes are unique. Appending one nonblank label per parent
            // creates unique children: equal children imply equal parent prefix and label.
            // Completed hypotheses still merge above, where distinct alignments can meet.
            let max_child = active
                .iter()
                .map(|h| length(h.history, &arena))
                .fold(0, usize::max)
                .checked_add(1)
                .ok_or("RNN-T comparison length overflow")?;
            comparisons.prepare(max_child)?;
            control.checkpoint()?;
            candidates.sort_unstable_by(|a, b| {
                b.mass
                    .total_cmp(&a.mass)
                    .then_with(|| comparisons.candidate_cmp(a, b, &active, &arena))
            });
            candidates.truncate(width);
            control.checkpoint()?;
            let mut next = Vec::new();
            next.try_reserve_exact(candidates.len())
                .map_err(|_| "RNN-T state allocation failed")?;
            for candidate in candidates {
                control.checkpoint()?;
                let parent = &active[candidate.parent];
                let state = transition.advance(
                    candidate.token,
                    states[parent.state].as_ref().ok_or("missing RNN-T state")?,
                )?;
                control.checkpoint()?;
                let len = length(parent.history, &arena) + 1;
                arena
                    .try_reserve(1)
                    .map_err(|_| "RNN-T history allocation failed")?;
                let history = arena.len();
                arena.push(History {
                    previous: parent.history,
                    occurrence: Occurrence {
                        frame: frame_index,
                        token: candidate.token,
                        score: candidate.score,
                    },
                    len,
                });
                let state = insert_state(&mut states, state)?;
                next.push(Hypothesis {
                    state,
                    mass: candidate.mass,
                    path: candidate.path,
                    history: Some(history),
                });
            }
            active = next;
            release_states(&mut states, &active, &done);
        }
        let max_done = done
            .iter()
            .map(|h| length(h.history, &arena))
            .fold(0, usize::max);
        comparisons.prepare(max_done)?;
        control.checkpoint()?;
        done.sort_unstable_by(|a, b| {
            b.mass
                .total_cmp(&a.mass)
                .then_with(|| comparisons.history_cmp(a.history, b.history, &arena, false))
        });
        control.checkpoint()?;
        done.truncate(width);
        active = done;
        release_states(&mut states, &active, &[]);
    }
    control.checkpoint()?;
    let winner = active
        .first()
        .ok_or("RNN-T beam has no completed hypothesis")?;
    let len = length(winner.history, &arena);
    let mut occurrences = Vec::new();
    occurrences
        .try_reserve_exact(len)
        .map_err(|_| "RNN-T output allocation failed")?;
    let mut node = winner.history;
    while let Some(current) = node {
        occurrences.push(arena[current].occurrence.clone());
        node = arena[current].previous;
    }
    occurrences.reverse();
    silence.resize(count, true);
    let mut tokens = Vec::new();
    tokens
        .try_reserve_exact(len)
        .map_err(|_| "RNN-T token allocation failed")?;
    for occurrence in occurrences {
        control.checkpoint()?;
        silence[occurrence.frame] = false;
        tokens.push(match detail {
            ScoreDetail::None => Token::new(occurrence.token, occurrence.frame),
            ScoreDetail::TokenLogProbabilities => {
                Token::scored(occurrence.token, occurrence.frame, occurrence.score)?
            }
        });
    }
    control.checkpoint()?;
    Ok(RnntBeamOutput {
        tokens,
        silence,
        #[cfg(test)]
        mass: winner.mass,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    struct Frames(Vec<Vec<f32>>);
    impl RnntFrameSource for Frames {
        fn output_frames(&self) -> usize {
            self.0.len()
        }
        fn frame(&self, index: usize) -> &[f32] {
            &self.0[index]
        }
    }
    struct Fake {
        rows: Vec<Vec<f32>>,
        calls: usize,
        advances: usize,
        cancel: Option<ExecutionControl>,
    }
    impl RnntBeamTransition for Fake {
        type State = Vec<usize>;
        fn start(&mut self, _: usize) -> Result<Self::State, String> {
            Ok(Vec::new())
        }
        fn prepare_frame(&mut self, _: &[f32]) -> Result<(), String> {
            Ok(())
        }
        fn distribution(&mut self, frame: &[f32], state: &Self::State) -> Result<Vec<f32>, String> {
            self.calls += 1;
            if let Some(control) = &self.cancel {
                control.request_cancellation();
            }
            let frame_index = if frame[0] == 0.0 { 0 } else { 1 };
            let index = row_index(frame_index, state, self.rows.len());
            Ok(self.rows[index].clone())
        }
        fn advance(&mut self, token: usize, state: &Self::State) -> Result<Self::State, String> {
            self.advances += 1;
            let mut next = state.clone();
            next.push(token);
            Ok(next)
        }
    }
    fn row_index(frame: usize, prefix: &[usize], count: usize) -> usize {
        prefix.iter().fold(frame % count, |value, token| {
            (value * 3 + token + 1) % count
        })
    }
    fn options(width: usize, cap: usize) -> RnntBeamOptions {
        RnntBeamOptions {
            vocabulary: 3,
            blank: 2,
            width,
            max_symbols_per_step: cap,
            detail: ScoreDetail::TokenLogProbabilities,
        }
    }
    fn fake() -> Fake {
        Fake {
            rows: vec![vec![1.0, 0.2, 0.5], vec![0.1, 1.0, 0.7]],
            calls: 0,
            advances: 0,
            cancel: None,
        }
    }
    type Alignment = Vec<(usize, usize)>;
    #[derive(Clone)]
    struct Reference {
        probability: f64,
        best: f64,
        alignment: Alignment,
    }
    fn merge(
        map: &mut BTreeMap<Vec<usize>, Reference>,
        prefix: Vec<usize>,
        probability: f64,
        best: f64,
        alignment: Alignment,
    ) {
        if let Some(old) = map.get_mut(&prefix) {
            old.probability += probability;
            if best > old.best || (best == old.best && alignment < old.alignment) {
                old.best = best;
                old.alignment = alignment;
            }
        } else {
            map.insert(
                prefix,
                Reference {
                    probability,
                    best,
                    alignment,
                },
            );
        }
    }
    fn prune(
        map: BTreeMap<Vec<usize>, Reference>,
        width: usize,
    ) -> BTreeMap<Vec<usize>, Reference> {
        let mut entries: Vec<_> = map.into_iter().collect();
        entries.sort_by(|a, b| {
            b.1.probability
                .total_cmp(&a.1.probability)
                .then_with(|| a.0.cmp(&b.0))
        });
        entries.truncate(width);
        entries.into_iter().collect()
    }
    // Independent ordinary-probability implementation of the time/symbol lattice.
    fn reference(
        rows: &[Vec<f32>],
        frames: usize,
        cap: usize,
        width: usize,
    ) -> (Vec<usize>, Reference) {
        let mut active = BTreeMap::new();
        merge(&mut active, vec![], 1.0, 1.0, vec![]);
        for frame in 0..frames {
            let mut done = BTreeMap::new();
            for layer in 0..=cap {
                let mut next = BTreeMap::new();
                for (prefix, value) in &active {
                    let logits = &rows[row_index(frame, prefix, rows.len())];
                    let weights: Vec<f64> = logits.iter().map(|v| f64::from(*v).exp()).collect();
                    let total: f64 = weights.iter().sum();
                    let blank = weights[2] / total;
                    merge(
                        &mut done,
                        prefix.clone(),
                        value.probability * blank,
                        value.best * blank,
                        value.alignment.clone(),
                    );
                    if layer < cap {
                        for (token, weight) in weights.iter().enumerate().take(2) {
                            let probability = weight / total;
                            let mut new_prefix = prefix.clone();
                            new_prefix.push(token);
                            let mut alignment = value.alignment.clone();
                            alignment.push((frame, token));
                            merge(
                                &mut next,
                                new_prefix,
                                value.probability * probability,
                                value.best * probability,
                                alignment,
                            );
                        }
                    }
                }
                if layer < cap {
                    active = prune(next, width);
                }
            }
            active = prune(done, width);
        }
        let mut best: Vec<_> = active.into_iter().collect();
        best.sort_by(|a, b| {
            b.1.probability
                .total_cmp(&a.1.probability)
                .then_with(|| a.0.cmp(&b.0))
        });
        best.remove(0)
    }
    // Enumerate individual alignments, without layer aggregation or pruning.
    fn enumerate(rows: &[Vec<f32>], frames: usize, cap: usize) -> BTreeMap<Vec<usize>, Reference> {
        struct Path {
            frame: usize,
            layer: usize,
            prefix: Vec<usize>,
            alignment: Alignment,
            probability: f64,
        }
        let mut work = vec![Path {
            frame: 0,
            layer: 0,
            prefix: vec![],
            alignment: vec![],
            probability: 1.0,
        }];
        let mut result = BTreeMap::new();
        while let Some(path) = work.pop() {
            if path.frame == frames {
                merge(
                    &mut result,
                    path.prefix,
                    path.probability,
                    path.probability,
                    path.alignment,
                );
                continue;
            }
            let logits = &rows[row_index(path.frame, &path.prefix, rows.len())];
            let weights: Vec<f64> = logits.iter().map(|v| f64::from(*v).exp()).collect();
            let total: f64 = weights.iter().sum();
            work.push(Path {
                frame: path.frame + 1,
                layer: 0,
                prefix: path.prefix.clone(),
                alignment: path.alignment.clone(),
                probability: path.probability * weights[2] / total,
            });
            if path.layer < cap {
                for (token, weight) in weights.iter().enumerate().take(2) {
                    let mut prefix = path.prefix.clone();
                    prefix.push(token);
                    let mut alignment = path.alignment.clone();
                    alignment.push((path.frame, token));
                    work.push(Path {
                        frame: path.frame,
                        layer: path.layer + 1,
                        prefix,
                        alignment,
                        probability: path.probability * weight / total,
                    });
                }
            }
        }
        result
    }
    fn append_history(
        arena: &mut Vec<History>,
        previous: Option<usize>,
        frame: usize,
        token: usize,
    ) -> Option<usize> {
        let len = length(previous, arena) + 1;
        let index = arena.len();
        arena.push(History {
            previous,
            occurrence: Occurrence {
                frame,
                token,
                score: -0.5,
            },
            len,
        });
        Some(index)
    }

    #[test]
    fn comparison_matches_flattened_oracle_and_ignores_stale_tails() -> Result<(), String> {
        let mut arena = Vec::new();
        let mut entries = vec![(None, vec![])];
        for a in 0..3 {
            let first = append_history(&mut arena, None, 2 - a, a);
            entries.push((first, vec![(2 - a, a)]));
            for b in 0..3 {
                let second = append_history(&mut arena, first, b, b);
                entries.push((second, vec![(2 - a, a), (b, b)]));
                for c in 0..3 {
                    let third = append_history(&mut arena, second, 2 - c, c);
                    entries.push((third, vec![(2 - a, a), (b, b), (2 - c, c)]));
                }
            }
        }
        let mut cmp = ComparisonBuffers::default();
        cmp.prepare(1)?;
        cmp.prepare(4)?;
        for (a, ak) in &entries {
            for (b, bk) in &entries {
                let expected_tokens = ak.iter().map(|x| x.1).cmp(bk.iter().map(|x| x.1));
                assert_eq!(cmp.history_cmp(*a, *b, &arena, false), expected_tokens);
                assert_eq!(cmp.history_cmp(*a, *b, &arena, true), ak.cmp(bk));
                let active = vec![
                    Hypothesis {
                        state: 0,
                        mass: 0.0,
                        path: 0.0,
                        history: *a,
                    },
                    Hypothesis {
                        state: 1,
                        mass: 0.0,
                        path: 0.0,
                        history: *b,
                    },
                ];
                for first in 0..3 {
                    for second in 0..3 {
                        let left = Candidate {
                            parent: 0,
                            token: first,
                            mass: 0.0,
                            path: 0.0,
                            score: 0.0,
                        };
                        let right = Candidate {
                            parent: 1,
                            token: second,
                            mass: 0.0,
                            path: 0.0,
                            score: 0.0,
                        };
                        let mut x: Vec<_> = ak.iter().map(|key| key.1).collect();
                        x.push(first);
                        let mut y: Vec<_> = bk.iter().map(|key| key.1).collect();
                        y.push(second);
                        assert_eq!(cmp.candidate_cmp(&left, &right, &active, &arena), x.cmp(&y));
                        let same_parent = Candidate {
                            parent: 0,
                            token: second,
                            mass: 0.0,
                            path: 0.0,
                            score: 0.0,
                        };
                        assert_eq!(
                            cmp.candidate_cmp(&left, &same_parent, &active, &arena),
                            first.cmp(&second)
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn unique_active_prefixes_generate_unique_children() {
        use std::collections::BTreeSet;
        let mut active = vec![vec![]];
        for length in 1..=3 {
            let previous: Vec<_> = active
                .iter()
                .filter(|v| v.len() == length - 1)
                .cloned()
                .collect();
            for prefix in previous {
                for token in 0..3 {
                    let mut child = prefix.clone();
                    child.push(token);
                    active.push(child);
                }
            }
        }
        let mut children = BTreeSet::new();
        for prefix in &active {
            for token in 0..3 {
                let mut child = prefix.clone();
                child.push(token);
                assert!(
                    children.insert(child),
                    "distinct active prefixes and labels produced equal children"
                );
            }
        }
    }

    #[test]
    fn long_common_prefix_comparison_visits_each_history_at_most_once() -> Result<(), String> {
        let mut arena = Vec::new();
        let mut common = None;
        for index in 0..4096 {
            common = append_history(&mut arena, common, index, index % 2);
        }
        let a = append_history(&mut arena, common, 4096, 0);
        let b = append_history(&mut arena, common, 4096, 1);
        let mut cmp = ComparisonBuffers::default();
        cmp.prepare(4098)?;
        assert_eq!(cmp.history_cmp(a, b, &arena, false), Ordering::Less);
        assert_eq!(cmp.visits, 8194);
        let active = vec![
            Hypothesis {
                state: 0,
                mass: 0.0,
                path: 0.0,
                history: a,
            },
            Hypothesis {
                state: 1,
                mass: 0.0,
                path: 0.0,
                history: b,
            },
        ];
        let x = Candidate {
            parent: 0,
            token: 1,
            mass: 0.0,
            path: 0.0,
            score: 0.0,
        };
        let y = Candidate {
            parent: 1,
            token: 0,
            mass: 0.0,
            path: 0.0,
            score: 0.0,
        };
        cmp.visits = 0;
        assert_eq!(cmp.candidate_cmp(&x, &y, &active, &arena), Ordering::Less);
        assert_eq!(cmp.visits, 8194);
        cmp.visits = 0;
        assert_eq!(cmp.history_cmp(a, a, &arena, true), Ordering::Equal);
        assert_eq!(cmp.visits, 0);
        Ok(())
    }

    #[test]
    fn exhaustive_individual_alignments_match_full_prefix_sensitive_search() -> Result<(), String> {
        for frame_count in 1..=2 {
            for cap in 0..=2 {
                let mut transition = fake();
                let mut oracle: Vec<_> = enumerate(&transition.rows, frame_count, cap)
                    .into_iter()
                    .collect();
                oracle.sort_by(|a, b| {
                    b.1.probability
                        .total_cmp(&a.1.probability)
                        .then_with(|| a.0.cmp(&b.0))
                });
                let (expected, representative) = &oracle[0];
                let frames = Frames(
                    (0..frame_count)
                        .map(|i| if i == 0 { vec![0.0] } else { vec![1.0] })
                        .collect(),
                );
                let output = decode(
                    &mut transition,
                    &frames,
                    options(128, cap),
                    &ExecutionControl::without_deadline(),
                )?;
                assert_eq!(
                    &output.tokens.iter().map(Token::id).collect::<Vec<_>>(),
                    expected
                );
                assert!((output.mass - representative.probability.ln()).abs() < 1e-10);
                assert_eq!(
                    output
                        .tokens
                        .iter()
                        .map(|t| (t.frame(), t.id()))
                        .collect::<Vec<_>>(),
                    representative.alignment
                );
            }
        }
        Ok(())
    }
    #[test]
    fn sibling_states_are_independent_and_prefix_sensitive() -> Result<(), String> {
        let mut transition = fake();
        let parent = transition.start(2)?;
        let first = transition.advance(0, &parent)?;
        let second = transition.advance(1, &parent)?;
        assert!(parent.is_empty());
        assert_eq!(first, vec![0]);
        assert_eq!(second, vec![1]);
        assert_ne!(
            transition.distribution(&[0.0], &first)?,
            transition.distribution(&[0.0], &second)?
        );
        Ok(())
    }

    #[test]
    fn matches_independent_lattice_and_pruning_reference() -> Result<(), String> {
        for frame_count in 1..=2 {
            for cap in 0..=2 {
                for width in [1, 2, 32] {
                    let mut transition = fake();
                    let (expected, representative) =
                        reference(&transition.rows, frame_count, cap, width);
                    let frames = Frames(
                        (0..frame_count)
                            .map(|i| if i == 0 { vec![0.0] } else { vec![1.0] })
                            .collect(),
                    );
                    let output = decode(
                        &mut transition,
                        &frames,
                        options(width, cap),
                        &ExecutionControl::without_deadline(),
                    )?;
                    assert_eq!(
                        output.tokens.iter().map(Token::id).collect::<Vec<_>>(),
                        expected
                    );
                    assert!((output.mass - representative.probability.ln()).abs() < 1e-10);
                    assert_eq!(
                        output
                            .tokens
                            .iter()
                            .map(|t| (t.frame(), t.id()))
                            .collect::<Vec<_>>(),
                        representative.alignment
                    );
                    assert!(transition.calls <= frame_count * width * (cap + 1));
                    assert!(transition.advances <= frame_count * width * cap);
                }
            }
        }
        Ok(())
    }
    #[test]
    fn same_frame_occurrences_keep_real_scores_and_tie_order() -> Result<(), String> {
        let mut transition = fake();
        transition.rows = vec![vec![4.0, 4.0, -2.0], vec![4.0, 4.0, -2.0]];
        let output = decode(
            &mut transition,
            &Frames(vec![vec![0.0], vec![1.0]]),
            options(32, 2),
            &ExecutionControl::without_deadline(),
        )?;
        let (expected, representative) = reference(&transition.rows, 2, 2, 32);
        assert_eq!(
            output.tokens.iter().map(Token::id).collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            output
                .tokens
                .iter()
                .map(|t| (t.frame(), t.id()))
                .collect::<Vec<_>>(),
            representative.alignment
        );
        let log_probability = -(2.0_f64 + (-6.0_f64).exp()).ln();
        for token in output.tokens() {
            let score = token.log_probability().ok_or("missing scored token")?;
            assert!((score - log_probability).abs() < 1e-12);
        }
        Ok(())
    }

    #[test]
    fn validates_rows_controls_and_empty_input() -> Result<(), String> {
        let mut transition = fake();
        let empty = decode(
            &mut transition,
            &Frames(vec![]),
            options(2, 2),
            &ExecutionControl::without_deadline(),
        )?;
        assert!(empty.tokens().is_empty());
        assert_eq!(transition.calls, 0);
        transition.rows[0][0] = f32::NAN;
        assert!(
            decode(
                &mut transition,
                &Frames(vec![vec![0.0]]),
                options(2, 2),
                &ExecutionControl::without_deadline()
            )
            .is_err()
        );
        let control = ExecutionControl::for_request();
        let mut transition = fake();
        transition.cancel = Some(control.clone());
        assert!(
            decode(
                &mut transition,
                &Frames(vec![vec![0.0]]),
                options(2, 2),
                &control
            )
            .is_err()
        );
        assert_eq!(transition.calls, 1);
        assert_eq!(transition.advances, 0);
        assert!(
            decode(
                &mut fake(),
                &Frames(vec![vec![f32::NAN]]),
                options(2, 2),
                &ExecutionControl::without_deadline()
            )
            .is_err()
        );
        assert!(
            decode(
                &mut fake(),
                &Frames(vec![vec![0.0]]),
                options(usize::MAX, 2),
                &ExecutionControl::without_deadline()
            )
            .is_err()
        );
        Ok(())
    }
}
