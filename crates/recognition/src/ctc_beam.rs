// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Pure CTC prefix beam search over borrowed score rows.
//!
//! Prefixes are ranked by the sum of retained alignment probabilities, while the
//! returned labels describe the highest-probability retained alignment. A finite
//! beam prunes paths, and width one is not necessarily equivalent to greedy decoding.

use crate::{ScoreDetail, Token};
use std::cmp::Ordering;

/// Search choices for a CTC score matrix.
#[derive(Clone, Copy, Debug)]
pub struct CtcBeamOptions {
    /// Blank label identifier within the matrix vocabulary.
    pub blank: usize,
    /// Maximum number of prefixes retained after each frame; must be positive.
    pub width: usize,
    /// Whether emitted tokens include their normalized row log probabilities.
    pub detail: ScoreDetail,
}

/// Validated search configuration and borrowed contiguous `[frames][vocabulary]` scores.
pub struct CtcBeamInput<'a> {
    scores: &'a [f32],
    frames: usize,
    vocabulary: usize,
    blank: usize,
    width: usize,
    detail: ScoreDetail,
}

impl<'a> CtcBeamInput<'a> {
    /// Validates dimensions, finite scores, blank identity, and positive beam width.
    pub fn new(
        scores: &'a [f32],
        frames: usize,
        vocabulary: usize,
        options: CtcBeamOptions,
    ) -> Result<Self, String> {
        let CtcBeamOptions {
            blank,
            width,
            detail,
        } = options;
        if vocabulary == 0 || blank >= vocabulary || width == 0 {
            return Err("invalid CTC vocabulary, blank, or beam width".into());
        }
        let count = frames
            .checked_mul(vocabulary)
            .ok_or("CTC matrix dimensions overflow")?;
        if scores.len() != count || scores.iter().any(|x| !x.is_finite()) {
            return Err("invalid CTC score matrix".into());
        }
        width
            .checked_mul(vocabulary)
            .and_then(|n| n.checked_add(width))
            .ok_or("CTC candidate count overflow")?;
        width
            .checked_mul(frames)
            .ok_or("CTC history count overflow")?;
        Ok(Self {
            scores,
            frames,
            vocabulary,
            blank,
            width,
            detail,
        })
    }
}

/// A single coherent retained alignment and its collapsed nonblank tokens.
#[derive(Debug, PartialEq)]
pub struct CtcBeamOutput {
    labels: Vec<usize>,
    tokens: Vec<Token>,
}
impl CtcBeamOutput {
    /// One label per input frame, including blanks.
    pub fn labels(&self) -> &[usize] {
        &self.labels
    }
    /// Collapsed nonblank tokens, with first-run frames and optional row scores.
    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }
}

// None represents impossible mass; every Some is finite.
type Mass = Option<f64>;
fn add(a: Mass, b: Mass) -> Result<Mass, String> {
    match (a, b) {
        (None, x) | (x, None) => Ok(x),
        (Some(x), Some(y)) => {
            let hi = x.max(y);
            let value = hi + (x.min(y) - hi).exp().ln_1p();
            if value.is_finite() {
                Ok(Some(value))
            } else {
                Err("invalid CTC mass".into())
            }
        }
    }
}
fn extend(a: Mass, q: f64) -> Result<Mass, String> {
    match a {
        None => Ok(None),
        Some(x) => {
            let value = x + q;
            if value.is_finite() {
                Ok(Some(value))
            } else {
                Err("invalid CTC path score".into())
            }
        }
    }
}
fn reserve<T>(v: &mut Vec<T>, n: usize) -> Result<(), String> {
    v.try_reserve_exact(n)
        .map_err(|_| "CTC allocation failed".to_owned())
}
fn copy_append<T: Copy>(source: &[T], extra: Option<T>) -> Result<Vec<T>, String> {
    let mut result = Vec::new();
    reserve(
        &mut result,
        source
            .len()
            .checked_add(usize::from(extra.is_some()))
            .ok_or("CTC storage overflow")?,
    )?;
    result.extend_from_slice(source);
    if let Some(value) = extra {
        result.push(value);
    }
    Ok(result)
}
#[derive(Clone)]
struct State {
    mass: Mass,
    score: Mass,
    path: Vec<usize>,
}
struct Prefix {
    ids: Vec<usize>,
    blank: State,
    nonblank: State,
}
#[derive(Clone, Copy)]
struct RefPath {
    parent: usize,
    blank: bool,
    label: usize,
    score: f64,
}
#[derive(Default)]
struct DestinationState {
    mass: Mass,
    best: Option<RefPath>,
}
struct Candidate {
    parent: usize,
    appended: Option<usize>,
    blank: DestinationState,
    nonblank: DestinationState,
}
fn state(prefix: &Prefix, blank: bool) -> &State {
    if blank {
        &prefix.blank
    } else {
        &prefix.nonblank
    }
}
fn key<'a>(candidate: &'a Candidate, current: &'a [Prefix]) -> impl Iterator<Item = usize> + 'a {
    current[candidate.parent]
        .ids
        .iter()
        .copied()
        .chain(candidate.appended)
}
fn path_cmp(a: RefPath, b: RefPath, current: &[Prefix]) -> Ordering {
    state(&current[a.parent], a.blank)
        .path
        .iter()
        .copied()
        .chain(Some(a.label))
        .cmp(
            state(&current[b.parent], b.blank)
                .path
                .iter()
                .copied()
                .chain(Some(b.label)),
        )
}
fn contribute(
    target: &mut DestinationState,
    source: &State,
    parent: usize,
    blank: bool,
    label: usize,
    q: f64,
    current: &[Prefix],
) -> Result<(), String> {
    target.mass = add(target.mass, extend(source.mass, q)?)?;
    if let Some(score) = extend(source.score, q)? {
        let candidate = RefPath {
            parent,
            blank,
            label,
            score,
        };
        let replace = match target.best {
            None => true,
            Some(old) => {
                score > old.score
                    || (score == old.score && path_cmp(candidate, old, current).is_lt())
            }
        };
        if replace {
            target.best = Some(candidate);
        }
    }
    Ok(())
}
fn materialize(s: &DestinationState, current: &[Prefix]) -> Result<State, String> {
    match s.best {
        None => Ok(State {
            mass: s.mass,
            score: None,
            path: Vec::new(),
        }),
        Some(best) => Ok(State {
            mass: s.mass,
            score: Some(best.score),
            path: copy_append(
                &state(&current[best.parent], best.blank).path,
                Some(best.label),
            )?,
        }),
    }
}

/// Runs prefix search, calling `checkpoint` before work, per frame, and during expansion.
/// An error from the checkpoint aborts without returning a partial result.
pub fn decode<F>(input: &CtcBeamInput<'_>, mut checkpoint: F) -> Result<CtcBeamOutput, String>
where
    F: FnMut() -> Result<(), String>,
{
    let current = search(input, &mut checkpoint)?;
    let mut winner: Option<(usize, f64)> = None;
    for (i, prefix) in current.iter().enumerate() {
        if let Some(mass) = add(prefix.blank.mass, prefix.nonblank.mass)?
            && winner
                .is_none_or(|(j, old)| mass > old || (mass == old && prefix.ids < current[j].ids))
        {
            winner = Some((i, mass));
        }
    }
    let (index, _) = winner.ok_or("CTC search produced no path")?;
    let prefix = &current[index];
    let best = match (prefix.blank.score, prefix.nonblank.score) {
        (Some(a), Some(b)) => {
            if a > b || (a == b && prefix.blank.path <= prefix.nonblank.path) {
                &prefix.blank
            } else {
                &prefix.nonblank
            }
        }
        (Some(_), None) => &prefix.blank,
        (None, Some(_)) => &prefix.nonblank,
        (None, None) => return Err("CTC winning prefix has no path".into()),
    };
    let labels = copy_append(&best.path, None)?;
    let mut tokens = Vec::new();
    reserve(&mut tokens, prefix.ids.len())?;
    let mut previous = None;
    for (frame, &label) in labels.iter().enumerate() {
        if label != input.blank && previous != Some(label) {
            let token = match input.detail {
                ScoreDetail::None => Token::new(label, frame),
                ScoreDetail::TokenLogProbabilities => Token::scored(
                    label,
                    frame,
                    crate::contracts::selected_log_probability(
                        &input.scores[frame * input.vocabulary..(frame + 1) * input.vocabulary],
                        label,
                    )?,
                )?,
            };
            tokens.push(token);
        }
        previous = Some(label);
    }
    Ok(CtcBeamOutput { labels, tokens })
}

fn search<F>(input: &CtcBeamInput<'_>, checkpoint: &mut F) -> Result<Vec<Prefix>, String>
where
    F: FnMut() -> Result<(), String>,
{
    search_observed(input, checkpoint, &mut |_, _, _, _| {})
}

// The observer permits tests to count live entries at the two peak-storage phases.
fn search_observed<F, O>(
    input: &CtcBeamInput<'_>,
    checkpoint: &mut F,
    observe: &mut O,
) -> Result<Vec<Prefix>, String>
where
    F: FnMut() -> Result<(), String>,
    O: FnMut(usize, (usize, usize), &[Prefix], &[Prefix]),
{
    checkpoint()?;
    let empty = State {
        mass: None,
        score: None,
        path: Vec::new(),
    };
    let mut current = Vec::new();
    reserve(&mut current, 1)?;
    current.push(Prefix {
        ids: Vec::new(),
        blank: State {
            mass: Some(0.0),
            score: Some(0.0),
            path: Vec::new(),
        },
        nonblank: empty,
    });
    let candidate_limit = input
        .width
        .checked_mul(input.vocabulary)
        .and_then(|n| n.checked_add(input.width))
        .ok_or("CTC candidate count overflow")?;
    for frame in 0..input.frames {
        checkpoint()?;
        let row = &input.scores[frame * input.vocabulary..(frame + 1) * input.vocabulary];
        let maximum = f64::from(row.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        let sum: f64 = row.iter().map(|v| (f64::from(*v) - maximum).exp()).sum();
        let normalizer = sum.ln();
        if !normalizer.is_finite() {
            return Err("invalid CTC normalization".into());
        }
        let mut candidates: Vec<Candidate> = Vec::new();
        reserve(&mut candidates, candidate_limit)?;
        let mut work = 0_usize;
        for (parent, prefix) in current.iter().enumerate() {
            for (label, &value) in row.iter().enumerate() {
                if work.is_multiple_of(256) {
                    checkpoint()?;
                }
                work = work.checked_add(1).ok_or("CTC work count overflow")?;
                let q = (f64::from(value) - maximum) - normalizer;
                let repeat = prefix.ids.last() == Some(&label);
                // Blank and repeated nonblank can both contribute to unchanged prefix.
                let appended = if label == input.blank {
                    None
                } else {
                    Some(label)
                };
                for unchanged_repeat in [false, true] {
                    if unchanged_repeat && (label == input.blank || !repeat) {
                        continue;
                    }
                    let extension = if unchanged_repeat { None } else { appended };
                    let probe = Candidate {
                        parent,
                        appended: extension,
                        blank: DestinationState::default(),
                        nonblank: DestinationState::default(),
                    };
                    let index = match candidates
                        .iter()
                        .position(|c| key(c, &current).eq(key(&probe, &current)))
                    {
                        Some(index) => index,
                        None => {
                            candidates.push(probe);
                            candidates.len() - 1
                        }
                    };
                    let dest = if label == input.blank {
                        &mut candidates[index].blank
                    } else {
                        &mut candidates[index].nonblank
                    };
                    if !unchanged_repeat {
                        contribute(dest, &prefix.blank, parent, true, label, q, &current)?;
                    }
                    if label == input.blank || (!repeat && !unchanged_repeat) || unchanged_repeat {
                        contribute(dest, &prefix.nonblank, parent, false, label, q, &current)?;
                    }
                }
            }
        }
        // Remove unreachable destinations, then order by mass and lexicographic identity.
        let mut ranked = Vec::new();
        reserve(&mut ranked, candidates.len())?;
        observe(
            candidates.len(),
            (candidates.capacity(), ranked.capacity()),
            &current,
            &[],
        );
        for candidate in candidates {
            if let Some(mass) = add(candidate.blank.mass, candidate.nonblank.mass)? {
                ranked.push((candidate, mass));
            }
        }
        ranked.sort_unstable_by(|(a, am), (b, bm)| {
            bm.total_cmp(am)
                .then_with(|| key(a, &current).cmp(key(b, &current)))
        });
        ranked.truncate(input.width);
        let mut next = Vec::new();
        reserve(&mut next, ranked.len())?;
        for (candidate, _) in ranked {
            next.push(Prefix {
                ids: copy_append(&current[candidate.parent].ids, candidate.appended)?,
                blank: materialize(&candidate.blank, &current)?,
                nonblank: materialize(&candidate.nonblank, &current)?,
            });
        }
        next.sort_unstable_by(|a, b| a.ids.cmp(&b.ids));
        observe(0, (0, 0), &current, &next);
        current = next;
    }
    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn test_input(
        scores: &[f32],
        frames: usize,
        vocabulary: usize,
        blank: usize,
        width: usize,
        detail: ScoreDetail,
    ) -> Result<CtcBeamInput<'_>, String> {
        CtcBeamInput::new(
            scores,
            frames,
            vocabulary,
            CtcBeamOptions {
                blank,
                width,
                detail,
            },
        )
    }

    fn collapse(path: &[usize], blank: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut previous = None;
        for &label in path {
            if label != blank && previous != Some(label) {
                out.push(label);
            }
            previous = Some(label);
        }
        out
    }

    // Enumerates raw frame paths and sums ordinary probabilities independently of
    // the blank/nonblank prefix recurrence. Pruning retains all paths belonging
    // to the selected prefixes at each frame.
    fn reference(
        rows: &[f32],
        vocabulary: usize,
        blank: usize,
        width: usize,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut paths = vec![(Vec::new(), 1.0_f64)];
        for row in rows.chunks_exact(vocabulary) {
            let weights: Vec<f64> = row.iter().map(|x| f64::from(*x).exp()).collect();
            let sum: f64 = weights.iter().sum();
            let mut expanded = Vec::new();
            for (path, probability) in &paths {
                for (label, weight) in weights.iter().enumerate() {
                    let mut new_path = path.clone();
                    new_path.push(label);
                    expanded.push((new_path, probability * weight / sum));
                }
            }
            let mut masses = BTreeMap::<Vec<usize>, f64>::new();
            for (path, probability) in &expanded {
                *masses.entry(collapse(path, blank)).or_insert(0.0) += probability;
            }
            let mut ranked: Vec<_> = masses.into_iter().collect();
            ranked.sort_by(|(a, am), (b, bm)| bm.total_cmp(am).then_with(|| a.cmp(b)));
            ranked.truncate(width);
            expanded
                .retain(|(path, _)| ranked.iter().any(|(ids, _)| *ids == collapse(path, blank)));
            paths = expanded;
        }
        let mut masses = BTreeMap::<Vec<usize>, f64>::new();
        for (path, probability) in &paths {
            *masses.entry(collapse(path, blank)).or_insert(0.0) += probability;
        }
        let mut ranked: Vec<_> = masses.into_iter().collect();
        ranked.sort_by(|(a, am), (b, bm)| bm.total_cmp(am).then_with(|| a.cmp(b)));
        let ids = ranked
            .first()
            .expect("the empty path or a finite lattice exists")
            .0
            .clone();
        paths.retain(|(path, _)| collapse(path, blank) == ids);
        paths.sort_by(|(a, ap), (b, bp)| bp.total_cmp(ap).then_with(|| a.cmp(b)));
        (
            ids,
            paths.first().expect("winning prefix has a path").0.clone(),
        )
    }

    #[test]
    fn lattice_and_pruned_reference() -> Result<(), String> {
        for vocabulary in 2..=3 {
            for frames in 0..=4 {
                for seed in 0..12_usize {
                    let rows: Vec<f32> = (0..frames * vocabulary)
                        .map(|i| {
                            let value = i32::try_from((i * 7 + seed * 3) % 11)
                                .expect("small fixture value");
                            f32::from(i16::try_from(value).expect("small fixture value")) / 3.0
                        })
                        .collect();
                    for width in [1, 2, 3, 128] {
                        let input = test_input(
                            &rows,
                            frames,
                            vocabulary,
                            0,
                            width,
                            ScoreDetail::TokenLogProbabilities,
                        )?;
                        let output = decode(&input, || Ok(()))?;
                        let (ids, labels) = reference(&rows, vocabulary, 0, width);
                        assert_eq!(
                            output.labels(),
                            labels,
                            "v={vocabulary} t={frames} seed={seed} w={width}"
                        );
                        assert_eq!(
                            output.tokens().iter().map(Token::id).collect::<Vec<_>>(),
                            ids
                        );
                        let mut previous = None;
                        let mut emitted = 0;
                        for (frame, &label) in labels.iter().enumerate() {
                            if label != 0 && previous != Some(label) {
                                assert_eq!(output.tokens()[emitted].frame(), frame);
                                let expected = crate::contracts::selected_log_probability(
                                    &rows[frame * vocabulary..(frame + 1) * vocabulary],
                                    label,
                                )?;
                                assert_eq!(
                                    output.tokens()[emitted].log_probability(),
                                    Some(expected)
                                );
                                emitted += 1;
                            }
                            previous = Some(label);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn unpruned_prefix_masses_match_full_path_enumeration() -> Result<(), String> {
        for vocabulary in 2..=3 {
            for frames in 0..=4 {
                let rows: Vec<f32> = (0..frames * vocabulary)
                    .map(|i| {
                        f32::from(i16::try_from((i * 5 + 2) % 7).expect("small fixture")) / 4.0
                    })
                    .collect();
                let mut paths = vec![(Vec::new(), 1.0_f64)];
                for row in rows.chunks_exact(vocabulary) {
                    let weights: Vec<_> = row.iter().map(|x| f64::from(*x).exp()).collect();
                    let sum: f64 = weights.iter().sum();
                    let mut next = Vec::new();
                    for (path, probability) in paths {
                        for (label, weight) in weights.iter().enumerate() {
                            let mut child = path.clone();
                            child.push(label);
                            next.push((child, probability * weight / sum));
                        }
                    }
                    paths = next;
                }
                let mut expected = BTreeMap::<Vec<usize>, f64>::new();
                for (path, probability) in paths {
                    *expected.entry(collapse(&path, 0)).or_insert(0.0) += probability;
                }
                let input = test_input(&rows, frames, vocabulary, 0, 128, ScoreDetail::None)?;
                let actual = search(&input, &mut || Ok(()))?;
                assert_eq!(actual.len(), expected.len());
                for prefix in actual {
                    let actual_mass = add(prefix.blank.mass, prefix.nonblank.mass)?
                        .expect("retained prefix has finite mass")
                        .exp();
                    let expected_mass = expected
                        .get(&prefix.ids)
                        .expect("prefix appears in exhaustive enumeration");
                    assert!((actual_mass - expected_mass).abs() < 1e-12);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn controlled_search_respects_live_storage_bounds() -> Result<(), String> {
        for vocabulary in [2, 5] {
            for width in [1, 3, 8] {
                let frames = 7;
                let rows: Vec<f32> = (0..frames * vocabulary)
                    .map(|i| {
                        f32::from(i16::try_from((i * 7 + 3) % 13).expect("small fixture")) / 5.0
                    })
                    .collect();
                let input = test_input(&rows, frames, vocabulary, 0, width, ScoreDetail::None)?;
                let mut observations = 0;
                let mut largest_scratch = 0;
                let mut largest_history = 0;
                search_observed(
                    &input,
                    &mut || Ok(()),
                    &mut |scratch, scratch_capacity, old, new| {
                        observations += 1;
                        largest_scratch = largest_scratch.max(scratch);
                        assert!(scratch <= width * (vocabulary + 1));
                        assert!(scratch_capacity.0 <= width * (vocabulary + 1));
                        assert!(scratch_capacity.1 <= width * (vocabulary + 1));
                        assert!(old.len() <= width && new.len() <= width);
                        let history: usize = old
                            .iter()
                            .chain(new)
                            .map(|prefix| {
                                prefix.ids.len()
                                    + prefix.blank.path.len()
                                    + prefix.nonblank.path.len()
                            })
                            .sum();
                        largest_history = largest_history.max(history);
                        assert!(history <= 6 * width * frames);
                    },
                )?;
                assert_eq!(observations, 2 * frames);
                assert!(largest_scratch >= vocabulary);
                assert!(largest_history >= frames);
            }
        }
        Ok(())
    }

    #[test]
    fn equal_scores_choose_lexicographic_alignment_and_space_is_nonblank() -> Result<(), String> {
        // At one frame, labels 1 and 2 tie; label 2 is treated as a space token.
        let tied = [-10.0, 0.0, 0.0];
        let input = test_input(&tied, 1, 3, 0, 3, ScoreDetail::None)?;
        assert_eq!(decode(&input, || Ok(()))?.labels(), &[1]);
        // Two equally likely alignments for prefix [2] choose [0, 2] over [2, 0].
        let space = [0.0, -10.0, 0.0, 0.0, -10.0, 0.0];
        let input = test_input(&space, 2, 3, 0, 8, ScoreDetail::None)?;
        let output = decode(&input, || Ok(()))?;
        assert_eq!(output.labels(), &[0, 2]);
        assert_eq!(
            output.tokens().iter().map(Token::id).collect::<Vec<_>>(),
            [2]
        );
        assert_eq!(output.tokens()[0].frame(), 1);
        Ok(())
    }

    #[test]
    fn input_and_checkpoint_refusal() -> Result<(), String> {
        assert!(test_input(&[], 0, 0, 0, 1, ScoreDetail::None).is_err());
        assert!(test_input(&[], 0, 2, 2, 1, ScoreDetail::None).is_err());
        assert!(test_input(&[], 0, 2, 0, 0, ScoreDetail::None).is_err());
        assert!(test_input(&[], usize::MAX, 2, 0, 1, ScoreDetail::None).is_err());
        assert!(test_input(&[], 0, 2, 0, usize::MAX, ScoreDetail::None).is_err());
        assert!(test_input(&[f32::NAN, 0.0], 1, 2, 0, 1, ScoreDetail::None).is_err());
        let rows = [0.0, 1.0, 0.0, 1.0];
        let input = test_input(&rows, 2, 2, 0, 2, ScoreDetail::None)?;
        let mut calls = 0;
        assert!(
            decode(&input, || {
                calls += 1;
                if calls == 3 {
                    Err("cancelled".into())
                } else {
                    Ok(())
                }
            })
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn width_one_can_differ_from_greedy_and_row_shifts_preserve_scores() -> Result<(), String> {
        let rows = [-2.0, 0.0, -2.0, -2.0, -2.0, 0.0];
        let shifted = [8.0, 10.0, 8.0, 8.0, 8.0, 10.0];
        let input = test_input(&rows, 3, 2, 0, 1, ScoreDetail::TokenLogProbabilities)?;
        let other = test_input(&shifted, 3, 2, 0, 1, ScoreDetail::TokenLogProbabilities)?;
        let output = decode(&input, || Ok(()))?;
        assert_eq!(output, decode(&other, || Ok(()))?);
        assert_eq!(
            output.tokens().iter().map(Token::id).collect::<Vec<_>>(),
            [1]
        );
        assert_eq!(crate::ctc::greedy(&rows, 3, 2, 0)?.len(), 2);
        Ok(())
    }

    #[test]
    fn repeated_labels_and_empty_frames() -> Result<(), String> {
        let rows = [-10.0, 10.0, 10.0, -10.0, -10.0, 10.0];
        let input = test_input(&rows, 3, 2, 0, 4, ScoreDetail::None)?;
        let output = decode(&input, || Ok(()))?;
        assert_eq!(output.labels(), &[1, 0, 1]);
        assert_eq!(
            output.tokens().iter().map(Token::frame).collect::<Vec<_>>(),
            [0, 2]
        );
        let input = test_input(&[], 0, 2, 0, 1, ScoreDetail::None)?;
        let output = decode(&input, || Ok(()))?;
        assert!(output.labels().is_empty());
        assert!(output.tokens().is_empty());
        Ok(())
    }
}
