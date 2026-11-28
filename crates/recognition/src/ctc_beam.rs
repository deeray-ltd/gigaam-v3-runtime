// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.

//! Pure CTC prefix beam search over borrowed score rows.
//!
//! Prefixes are ranked by the sum of retained alignment probabilities, while the
//! returned labels describe the highest-probability retained alignment. A finite
//! beam prunes paths, and width one is not necessarily equivalent to greedy decoding.

use crate::{ScoreDetail, Token};
use std::num::NonZeroUsize;

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
    pub fn into_parts(self) -> (Vec<usize>, Vec<Token>) {
        (self.labels, self.tokens)
    }
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Rank(usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NodeId(NonZeroUsize);
struct Node {
    parent: Option<NodeId>,
    label: usize,
}
#[derive(Clone, Copy)]
struct Representative {
    score: f64,
    tail: Option<NodeId>,
    rank: Rank,
}
#[derive(Clone, Copy)]
struct State {
    mass: Mass,
    representative: Option<Representative>,
}
struct SearchResult {
    prefixes: Vec<Prefix>,
    arena: Vec<Node>,
}
struct Prefix {
    ids: Vec<usize>,
    blank: State,
    nonblank: State,
}
#[derive(Clone, Copy)]
struct RefPath<'a> {
    parent: &'a Representative,
    label: usize,
    score: f64,
}
#[derive(Default)]
struct DestinationState<'a> {
    mass: Mass,
    best: Option<RefPath<'a>>,
}
struct Candidate<'a> {
    parent: usize,
    appended: Option<usize>,
    blank: DestinationState<'a>,
    nonblank: DestinationState<'a>,
}
fn key<'a>(
    candidate: &'a Candidate<'_>,
    current: &'a [Prefix],
) -> impl Iterator<Item = usize> + 'a {
    current[candidate.parent]
        .ids
        .iter()
        .copied()
        .chain(candidate.appended)
}
fn path_key(path: RefPath<'_>) -> (Rank, usize) {
    (path.parent.rank, path.label)
}
fn contribute<'a>(
    target: &mut DestinationState<'a>,
    source: &'a State,
    label: usize,
    q: f64,
) -> Result<(), String> {
    target.mass = add(target.mass, extend(source.mass, q)?)?;
    if let Some(parent) = source.representative.as_ref() {
        let score = parent.score + q;
        if !score.is_finite() {
            return Err("invalid CTC path score".into());
        }
        let candidate = RefPath {
            parent,
            label,
            score,
        };
        let replace = match target.best {
            None => true,
            Some(old) => {
                score > old.score || (score == old.score && path_key(candidate) < path_key(old))
            }
        };
        if replace {
            target.best = Some(candidate);
        }
    }
    Ok(())
}
fn materialize(
    s: &DestinationState<'_>,
    keys: &[(Rank, usize)],
    arena: &mut Vec<Node>,
) -> Result<State, String> {
    let representative = if let Some(best) = s.best {
        let parent = best.parent;
        let key = (parent.rank, best.label);
        let rank = Rank(keys.binary_search(&key).expect("rank key exists"));
        let index = arena
            .len()
            .checked_add(1)
            .and_then(NonZeroUsize::new)
            .ok_or("CTC history count overflow")?;
        arena.push(Node {
            parent: parent.tail,
            label: best.label,
        });
        Some(Representative {
            score: best.score,
            tail: Some(NodeId(index)),
            rank,
        })
    } else {
        None
    };
    Ok(State {
        mass: s.mass,
        representative,
    })
}

/// Runs prefix search, calling `checkpoint` before work, per frame, and during expansion.
/// An error from the checkpoint aborts without returning a partial result.
pub fn decode<F>(input: &CtcBeamInput<'_>, mut checkpoint: F) -> Result<CtcBeamOutput, String>
where
    F: FnMut() -> Result<(), String>,
{
    let SearchResult {
        prefixes: current,
        arena,
    } = search(input, &mut checkpoint)?;
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
    let best = match (prefix.blank.representative, prefix.nonblank.representative) {
        (Some(a), Some(b)) => {
            if a.score > b.score || (a.score == b.score && a.rank <= b.rank) {
                a
            } else {
                b
            }
        }
        (Some(a), None) | (None, Some(a)) => a,
        (None, None) => return Err("CTC winning prefix has no path".into()),
    };
    let token_capacity = prefix.ids.len();
    drop(current);
    let mut labels = Vec::new();
    reserve(&mut labels, input.frames)?;
    let mut tail = best.tail;
    for position in 0..input.frames {
        if position.is_multiple_of(256) {
            checkpoint()?;
        }
        let id = tail.ok_or("CTC history ended early")?;
        let node = arena.get(id.0.get() - 1).ok_or("invalid CTC history")?;
        labels.push(node.label);
        tail = node.parent;
    }
    if tail.is_some() {
        return Err("CTC history exceeds frame count".into());
    }
    labels.reverse();
    drop(arena);
    checkpoint()?;
    let mut tokens = Vec::new();
    reserve(&mut tokens, token_capacity)?;
    let mut previous = None;
    for (frame, &label) in labels.iter().enumerate() {
        if frame.is_multiple_of(256) {
            checkpoint()?;
        }
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
    checkpoint()?;
    Ok(CtcBeamOutput { labels, tokens })
}

fn search<F>(input: &CtcBeamInput<'_>, checkpoint: &mut F) -> Result<SearchResult, String>
where
    F: FnMut() -> Result<(), String>,
{
    search_observed(
        input,
        checkpoint,
        #[cfg(test)]
        &mut |_, _, _| {},
    )
}

#[cfg(test)]
#[derive(Default)]
struct SearchObservation<'a> {
    candidates: &'a [Candidate<'a>],
    candidate_capacity: usize,
    rank_capacity: usize,
    slot_capacity: usize,
    arena_len: usize,
    arena_capacity: usize,
    key_capacity: usize,
    current_capacity: usize,
    next_capacity: usize,
    ancestor_comparisons: usize,
    slot_lookups: usize,
}

// Observe lookup work and storage as the live buffers change between search phases.
fn search_observed<F>(
    input: &CtcBeamInput<'_>,
    checkpoint: &mut F,
    #[cfg(test)] observe: &mut impl FnMut(SearchObservation<'_>, &[Prefix], &[Prefix]),
) -> Result<SearchResult, String>
where
    F: FnMut() -> Result<(), String>,
{
    checkpoint()?;
    let empty = State {
        mass: None,
        representative: None,
    };
    let mut current = Vec::new();
    reserve(&mut current, 1)?;
    current.push(Prefix {
        ids: Vec::new(),
        blank: State {
            mass: Some(0.0),
            representative: Some(Representative {
                score: 0.0,
                tail: None,
                rank: Rank(0),
            }),
        },
        nonblank: empty,
    });
    let mut arena = Vec::new();
    const NO_CANDIDATE: usize = usize::MAX;
    let stride = input
        .vocabulary
        .checked_add(1)
        .ok_or("CTC slot count overflow")?;
    for frame in 0..input.frames {
        checkpoint()?;
        let row = &input.scores[frame * input.vocabulary..(frame + 1) * input.vocabulary];
        let maximum = f64::from(row.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        let sum: f64 = row.iter().map(|v| (f64::from(*v) - maximum).exp()).sum();
        let normalizer = sum.ln();
        if !normalizer.is_finite() {
            return Err("invalid CTC normalization".into());
        }
        let mut candidates: Vec<Candidate<'_>> = Vec::new();
        let slot_count = current
            .len()
            .checked_mul(stride)
            .ok_or("CTC slot count overflow")?;
        // Each retained prefix has at most one unchanged and one destination per label.
        reserve(&mut candidates, slot_count)?;
        let mut slots = Vec::new();
        reserve(&mut slots, slot_count)?;
        slots.resize(slot_count, NO_CANDIDATE);
        #[cfg(test)]
        let mut comparisons = 0_usize;
        #[cfg(test)]
        let mut lookups = 0_usize;
        let mut work = 0_usize;
        for (parent, prefix) in current.iter().enumerate() {
            checkpoint()?;
            let own_slot = parent * stride + input.vocabulary;
            // An unchanged prefix aliases its unique extension only if the immediate
            // predecessor survived pruning. Retained prefixes are lexicographically sorted.
            let unchanged_slot = match prefix.ids.split_last() {
                Some((&last, ancestor)) => match current.binary_search_by(|p| {
                    #[cfg(test)]
                    {
                        comparisons += 1;
                    }
                    p.ids.as_slice().cmp(ancestor)
                }) {
                    Ok(index) => index * stride + last,
                    Err(_) => own_slot,
                },
                None => own_slot,
            };
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
                    #[cfg(test)]
                    {
                        lookups += 1;
                    }
                    let slot = match extension {
                        Some(label) => parent * stride + label,
                        None => unchanged_slot,
                    };
                    let index = &mut slots[slot];
                    if *index == NO_CANDIDATE {
                        *index = candidates.len();
                        candidates.push(Candidate {
                            parent,
                            appended: extension,
                            blank: DestinationState::default(),
                            nonblank: DestinationState::default(),
                        });
                    }
                    let dest = if label == input.blank {
                        &mut candidates[*index].blank
                    } else {
                        &mut candidates[*index].nonblank
                    };
                    if !unchanged_repeat {
                        contribute(dest, &prefix.blank, label, q)?;
                    }
                    if label == input.blank || (!repeat && !unchanged_repeat) || unchanged_repeat {
                        contribute(dest, &prefix.nonblank, label, q)?;
                    }
                }
            }
        }
        #[cfg(test)]
        observe(
            SearchObservation {
                current_capacity: current.capacity(),
                candidates: &candidates,
                candidate_capacity: candidates.capacity(),
                slot_capacity: slots.capacity(),
                arena_len: arena.len(),
                arena_capacity: arena.capacity(),
                ancestor_comparisons: comparisons,
                slot_lookups: lookups,
                ..SearchObservation::default()
            },
            &current,
            &[],
        );
        drop(slots);
        // Remove unreachable destinations, then order by mass and lexicographic identity.
        let mut ranked = Vec::new();
        reserve(&mut ranked, candidates.len())?;
        #[cfg(test)]
        observe(
            SearchObservation {
                current_capacity: current.capacity(),
                candidates: &candidates,
                candidate_capacity: candidates.capacity(),
                rank_capacity: ranked.capacity(),
                arena_len: arena.len(),
                arena_capacity: arena.capacity(),
                ..SearchObservation::default()
            },
            &current,
            &[],
        );
        for candidate in candidates {
            if let Some(mass) = add(candidate.blank.mass, candidate.nonblank.mass)? {
                ranked.push((candidate, mass));
            }
        }
        checkpoint()?;
        ranked.sort_unstable_by(|(a, am), (b, bm)| {
            bm.total_cmp(am)
                .then_with(|| key(a, &current).cmp(key(b, &current)))
        });
        checkpoint()?;
        ranked.truncate(input.width);
        checkpoint()?;
        let mut keys = Vec::new();
        let needed = ranked
            .len()
            .checked_mul(2)
            .ok_or("CTC rank count overflow")?;
        reserve(&mut keys, needed)?;
        for (candidate, _) in &ranked {
            for best in [candidate.blank.best, candidate.nonblank.best]
                .into_iter()
                .flatten()
            {
                keys.push(path_key(best));
            }
        }
        arena
            .len()
            .checked_add(keys.len())
            .ok_or("CTC history count overflow")?;
        arena
            .try_reserve(keys.len())
            .map_err(|_| "CTC allocation failed".to_owned())?;
        checkpoint()?;
        keys.sort_unstable();
        keys.dedup();
        checkpoint()?;
        let mut next = Vec::new();
        reserve(&mut next, ranked.len())?;
        #[cfg(test)]
        let ranked_capacity = ranked.capacity();
        for (candidate, _) in ranked {
            checkpoint()?;
            next.push(Prefix {
                ids: copy_append(&current[candidate.parent].ids, candidate.appended)?,
                blank: materialize(&candidate.blank, &keys, &mut arena)?,
                nonblank: materialize(&candidate.nonblank, &keys, &mut arena)?,
            });
            #[cfg(test)]
            observe(
                SearchObservation {
                    current_capacity: current.capacity(),
                    rank_capacity: ranked_capacity,
                    arena_len: arena.len(),
                    arena_capacity: arena.capacity(),
                    key_capacity: keys.capacity(),
                    next_capacity: next.capacity(),
                    ..SearchObservation::default()
                },
                &current,
                &next,
            );
        }
        next.sort_unstable_by(|a, b| a.ids.cmp(&b.ids));
        #[cfg(test)]
        observe(
            SearchObservation {
                current_capacity: current.capacity(),
                arena_len: arena.len(),
                arena_capacity: arena.capacity(),
                key_capacity: keys.capacity(),
                next_capacity: next.capacity(),
                ..SearchObservation::default()
            },
            &current,
            &next,
        );
        current = next;
    }
    Ok(SearchResult {
        prefixes: current,
        arena,
    })
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
                let actual = search(&input, &mut || Ok(()))?.prefixes;
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
                let mut saw_expansion = false;
                let mut saw_ranking = false;
                let mut saw_materialization = false;
                let mut largest_scratch = 0;
                let mut largest_history = 0;
                search_observed(&input, &mut || Ok(()), &mut |storage, old, new| {
                    saw_expansion |= storage.slot_capacity > 0;
                    saw_ranking |= storage.rank_capacity > 0 && storage.candidate_capacity > 0;
                    saw_materialization |= storage.rank_capacity > 0 && storage.next_capacity > 0;
                    let parent_capacity = old.len() * (vocabulary + 1);
                    assert!(storage.slot_capacity <= parent_capacity);
                    assert!(storage.candidate_capacity <= parent_capacity);
                    assert!(storage.rank_capacity <= parent_capacity);
                    assert!(storage.arena_len <= 2 * width * frames);
                    assert!(storage.arena_capacity >= storage.arena_len);
                    assert!(storage.key_capacity <= 2 * width);
                    assert!(storage.next_capacity <= width);
                    let live_scratch = storage.slot_capacity * size_of::<usize>()
                        + storage.candidate_capacity * size_of::<Candidate>()
                        + storage.rank_capacity * size_of::<(Candidate, f64)>();
                    let scratch_budget =
                        parent_capacity * (size_of::<Candidate>() + size_of::<(Candidate, f64)>());
                    assert!(live_scratch <= scratch_budget);
                    assert!(storage.slot_capacity == 0 || storage.rank_capacity == 0);
                    largest_scratch = largest_scratch.max(storage.candidates.len());
                    assert!(old.len() <= width && new.len() <= width);
                    let history: usize = old.iter().chain(new).map(|prefix| prefix.ids.len()).sum();
                    largest_history = largest_history.max(history);
                    assert!(history <= 6 * width * frames);
                })?;
                assert!(saw_expansion && saw_ranking && saw_materialization);
                assert!(largest_scratch >= vocabulary);
                assert!(largest_history > 0);
            }
        }
        Ok(())
    }

    #[test]
    fn live_capacity_accounts_for_materialization_and_output() -> Result<(), String> {
        for (frames, width) in [(64, 2), (256, 2), (256, 8), (1024, 8)] {
            let rows = vec![0.0; frames * 3];
            let input = test_input(&rows, frames, 3, 0, width, ScoreDetail::None)?;
            let mut peak = 0;
            let mut materializations = 0;
            let result = search_observed(&input, &mut || Ok(()), &mut |storage, old, next| {
                let ids = old
                    .iter()
                    .chain(next)
                    .map(|p| p.ids.capacity() * size_of::<usize>())
                    .sum::<usize>();
                let bytes = storage.arena_capacity * size_of::<Node>()
                    + storage.slot_capacity * size_of::<usize>()
                    + storage.candidate_capacity * size_of::<Candidate>()
                    + storage.rank_capacity * size_of::<(Candidate, f64)>()
                    + storage.key_capacity * size_of::<(Rank, usize)>()
                    + (storage.current_capacity + storage.next_capacity) * size_of::<Prefix>()
                    + ids;
                peak = peak.max(bytes);
                if storage.next_capacity > 0 && storage.rank_capacity > 0 {
                    materializations += 1;
                    assert!(
                        storage.key_capacity > 0 && storage.arena_capacity >= storage.arena_len
                    );
                }
            })?;
            assert!(materializations > 0);
            assert!(result.arena.len() <= 2 * width * frames);
            let output = decode(&input, || Ok(()))?;
            let reconstruction_bytes = result.arena.capacity() * size_of::<Node>()
                + output.labels.capacity() * size_of::<usize>();
            let projection_bytes = output.labels.capacity() * size_of::<usize>()
                + output.tokens.capacity() * size_of::<Token>();
            eprintln!(
                "frames={frames} width={width} search_peak={peak} reconstruction={reconstruction_bytes} projection={projection_bytes} nodes={} arena_capacity={}",
                result.arena.len(),
                result.arena.capacity()
            );
        }
        Ok(())
    }

    #[test]
    fn duplicate_rank_keys_share_identity_across_distinct_nodes() -> Result<(), String> {
        let first = Representative {
            score: 0.0,
            tail: Some(NodeId(NonZeroUsize::MIN)),
            rank: Rank(0),
        };
        let second = Representative {
            score: 0.0,
            tail: Some(NodeId(NonZeroUsize::new(2).expect("two is nonzero"))),
            rank: Rank(0),
        };
        let a = RefPath {
            parent: &first,
            label: 2,
            score: 0.0,
        };
        let b = RefPath {
            parent: &second,
            ..a
        };
        assert_eq!(path_key(a), path_key(b));
        let keys = [path_key(a)];
        let mut arena = vec![
            Node {
                parent: None,
                label: 1,
            },
            Node {
                parent: None,
                label: 1,
            },
        ];
        let x = materialize(
            &DestinationState {
                mass: Some(0.0),
                best: Some(a),
            },
            &keys,
            &mut arena,
        )?
        .representative
        .expect("reachable destination has a representative");
        let y = materialize(
            &DestinationState {
                mass: Some(0.0),
                best: Some(b),
            },
            &keys,
            &mut arena,
        )?
        .representative
        .expect("reachable destination has a representative");
        assert_ne!(x.tail, y.tail);
        assert_eq!(x.rank, y.rank);
        Ok(())
    }

    #[test]
    fn history_ranks_match_full_paths_and_growth_is_bounded() -> Result<(), String> {
        for width in [1, 4, 12] {
            for frames in [16, 64, 256] {
                let rows = vec![0.0; frames * 3];
                let input = test_input(&rows, frames, 3, 1, width, ScoreDetail::None)?;
                let result = search(&input, &mut || Ok(()))?;
                assert!(result.arena.len() <= 2 * width * frames);
                let mut paths = Vec::new();
                for prefix in &result.prefixes {
                    for rep in [prefix.blank.representative, prefix.nonblank.representative]
                        .into_iter()
                        .flatten()
                    {
                        let mut path = Vec::new();
                        let mut tail = rep.tail;
                        while let Some(id) = tail {
                            let node = &result.arena[id.0.get() - 1];
                            path.push(node.label);
                            tail = node.parent;
                        }
                        assert_eq!(path.len(), frames);
                        path.reverse();
                        paths.push((rep.rank, path));
                    }
                }
                for (ar, ap) in &paths {
                    for (br, bp) in &paths {
                        assert_eq!(ar.cmp(br), ap.cmp(bp));
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn cancellation_is_checked_during_reconstruction_and_projection() -> Result<(), String> {
        let frames = 1024;
        let rows = vec![0.0; frames * 2];
        let input = test_input(&rows, frames, 2, 0, 2, ScoreDetail::None)?;
        let mut search_calls = 0;
        search(&input, &mut || {
            search_calls += 1;
            Ok(())
        })?;
        for offset in [1, 4, 6, 9] {
            let mut calls = 0;
            let result = decode(&input, || {
                calls += 1;
                if calls == search_calls + offset {
                    Err("cancelled".into())
                } else {
                    Ok(())
                }
            });
            assert_eq!(result.unwrap_err(), "cancelled");
        }
        Ok(())
    }

    fn linear_destination_keys(
        retained: &[Prefix],
        vocabulary: usize,
        blank: usize,
    ) -> (Vec<Vec<usize>>, usize) {
        let mut identities = Vec::new();
        let mut comparisons = 0;
        let mut add = |identity| {
            if !identities.iter().any(|existing| {
                comparisons += 1;
                *existing == identity
            }) {
                identities.push(identity);
            }
        };
        for prefix in retained {
            for label in 0..vocabulary {
                let mut extension = prefix.ids.clone();
                if label != blank {
                    extension.push(label);
                }
                add(extension);
                if prefix.ids.last() == Some(&label) {
                    add(prefix.ids.clone());
                }
            }
        }
        (identities, comparisons)
    }

    #[test]
    fn indexed_destinations_match_ordered_scan_across_pruning_and_vocabularies()
    -> Result<(), String> {
        let mut retained_ancestor = false;
        let mut pruned_ancestor = false;
        let mut empty_prefix = false;
        for (vocabulary, width) in [(3, 2), (3, 8), (16, 4), (64, 8), (256, 32)] {
            let frames = 5;
            let blank = vocabulary / 2;
            let rows: Vec<f32> = (0..frames * vocabulary)
                .map(|i| {
                    f32::from(
                        i16::try_from((i * 17 + i / vocabulary * 11) % 97)
                            .expect("bounded fixture"),
                    ) / 13.0
                })
                .collect();
            let input = test_input(&rows, frames, vocabulary, blank, width, ScoreDetail::None)?;
            let mut observed_full_width = false;
            search_observed(&input, &mut || Ok(()), &mut |work, old, _| {
                if work.slot_capacity == 0 {
                    return;
                }
                observed_full_width |= old.len() == width;
                let (expected, linear_comparisons) =
                    linear_destination_keys(old, vocabulary, blank);
                let actual: Vec<Vec<usize>> = work
                    .candidates
                    .iter()
                    .map(|c| key(c, old).collect())
                    .collect();
                assert_eq!(actual, expected);
                let nonempty = old.iter().filter(|p| !p.ids.is_empty()).count();
                assert_eq!(work.slot_lookups, old.len() * vocabulary + nonempty);
                let comparison_bound = usize::try_from(old.len().ilog2())
                    .expect("log of retained count fits usize")
                    + 2;
                assert!(work.ancestor_comparisons <= nonempty * comparison_bound);
                if vocabulary >= 16 {
                    assert!(work.ancestor_comparisons + work.slot_lookups < linear_comparisons);
                }
                for prefix in old {
                    match prefix.ids.split_last() {
                        Some((_, predecessor)) => {
                            let present = old.iter().any(|p| p.ids == predecessor);
                            retained_ancestor |= present;
                            pruned_ancestor |= !present;
                        }
                        None => empty_prefix = true,
                    }
                }
            })?;
            assert!(observed_full_width);
        }
        assert!(retained_ancestor && pruned_ancestor && empty_prefix);
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
