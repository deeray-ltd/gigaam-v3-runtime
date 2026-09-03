// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Text comparison and word-level edit-distance operations.

/// Normalizes a word for lexical comparison while retaining apostrophes and hyphens.
pub fn normalize_word(word: &str) -> String {
    word.chars()
        .filter(|character| character.is_alphanumeric() || matches!(character, '\'' | '-'))
        .flat_map(char::to_lowercase)
        .map(|character| {
            if character == '\u{0451}' {
                '\u{0435}'
            } else {
                character
            }
        })
        .collect()
}

/// Returns the Levenshtein distance between normalized reference and hypothesis words.
///
/// The distance counts insertions, deletions, and substitutions. Divide it by [`word_count`]
/// of the reference to calculate word error rate.
pub fn word_edits(reference: &str, hypothesis: &str) -> usize {
    let reference_words = normalized_words(reference);
    let hypothesis_words = normalized_words(hypothesis);
    let mut previous: Vec<usize> = (0..=hypothesis_words.len()).collect();
    let mut current = vec![0_usize; hypothesis_words.len() + 1];

    for reference_index in 1..=reference_words.len() {
        current[0] = reference_index;
        for hypothesis_index in 1..=hypothesis_words.len() {
            let substitution_cost = usize::from(
                reference_words[reference_index - 1] != hypothesis_words[hypothesis_index - 1],
            );
            let substitution = previous[hypothesis_index - 1] + substitution_cost;
            let deletion = previous[hypothesis_index] + 1;
            let insertion = current[hypothesis_index - 1] + 1;
            current[hypothesis_index] = substitution.min(deletion).min(insertion);
        }
        std::mem::swap(&mut previous, &mut current);
    }

    previous[hypothesis_words.len()]
}

/// Counts normalized words that contain at least one letter or digit.
pub fn word_count(text: &str) -> usize {
    normalized_words(text).len()
}

fn normalized_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(normalize_word)
        .filter(|word| contains_alphanumeric(word))
        .collect()
}

fn contains_alphanumeric(word: &str) -> bool {
    word.chars().any(char::is_alphanumeric)
}

#[cfg(test)]
mod tests {
    use super::{normalize_word, word_count, word_edits};

    #[test]
    fn normalization_preserves_lexical_equivalence() {
        assert_eq!(normalize_word("Hello,!"), "hello");
        assert_eq!(normalize_word("\u{0451}"), "\u{0435}");
        assert_eq!(normalize_word("rock-'n'-roll"), "rock-'n'-roll");
    }

    #[test]
    fn edits_ignore_case_and_punctuation() {
        assert_eq!(word_edits("Hello, world!", "hello world"), 0);
        assert_eq!(word_edits("one two three", "one three"), 1);
        assert_eq!(word_edits("one two three", "one two three four"), 1);
        assert_eq!(word_edits("one two three", "one three two"), 2);
    }

    #[test]
    fn word_count_excludes_punctuation_only_tokens() {
        assert_eq!(word_count("one, two — three"), 3);
        assert_eq!(word_count("' - —"), 0);
    }

    #[test]
    fn edit_distance_is_symmetric() {
        let left = "one two three";
        let right = "one four three five";
        assert_eq!(word_edits(left, right), word_edits(right, left));
    }
}
