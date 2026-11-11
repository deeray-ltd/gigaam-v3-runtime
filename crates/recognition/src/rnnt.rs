// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 Yuriy Krasilnikov
// Copyright (C) 2026 Deeray Ltd.
//
// This file is part of GigaAM v3 Runtime, free software distributed under
// the terms of the GNU Lesser General Public License, either version 3 of the
// License, or (at your option) any later version. See COPYING.LESSER and COPYING
// for the full terms. There is NO WARRANTY, to the extent permitted by law.

//! Greedy RNN-T transition ordering over injected prediction, joint, and frame ports.

use crate::contracts::Token;

/// Stateful prediction/joint execution required by greedy RNN-T decoding.
///
/// The execution adapter owns its sessions and state representation. This algorithm controls only
/// hypothesis transition order: initialize from blank, query one frame, and advance after each
/// non-blank token.
pub trait RnntTransition<T = f32> {
    type State;

    fn start(&mut self, blank: usize) -> Result<Self::State, String>;

    fn prepare_frame(&mut self, _frame: &[T]) -> Result<(), String> {
        Ok(())
    }

    fn select(&mut self, encoder_frame: &[T], state: &Self::State) -> Result<usize, String>;

    fn select_scored(
        &mut self,
        _encoder_frame: &[T],
        _state: &Self::State,
    ) -> Result<(usize, f64), String> {
        Err("RNN-T transition does not support token scores".into())
    }

    fn advance(&mut self, token: usize, state: &mut Self::State) -> Result<(), String>;
}

/// Row-major encoder-frame access supplied by the execution adapter.
///
/// The adapter owns frame shape and storage cardinality. The greedy algorithm consumes those
/// frames through this port and refuses a non-finite value before invoking a transition.
pub trait RnntFrameSource<T = f32> {
    fn output_frames(&self) -> usize;

    fn frame(&self, index: usize) -> &[T];
}

/// Greedy RNN-T output before vocabulary/timestamp projection.
#[derive(Debug, Clone, PartialEq)]
pub struct RnntOutput {
    tokens: Vec<Token>,
    silence: Vec<bool>,
}

impl RnntOutput {
    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }

    pub fn silence(&self) -> &[bool] {
        &self.silence
    }
}

/// Runs the current greedy RNN-T transition behavior for every frame supplied by the adapter.
pub fn greedy<T: RnntTransition, F: RnntFrameSource>(
    transition: &mut T,
    frames: &F,
    blank: usize,
    max_symbols_per_step: usize,
) -> Result<RnntOutput, String> {
    greedy_with_detail(transition, frames, blank, max_symbols_per_step, false)
}

pub fn greedy_with_detail<T: RnntTransition, F: RnntFrameSource>(
    transition: &mut T,
    frames: &F,
    blank: usize,
    max_symbols_per_step: usize,
    scored: bool,
) -> Result<RnntOutput, String> {
    greedy_typed_with_detail(transition, frames, blank, max_symbols_per_step, scored)
}

pub(crate) trait FiniteFrameValue {
    fn is_finite_value(&self) -> bool;
}

impl FiniteFrameValue for f32 {
    fn is_finite_value(&self) -> bool {
        self.is_finite()
    }
}

impl FiniteFrameValue for half::f16 {
    fn is_finite_value(&self) -> bool {
        self.is_finite()
    }
}

pub(crate) fn greedy_typed_with_detail<
    T: FiniteFrameValue,
    R: RnntTransition<T>,
    F: RnntFrameSource<T>,
>(
    transition: &mut R,
    frames: &F,
    blank: usize,
    max_symbols_per_step: usize,
    scored: bool,
) -> Result<RnntOutput, String> {
    let mut state = transition.start(blank)?;
    let output_frames = frames.output_frames();
    let mut tokens = Vec::new();
    let mut silence = Vec::with_capacity(output_frames);
    for frame_index in 0..output_frames {
        let frame = frames.frame(frame_index);
        validate_encoder_frame(frame, frame_index)?;
        transition.prepare_frame(frame)?;
        let mut emitted = false;
        for _ in 0..max_symbols_per_step {
            let (token, score) = if scored {
                let (token, score) = transition.select_scored(frame, &state)?;
                (token, Some(score))
            } else {
                (transition.select(frame, &state)?, None)
            };
            if token > blank {
                return Err(format!(
                    "RNN-T selected token {token} exceeds blank vocabulary boundary {blank}"
                ));
            }
            if token == blank {
                break;
            }
            tokens.push(match score {
                Some(value) => Token::scored(token, frame_index, value)?,
                None => Token::new(token, frame_index),
            });
            emitted = true;
            transition.advance(token, &mut state)?;
        }
        silence.push(!emitted);
    }
    Ok(RnntOutput { tokens, silence })
}

fn validate_encoder_frame<T: FiniteFrameValue>(
    frame: &[T],
    frame_index: usize,
) -> Result<(), String> {
    for (value_index, value) in frame.iter().enumerate() {
        if !value.is_finite_value() {
            return Err(format!(
                "RNN-T encoder value at frame {frame_index}, position {value_index} must be finite"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeTransition {
        selections: Vec<usize>,
        next_selection: usize,
        starts: Vec<usize>,
        advances: Vec<usize>,
    }

    impl RnntTransition for FakeTransition {
        type State = usize;

        fn start(&mut self, blank: usize) -> Result<Self::State, String> {
            self.starts.push(blank);
            Ok(blank)
        }

        fn select(&mut self, encoder_frame: &[f32], _state: &Self::State) -> Result<usize, String> {
            if encoder_frame.is_empty() {
                return Err("fake transition requires a nonempty encoder frame".into());
            }
            let token = *self
                .selections
                .get(self.next_selection)
                .ok_or_else(|| "fake transition ran out of selections".to_owned())?;
            self.next_selection = self
                .next_selection
                .checked_add(1)
                .ok_or_else(|| "fake transition selection index overflows usize".to_owned())?;
            Ok(token)
        }

        fn advance(&mut self, token: usize, state: &mut Self::State) -> Result<(), String> {
            self.advances.push(token);
            *state = token;
            Ok(())
        }
    }

    struct FakeFrames(Vec<Vec<f32>>);

    impl RnntFrameSource for FakeFrames {
        fn output_frames(&self) -> usize {
            self.0.len()
        }

        fn frame(&self, index: usize) -> &[f32] {
            &self.0[index]
        }
    }

    #[test]
    fn greedy_transitions_emit_until_blank_and_keep_frame_order() {
        let mut transition = FakeTransition {
            selections: vec![1, 3, 2, 3],
            next_selection: 0,
            starts: Vec::new(),
            advances: Vec::new(),
        };
        let frames = FakeFrames(vec![vec![10.0], vec![20.0]]);
        let output =
            greedy(&mut transition, &frames, 3, 3).expect("fake RNN-T transition is complete");
        assert_eq!(output.tokens(), &[Token::new(1, 0), Token::new(2, 1)]);
        assert_eq!(output.silence(), &[false, false]);
        assert_eq!(transition.starts, vec![3]);
        assert_eq!(transition.advances, vec![1, 2]);
    }

    #[test]
    fn symbol_limit_prevents_an_unbounded_transition_loop() {
        let mut transition = FakeTransition {
            selections: vec![1, 2, 3],
            next_selection: 0,
            starts: Vec::new(),
            advances: Vec::new(),
        };
        let frames = FakeFrames(vec![vec![10.0]]);
        let output = greedy(&mut transition, &frames, 3, 2)
            .expect("the bounded fake RNN-T transition is valid");
        assert_eq!(output.tokens(), &[Token::new(1, 0), Token::new(2, 0)]);
        assert_eq!(transition.advances, vec![1, 2]);
    }

    #[test]
    fn greedy_refuses_nonfinite_encoder_values_before_selecting_each_frame() {
        for (name, invalid) in [
            ("NaN", f32::NAN),
            ("positive infinity", f32::INFINITY),
            ("negative infinity", f32::NEG_INFINITY),
        ] {
            for frame_index in [0, 1, 2] {
                for value_index in [0, 1, 2] {
                    let mut frames = FakeFrames(vec![vec![10.0, 20.0, 30.0]; 3]);
                    frames.0[frame_index][value_index] = invalid;
                    let mut transition = FakeTransition {
                        selections: vec![0, 0, 0],
                        next_selection: 0,
                        starts: Vec::new(),
                        advances: Vec::new(),
                    };

                    assert!(
                        greedy(&mut transition, &frames, 0, 1).is_err(),
                        "{name} at frame {frame_index}, value {value_index} must refuse"
                    );
                    assert_eq!(
                        transition.next_selection, frame_index,
                        "the invalid frame must not reach the transition selection port"
                    );
                    assert!(
                        transition.advances.is_empty(),
                        "a nonfinite frame must not advance recurrent state"
                    );
                }
            }
        }
    }

    #[test]
    fn greedy_refuses_a_selected_token_outside_the_blank_domain_before_advance() {
        let mut transition = FakeTransition {
            selections: vec![3],
            next_selection: 0,
            starts: Vec::new(),
            advances: Vec::new(),
        };
        let frames = FakeFrames(vec![vec![10.0]]);

        assert!(greedy(&mut transition, &frames, 2, 1).is_err());
        assert!(
            transition.advances.is_empty(),
            "an out-of-domain selected token must not mutate recurrent state"
        );
    }
    #[test]
    fn scored_multiple_symbols_keep_each_actual_joint_step() -> Result<(), String> {
        struct Scored {
            step: usize,
        }
        impl RnntTransition for Scored {
            type State = ();
            fn start(&mut self, _: usize) -> Result<(), String> {
                Ok(())
            }
            fn select(&mut self, _: &[f32], _: &()) -> Result<usize, String> {
                Err("unexpected unscored selection".into())
            }
            fn select_scored(&mut self, _: &[f32], _: &()) -> Result<(usize, f64), String> {
                let result = match self.step {
                    0 => (0, -0.25),
                    1 => (1, -0.5),
                    _ => (2, -0.75),
                };
                self.step += 1;
                Ok(result)
            }
            fn advance(&mut self, _: usize, _: &mut ()) -> Result<(), String> {
                Ok(())
            }
        }
        let mut transition = Scored { step: 0 };
        let output = greedy_with_detail(&mut transition, &FakeFrames(vec![vec![1.0]]), 2, 4, true)?;
        assert_eq!(output.tokens().len(), 2);
        assert_eq!(output.tokens()[0].frame(), 0);
        assert_eq!(output.tokens()[1].frame(), 0);
        assert_eq!(output.tokens()[0].log_probability(), Some(-0.25));
        assert_eq!(output.tokens()[1].log_probability(), Some(-0.5));
        Ok(())
    }
    #[test]
    fn typed_frames_prepare_once_and_preserve_state_for_both_precisions() -> Result<(), String> {
        struct Frames<T>(Vec<T>);
        impl<T> RnntFrameSource<T> for Frames<T> {
            fn output_frames(&self) -> usize {
                1
            }
            fn frame(&self, _: usize) -> &[T] {
                &self.0
            }
        }
        struct Transition {
            preparations: usize,
            seen: Vec<usize>,
        }
        impl<T> RnntTransition<T> for Transition {
            type State = usize;
            fn start(&mut self, _: usize) -> Result<usize, String> {
                Ok(0)
            }
            fn prepare_frame(&mut self, _: &[T]) -> Result<(), String> {
                self.preparations += 1;
                Ok(())
            }
            fn select(&mut self, _: &[T], state: &usize) -> Result<usize, String> {
                self.seen.push(*state);
                Ok(if *state < 2 { *state } else { 2 })
            }
            fn advance(&mut self, _: usize, state: &mut usize) -> Result<(), String> {
                *state += 1;
                Ok(())
            }
        }
        fn check<T: FiniteFrameValue>(value: T) -> Result<(), String> {
            let mut transition = Transition {
                preparations: 0,
                seen: Vec::new(),
            };
            let output =
                greedy_typed_with_detail(&mut transition, &Frames(vec![value]), 2, 4, false)?;
            assert_eq!(output.tokens(), &[Token::new(0, 0), Token::new(1, 0)]);
            assert_eq!(transition.preparations, 1);
            assert_eq!(transition.seen, [0, 1, 2]);
            Ok(())
        }
        check(1.0_f32)?;
        check(half::f16::from_f32(1.0))?;
        let mut transition = Transition {
            preparations: 0,
            seen: Vec::new(),
        };
        assert!(
            greedy_typed_with_detail(&mut transition, &Frames(vec![half::f16::NAN]), 2, 4, false)
                .is_err()
        );
        assert_eq!(transition.preparations, 0);
        Ok(())
    }
}
