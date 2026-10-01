//! Trainers — turn a corpus into a model.
//!
//! Phase 2 ships [`BpeTrainer`]. Phase 2.1 adds [`WordPieceTrainer`]
//! and [`UnigramTrainer`].

pub mod bpe;
pub mod unigram;
pub mod wordpiece;

use crate::error::Result;
use crate::vocab::Vocab;

/// Output of the BPE trainer. WordPiece and Unigram trainers return
/// their own concrete types since the model-specific data is
/// different (no merges, just a vocab + continuing prefix).
#[derive(Debug, Clone)]
pub struct TrainedModel {
    /// Ordered vocabulary. Id `i` is the token at index `i`.
    pub vocab: Vocab,
    /// Ordered merge list. Index 0 is the highest-priority merge.
    pub merges: Vec<(String, String)>,
    /// End-of-word marker, e.g. `</w>`.
    pub end_of_word_suffix: String,
}

/// Input corpus for a trainer — anything we can iterate over line by
/// line. The trainer doesn't care whether the lines come from a file,
/// a network stream, or an in-memory `Vec<String>`.
pub trait Corpus {
    /// Iterate over the lines of the corpus.
    fn for_each_line<F: FnMut(&str)>(&self, f: F);
}

impl Corpus for &str {
    fn for_each_line<F: FnMut(&str)>(&self, mut f: F) {
        for line in self.lines() {
            f(line);
        }
    }
}

impl Corpus for String {
    fn for_each_line<F: FnMut(&str)>(&self, f: F) {
        self.as_str().for_each_line(f);
    }
}

impl Corpus for std::path::PathBuf {
    fn for_each_line<F: FnMut(&str)>(&self, mut f: F) {
        let bytes = match std::fs::read(self) {
            Ok(b) => b,
            Err(_) => return,
        };
        let s = match std::str::from_utf8(&bytes) {
            Ok(s) => s,
            Err(_) => return,
        };
        for line in s.lines() {
            f(line);
        }
    }
}

impl Corpus for &[&str] {
    fn for_each_line<F: FnMut(&str)>(&self, mut f: F) {
        for line in self.iter() {
            f(line);
        }
    }
}

/// Common interface for trainers that produce a [`TrainedModel`] (BPE
/// only in v0.1). WordPiece and Unigram trainers expose their own
/// concrete `train(...)` method since their output shape is
/// different.
pub trait Trainer {
    /// Train and return the resulting model.
    fn train<C: Corpus>(&self, corpus: C) -> Result<TrainedModel>;
}

pub use bpe::{BpeTrainer, BpeTrainerBuilder};
pub use unigram::{UnigramTrainedModel, UnigramTrainer, UnigramTrainerBuilder};
pub use wordpiece::{WordPieceTrainedModel, WordPieceTrainer, WordPieceTrainerBuilder};
