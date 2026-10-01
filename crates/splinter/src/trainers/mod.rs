//! Trainers: learn a model from a text corpus.
//!
//! Use them through [`crate::Tokenizer::train`] (or
//! [`crate::Tokenizer::train_from_files`]) so the corpus goes through the
//! tokenizer's own normalizer and pre-tokenizer first.

pub mod bpe;
pub mod unigram;
pub mod wordlevel;
pub mod wordpiece;

use crate::added_vocabulary::AddedToken;
use crate::error::Result;
use crate::models::{Bpe, ModelWrapper, Unigram, WordLevel, WordPiece};
use crate::traits::Trainer;

pub use bpe::{BpeTrainer, BpeTrainerBuilder};
pub use unigram::{UnigramTrainer, UnigramTrainerBuilder};
pub use wordlevel::{WordLevelTrainer, WordLevelTrainerBuilder};
pub use wordpiece::{WordPieceTrainer, WordPieceTrainerBuilder};

/// Any built-in trainer.
#[derive(Debug, Clone)]
pub enum TrainerWrapper {
    /// Trains a [`Bpe`] model.
    Bpe(BpeTrainer),
    /// Trains a [`WordPiece`] model.
    WordPiece(WordPieceTrainer),
    /// Trains a [`WordLevel`] model.
    WordLevel(WordLevelTrainer),
    /// Trains a [`Unigram`] model.
    Unigram(UnigramTrainer),
}

impl Trainer for TrainerWrapper {
    type Model = ModelWrapper;

    fn should_show_progress(&self) -> bool {
        match self {
            TrainerWrapper::Bpe(t) => t.should_show_progress(),
            TrainerWrapper::WordPiece(t) => t.should_show_progress(),
            TrainerWrapper::WordLevel(t) => t.should_show_progress(),
            TrainerWrapper::Unigram(t) => t.should_show_progress(),
        }
    }

    /// Train, replacing `model` with a model of the trainer's kind if it
    /// is a different kind.
    fn train(&self, model: &mut ModelWrapper) -> Result<Vec<AddedToken>> {
        macro_rules! train_as {
            ($trainer:expr, $variant:ident, $ty:ty) => {{
                if !matches!(model, ModelWrapper::$variant(_)) {
                    *model = ModelWrapper::$variant(<$ty>::default());
                }
                match model {
                    ModelWrapper::$variant(m) => $trainer.train(m),
                    _ => unreachable!("model replaced above"),
                }
            }};
        }
        match self {
            TrainerWrapper::Bpe(t) => train_as!(t, Bpe, Bpe),
            TrainerWrapper::WordPiece(t) => train_as!(t, WordPiece, WordPiece),
            TrainerWrapper::WordLevel(t) => train_as!(t, WordLevel, WordLevel),
            TrainerWrapper::Unigram(t) => train_as!(t, Unigram, Unigram),
        }
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        match self {
            TrainerWrapper::Bpe(t) => t.feed(iterator, process),
            TrainerWrapper::WordPiece(t) => t.feed(iterator, process),
            TrainerWrapper::WordLevel(t) => t.feed(iterator, process),
            TrainerWrapper::Unigram(t) => t.feed(iterator, process),
        }
    }
}

impl From<BpeTrainer> for TrainerWrapper {
    fn from(t: BpeTrainer) -> Self {
        TrainerWrapper::Bpe(t)
    }
}

impl From<WordPieceTrainer> for TrainerWrapper {
    fn from(t: WordPieceTrainer) -> Self {
        TrainerWrapper::WordPiece(t)
    }
}

impl From<WordLevelTrainer> for TrainerWrapper {
    fn from(t: WordLevelTrainer) -> Self {
        TrainerWrapper::WordLevel(t)
    }
}

impl From<UnigramTrainer> for TrainerWrapper {
    fn from(t: UnigramTrainer) -> Self {
        TrainerWrapper::Unigram(t)
    }
}
