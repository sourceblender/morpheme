//! WordLevel trainer: keep the most frequent words.

use std::collections::HashMap;

use crate::added_vocabulary::AddedToken;
use crate::error::{Error, Result};
use crate::models::WordLevel;
use crate::trainers::bpe::count_words;
use crate::traits::Trainer;

/// Builder for [`WordLevelTrainer`].
#[derive(Debug, Clone)]
pub struct WordLevelTrainerBuilder {
    trainer: WordLevelTrainer,
}

impl Default for WordLevelTrainerBuilder {
    fn default() -> Self {
        Self {
            trainer: WordLevelTrainer {
                min_frequency: 0,
                vocab_size: 30_000,
                show_progress: true,
                special_tokens: Vec::new(),
                words: HashMap::new(),
            },
        }
    }
}

impl WordLevelTrainerBuilder {
    /// A builder with HF defaults (vocab size 30 000).
    pub fn new() -> Self {
        Self::default()
    }

    /// Words occurring fewer times than this are dropped.
    #[must_use]
    pub fn min_frequency(mut self, n: u64) -> Self {
        self.trainer.min_frequency = n;
        self
    }

    /// Target vocabulary size (special tokens included).
    #[must_use]
    pub fn vocab_size(mut self, n: usize) -> Self {
        self.trainer.vocab_size = n;
        self
    }

    /// Whether to report progress.
    #[must_use]
    pub fn show_progress(mut self, v: bool) -> Self {
        self.trainer.show_progress = v;
        self
    }

    /// Special tokens, placed first in the vocabulary.
    #[must_use]
    pub fn special_tokens(mut self, tokens: Vec<AddedToken>) -> Self {
        self.trainer.special_tokens = tokens;
        self
    }

    /// Build the trainer.
    ///
    /// # Errors
    /// Fails if `vocab_size` is zero.
    pub fn build(self) -> Result<WordLevelTrainer> {
        if self.trainer.vocab_size == 0 {
            return Err(Error::Config(
                "WordLevelTrainer: vocab_size must be > 0".into(),
            ));
        }
        Ok(self.trainer)
    }
}

/// Trains a [`WordLevel`] model.
///
/// # Example
///
/// ```
/// use splinter::models::WordLevel;
/// use splinter::pre_tokenizers::Whitespace;
/// use splinter::trainers::WordLevelTrainer;
/// use splinter::{AddedToken, Tokenizer};
///
/// let model = WordLevel::builder().unk_token("[UNK]").build()?;
/// let mut tokenizer = Tokenizer::new(model).with_pre_tokenizer(Whitespace);
/// let trainer = WordLevelTrainer::builder()
///     .special_tokens(vec![AddedToken::new("[UNK]", true)])
///     .show_progress(false)
///     .build()?;
/// tokenizer.train(trainer, ["the cat", "the dog"].into_iter())?;
///
/// assert_eq!(tokenizer.encode("the bird", false)?.tokens(), ["the", "[UNK]"]);
/// # Ok::<(), splinter::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct WordLevelTrainer {
    /// Minimum word frequency.
    pub(crate) min_frequency: u64,
    /// Target vocabulary size.
    pub(crate) vocab_size: usize,
    /// Whether to report progress.
    pub(crate) show_progress: bool,
    /// Special tokens, placed first in the vocabulary.
    pub(crate) special_tokens: Vec<AddedToken>,
    words: HashMap<String, u64>,
}

impl Default for WordLevelTrainer {
    fn default() -> Self {
        WordLevelTrainerBuilder::default()
            .build()
            .expect("the default configuration is valid")
    }
}

impl WordLevelTrainer {
    /// Start building a trainer.
    pub fn builder() -> WordLevelTrainerBuilder {
        WordLevelTrainerBuilder::new()
    }

    fn do_train(&self, word_counts: &HashMap<String, u64>, model: &mut WordLevel) {
        let mut ordered: Vec<(&String, u64)> = word_counts.iter().map(|(w, c)| (w, *c)).collect();
        // Most frequent first; ties alphabetical.
        ordered.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        let vocab: HashMap<String, u32> = self
            .special_tokens
            .iter()
            .map(|t| t.content.clone())
            .chain(
                ordered
                    .into_iter()
                    .filter(|(_, n)| *n >= self.min_frequency)
                    .map(|(w, _)| w.clone()),
            )
            .take(self.vocab_size)
            .enumerate()
            .map(|(i, w)| (w, i as u32))
            .collect();
        model.set_vocab(vocab);
    }
}

impl Trainer for WordLevelTrainer {
    type Model = WordLevel;

    fn should_show_progress(&self) -> bool {
        self.show_progress
    }

    fn train(&self, model: &mut WordLevel) -> Result<Vec<AddedToken>> {
        self.do_train(&self.words, model);
        Ok(self.special_tokens.clone())
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        self.words = count_words(iterator, process, self.show_progress)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::Model;

    #[test]
    fn keeps_most_frequent_words() {
        // Same data as the HF unit test.
        let counts: HashMap<String, u64> = [
            ("the", 25),
            ("roses", 22),
            ("are", 24),
            ("red", 12),
            ("voilets", 10),
            ("blue", 16),
        ]
        .iter()
        .map(|(w, c)| (w.to_string(), *c))
        .collect();
        let mut trainer = WordLevelTrainer::builder().vocab_size(5).build().unwrap();
        let mut model = WordLevel::default();
        trainer.do_train(&counts, &mut model);
        let expected: HashMap<String, u32> = [
            ("the", 0),
            ("are", 1),
            ("roses", 2),
            ("blue", 3),
            ("red", 4),
        ]
        .iter()
        .map(|(w, i)| (w.to_string(), *i))
        .collect();
        assert_eq!(model.vocab(), expected);

        trainer.min_frequency = 15;
        trainer.do_train(&counts, &mut model);
        assert_eq!(model.vocab_size(), 4);
    }
}
