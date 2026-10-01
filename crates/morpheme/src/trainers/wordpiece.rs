//! WordPiece trainer. Like Hugging Face `tokenizers`, it trains a BPE
//! model with a `##` continuing-subword prefix and keeps its vocabulary.

use std::collections::HashSet;

use crate::added_vocabulary::AddedToken;
use crate::error::Result;
use crate::models::{Bpe, WordPiece};
use crate::trainers::bpe::{BpeTrainer, BpeTrainerBuilder};
use crate::traits::Trainer;

/// Builder for [`WordPieceTrainer`].
#[derive(Debug, Clone)]
pub struct WordPieceTrainerBuilder {
    bpe: BpeTrainerBuilder,
}

impl Default for WordPieceTrainerBuilder {
    fn default() -> Self {
        Self {
            bpe: BpeTrainerBuilder::new().continuing_subword_prefix("##"),
        }
    }
}

impl WordPieceTrainerBuilder {
    /// A builder with HF defaults (vocab size 30 000, prefix `##`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Pairs occurring fewer times than this are never merged.
    #[must_use]
    pub fn min_frequency(mut self, n: u64) -> Self {
        self.bpe = self.bpe.min_frequency(n);
        self
    }

    /// Target vocabulary size.
    #[must_use]
    pub fn vocab_size(mut self, n: usize) -> Self {
        self.bpe = self.bpe.vocab_size(n);
        self
    }

    /// Whether to report progress.
    #[must_use]
    pub fn show_progress(mut self, v: bool) -> Self {
        self.bpe = self.bpe.show_progress(v);
        self
    }

    /// Special tokens, placed first in the vocabulary.
    #[must_use]
    pub fn special_tokens(mut self, tokens: Vec<AddedToken>) -> Self {
        self.bpe = self.bpe.special_tokens(tokens);
        self
    }

    /// Maximum alphabet size.
    #[must_use]
    pub fn limit_alphabet(mut self, n: usize) -> Self {
        self.bpe = self.bpe.limit_alphabet(n);
        self
    }

    /// Chars always included in the alphabet.
    #[must_use]
    pub fn initial_alphabet(mut self, alphabet: HashSet<char>) -> Self {
        self.bpe = self.bpe.initial_alphabet(alphabet);
        self
    }

    /// Prefix of non-initial subwords (default `##`).
    #[must_use]
    pub fn continuing_subword_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.bpe = self.bpe.continuing_subword_prefix(prefix);
        self
    }

    /// Suffix of word-final subwords.
    #[must_use]
    pub fn end_of_word_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.bpe = self.bpe.end_of_word_suffix(suffix);
        self
    }

    /// Build the trainer.
    ///
    /// # Errors
    /// Fails if `vocab_size` or `limit_alphabet` is zero.
    pub fn build(self) -> Result<WordPieceTrainer> {
        Ok(WordPieceTrainer {
            bpe: self.bpe.build()?,
        })
    }
}

/// Trains a [`WordPiece`] model.
///
/// # Example
///
/// ```
/// use morpheme::models::WordPiece;
/// use morpheme::pre_tokenizers::BertPreTokenizer;
/// use morpheme::trainers::WordPieceTrainer;
/// use morpheme::{AddedToken, Tokenizer};
///
/// let mut tokenizer = Tokenizer::new(WordPiece::default()).with_pre_tokenizer(BertPreTokenizer);
/// let trainer = WordPieceTrainer::builder()
///     .vocab_size(60)
///     .special_tokens(vec![AddedToken::new("[UNK]", true)])
///     .show_progress(false)
///     .build()?;
/// tokenizer.train(trainer, ["playing played player", "plays"].into_iter())?;
///
/// let tokens = tokenizer.encode("replay", false)?.tokens().to_vec();
/// assert!(tokens.iter().skip(1).all(|t| t.starts_with("##")), "{tokens:?}");
/// # Ok::<(), morpheme::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct WordPieceTrainer {
    bpe: BpeTrainer,
}

/// The same trainer as `WordPieceTrainer::builder().build()`: HF defaults
/// with the `##` continuation prefix. (Deriving `Default` would wrap
/// `BpeTrainer::default()`, which has no prefix and would train a
/// vocabulary without any `##` tokens.)
impl Default for WordPieceTrainer {
    fn default() -> Self {
        WordPieceTrainerBuilder::default()
            .build()
            .expect("the default configuration is valid")
    }
}

impl WordPieceTrainer {
    /// Start building a trainer.
    pub fn builder() -> WordPieceTrainerBuilder {
        WordPieceTrainerBuilder::new()
    }
}

impl Trainer for WordPieceTrainer {
    type Model = WordPiece;

    fn should_show_progress(&self) -> bool {
        self.bpe.show_progress
    }

    fn train(&self, model: &mut WordPiece) -> Result<Vec<AddedToken>> {
        let mut bpe = Bpe::default();
        let special = self.bpe.train(&mut bpe)?;
        let trained = WordPiece::from_bpe(&bpe);
        model.set_vocab(crate::traits::Model::vocab(&trained));
        model.continuing_subword_prefix = trained.continuing_subword_prefix;
        Ok(special)
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        self.bpe.feed(iterator, process)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainers::bpe::tests::whitespace_words;
    use crate::traits::Model;

    #[test]
    fn trained_wordpiece_tokenizes_corpus() {
        let mut t = WordPieceTrainer::builder()
            .vocab_size(60)
            .show_progress(false)
            .special_tokens(vec![AddedToken::new("[UNK]", true)])
            .build()
            .unwrap();
        t.feed(
            ["unaffable affable unable able", "hello world hello"].into_iter(),
            whitespace_words,
        )
        .unwrap();
        let mut m = WordPiece::default();
        t.train(&mut m).unwrap();
        assert_eq!(m.token_to_id("[UNK]"), Some(0));
        assert!(m.token_to_id("##e").is_some(), "continuation tokens exist");
        // Every corpus word tokenizes without unk.
        for w in ["unaffable", "hello", "world", "able"] {
            let toks = m.tokenize(w).unwrap();
            assert!(toks.iter().all(|t| t.value != "[UNK]"), "{w}: {toks:?}");
            let rebuilt: String = toks
                .iter()
                .map(|t| t.value.trim_start_matches("##"))
                .collect();
            assert_eq!(rebuilt, w);
        }
        assert_eq!(m.tokenize("xyz").unwrap()[0].value, "[UNK]");
    }

    #[test]
    fn default_equals_builder_and_trains_continuation_tokens() {
        let mut t = WordPieceTrainer::default();
        assert_eq!(
            t.bpe.continuing_subword_prefix.as_deref(),
            Some("##"),
            "Default must equal builder().build()"
        );
        t.feed(
            ["playing played player plays"].into_iter(),
            whitespace_words,
        )
        .unwrap();
        let mut m = WordPiece::default();
        t.train(&mut m).unwrap();
        assert!(
            m.vocab().keys().any(|k| k.starts_with("##")),
            "no ## tokens in {:?}",
            m.vocab().keys().collect::<Vec<_>>()
        );
        // Every corpus word tokenizes (no [UNK] needed) and rebuilds.
        for w in ["playing", "played", "player", "plays"] {
            let toks = m.tokenize(w).unwrap();
            let rebuilt: String = toks
                .iter()
                .map(|t| t.value.trim_start_matches("##"))
                .collect();
            assert_eq!(rebuilt, w, "{toks:?}");
        }
    }

    #[test]
    fn matches_hf_vocab_as_a_set() {
        // Python tokenizers 0.23.2: WordPiece model + Whitespace
        // pre-tokenizer + WordPieceTrainer(vocab_size=25,
        // special_tokens=["[UNK]"]). HF assigns `##x` ids in hash-map
        // order, so only the token set is stable.
        let mut t = WordPieceTrainer::builder()
            .vocab_size(25)
            .show_progress(false)
            .special_tokens(vec![AddedToken::new("[UNK]", true)])
            .build()
            .unwrap();
        t.feed(["hello hello help"].into_iter(), whitespace_words)
            .unwrap();
        let mut m = WordPiece::default();
        t.train(&mut m).unwrap();
        let mut got: Vec<String> = m.vocab().into_keys().collect();
        got.sort();
        let mut want: Vec<String> = HF_WORDPIECE_VOCAB.iter().map(|s| s.to_string()).collect();
        want.sort();
        assert_eq!(got, want);
    }

    const HF_WORDPIECE_VOCAB: &[&str] = &[
        "##e", "##l", "##lo", "##o", "##p", "[UNK]", "e", "h", "he", "hel", "hello", "help", "l",
        "o", "p",
    ];
}
