//! The five pipeline component traits.
//!
//! A [`crate::Tokenizer`] runs, in order: added-token extraction →
//! [`Normalizer`] → [`PreTokenizer`] → [`Model`] → truncation →
//! [`PostProcessor`] → padding. [`Decoder`] turns tokens back into text.

use std::collections::HashMap;

use crate::Token;
use crate::encoding::Encoding;
use crate::error::Result;
use crate::normalized_string::NormalizedString;
use crate::pre_tokenized_string::PreTokenizedString;

/// Rewrites text in place, keeping alignments to the original.
pub trait Normalizer {
    /// Normalize `normalized` in place.
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()>;
}

/// Splits text into pieces ("words") that the model tokenizes
/// independently.
pub trait PreTokenizer {
    /// Split (and possibly rewrite) the pieces of `pretokenized`.
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()>;
}

/// Maps a piece of text to tokens.
pub trait Model {
    /// Tokenize `sequence`. Token offsets are byte offsets into
    /// `sequence`.
    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>>;
    /// Id of `token`, if it is in the vocabulary.
    fn token_to_id(&self, token: &str) -> Option<u32>;
    /// Token for `id`, if it is in the vocabulary.
    fn id_to_token(&self, id: u32) -> Option<String>;
    /// The full vocabulary.
    fn get_vocab(&self) -> HashMap<String, u32>;
    /// Vocabulary size.
    fn get_vocab_size(&self) -> usize;
}

/// Adds special tokens and combines sequence pairs.
pub trait PostProcessor {
    /// How many tokens `process` adds for a single input (`false`) or a
    /// pair (`true`) when `add_special_tokens` is on.
    fn added_tokens(&self, is_pair: bool) -> usize;

    /// Process each encoding (one per input sequence). Implementations
    /// may merge them into a single encoding.
    fn process_encodings(
        &self,
        encodings: Vec<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>>;

    /// Process a single encoding or a pair and merge the result.
    ///
    /// Each sequence first gets its sequence id and a uniform type id
    /// (`0` for the first, `1` for the second).
    fn process(
        &self,
        encoding: Encoding,
        pair_encoding: Option<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        let mut encodings = vec![encoding];
        encodings.extend(pair_encoding);
        for (i, e) in encodings.iter_mut().enumerate() {
            e.set_sequence_id(i);
            for o in e.overflowing_mut() {
                o.set_sequence_id(i);
            }
            e.set_type_ids(vec![i as u32; e.len()]);
        }
        let encodings = self.process_encodings(encodings, add_special_tokens)?;
        Ok(Encoding::merge(encodings, false))
    }
}

/// Turns tokens back into text.
pub trait Decoder {
    /// Transform each token; the final text is the concatenation.
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>>;

    /// Decode `tokens` into a string.
    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        Ok(self.decode_chain(tokens)?.join(""))
    }
}

/// Learns a model from text.
///
/// Training is two-phase: [`feed`](Trainer::feed) receives the corpus
/// (already normalized and pre-tokenized by the tokenizer, via
/// `process`), then [`train`](Trainer::train) builds the model.
pub trait Trainer {
    /// The model this trainer produces.
    type Model: Model + Sized;

    /// Whether to report progress.
    fn should_show_progress(&self) -> bool {
        false
    }

    /// Train `model` in place from everything fed so far. Returns the
    /// special tokens the tokenizer must register as added tokens.
    fn train(&self, model: &mut Self::Model) -> Result<Vec<crate::AddedToken>>;

    /// Consume a corpus. `process` turns one input sequence into the
    /// words (pre-tokens) to count.
    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync;
}
