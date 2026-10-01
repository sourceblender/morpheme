//! WordPiece model — greedy longest-match subword lookup.
//!
//! Given a word and a vocabulary, the algorithm:
//!
//! 1. Greedily find the longest vocab entry that matches the start
//!    of the remaining substring.
//! 2. Emit it as a token. If no match, emit `<unk>`.
//! 3. Continue with the rest of the word.
//!
//! This is the BERT-style greedy WordPiece used at inference time
//! once a model is trained. It pairs with [`crate::decoder::WordPieceDecoder`].

use crate::error::{Error as SplinterError, Result};
use crate::model::Model;
use crate::pre_tokenizer::PreToken;
use crate::vocab::Vocab;

/// WordPiece model with greedy longest-match subword lookup.
#[derive(Debug, Clone)]
pub struct WordPiece {
    /// Vocabulary mapping `token -> id`.
    pub vocab: Vocab,
    /// Continuing-subword prefix. BERT uses `##`.
    pub continuing_subword_prefix: String,
    /// End-of-word marker. Appended to the last character of every
    /// word when matching against the vocab. Default `</w>`.
    pub end_of_word_suffix: String,
    /// Token to emit for out-of-vocabulary characters. Default `<unk>`.
    pub unk_token: String,
    /// Maximum input chars per word before falling back to unk.
    pub max_input_chars_per_word: usize,
}

impl WordPiece {
    /// Construct a WordPiece model with the given config.
    pub fn new(
        vocab: Vocab,
        continuing_subword_prefix: impl Into<String>,
        unk_token: impl Into<String>,
        max_input_chars_per_word: usize,
    ) -> Self {
        Self::with_end_of_word_suffix(
            vocab,
            continuing_subword_prefix,
            "</w>",
            unk_token,
            max_input_chars_per_word,
        )
    }

    /// Construct with an explicit end-of-word suffix.
    pub fn with_end_of_word_suffix(
        vocab: Vocab,
        continuing_subword_prefix: impl Into<String>,
        end_of_word_suffix: impl Into<String>,
        unk_token: impl Into<String>,
        max_input_chars_per_word: usize,
    ) -> Self {
        Self {
            vocab,
            continuing_subword_prefix: continuing_subword_prefix.into(),
            end_of_word_suffix: end_of_word_suffix.into(),
            unk_token: unk_token.into(),
            max_input_chars_per_word,
        }
    }

    /// Apply WordPiece to a single pre-token string.
    fn apply(&self, word: &str) -> Result<Vec<String>> {
        if word.is_empty() {
            return Ok(Vec::new());
        }
        let chars: Vec<char> = word.chars().collect();
        if chars.len() > self.max_input_chars_per_word {
            return Err(SplinterError::UnknownToken(self.unk_token.clone()));
        }

        let mut out: Vec<String> = Vec::new();
        let mut start = 0usize;
        while start < chars.len() {
            let mut end = chars.len();
            let mut cur_str: Option<String> = None;
            while end > start {
                let mut candidate = String::new();
                // Build the candidate. At a fresh word (start == 0),
                // no char gets the continuing prefix. At a
                // continuation (start > 0), only the FIRST char of
                // the candidate gets `##` — subsequent chars don't.
                let prefix_first = start > 0;
                if prefix_first {
                    candidate.push_str(&self.continuing_subword_prefix);
                }
                let first = chars[start];
                candidate.push(first);
                for &c in chars.iter().skip(start + 1).take(end - start - 1) {
                    candidate.push(c);
                }
                // The end-of-word suffix is appended to candidates
                // that span the final char of the word.
                if end == chars.len() {
                    candidate.push_str(&self.end_of_word_suffix);
                }
                if self.vocab.token_to_id(&candidate).is_some() {
                    cur_str = Some(candidate);
                    break;
                }
                end -= 1;
            }
            match cur_str {
                Some(s) => {
                    out.push(s);
                    // Advance `start` past the matched span.
                    start = end;
                }
                None => {
                    // OOV: emit the unk token. If unk itself isn't in
                    // the vocab, surface that as an error.
                    if self.vocab.token_to_id(&self.unk_token).is_some() {
                        out.push(self.unk_token.clone());
                    } else {
                        return Err(SplinterError::UnknownToken(self.unk_token.clone()));
                    }
                    start = chars.len();
                }
            }
        }
        Ok(out)
    }
}

impl Model for WordPiece {
    fn tokenize(&self, pre_token: PreToken<'_>) -> Result<Vec<String>> {
        self.apply(pre_token.text.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build() -> WordPiece {
        let tokens: Vec<String> = vec![
            "<unk>".into(),
            "h".into(),
            "##e".into(),
            "##l".into(),
            "##o</w>".into(),
            "hell</w>".into(),
            "hello</w>".into(),
        ];
        let vocab = Vocab::from_tokens(tokens).unwrap();
        WordPiece::new(vocab, "##", "<unk>", 100)
    }

    #[test]
    fn greedy_match_longest() {
        let m = build();
        let toks = m.apply("hello").unwrap();
        assert_eq!(toks, vec!["hello</w>"]);
    }

    #[test]
    fn greedy_falls_back_to_subwords() {
        let m = build();
        let toks = m.apply("hell").unwrap();
        assert_eq!(toks, vec!["hell</w>"]);
    }

    #[test]
    fn unknown_char_emits_unk() {
        let m = build();
        // 'x' is not in vocab. WordPiece emits `<unk>` for OOV.
        // 'h' and '##e' both match; '##x' does not.
        let toks = m.apply("hex").unwrap();
        assert_eq!(toks, vec!["h", "##e", "<unk>"]);
    }
}
