//! Unigram (SentencePiece-style) model — Viterbi best-path over a
//! subword lattice.
//!
//! Each subword has a log-probability; tokenization finds the
//! segmentation of a word that maximizes the total log-prob. We run
//! Viterbi (forward + backtrace) over the character lattice, with
//! `min_score` as a floor — anything at or below this is treated as
//! unknown and replaced with `unk_token`.
//!
//! ## Performance
//!
//! Matching vocab entries against the input word uses a trie built
//! once per `apply` call. The trie finds all candidate (start, end,
//! id) matches in `O(n + z)` where `z` is the number of matches —
//! roughly linear in the input length plus the total number of
//! vocabulary occurrences.

use crate::error::{Error as SplinterError, Result};
use crate::model::Model;
use crate::pre_tokenizer::PreToken;
use crate::vocab::Vocab;

/// Node in the subword trie. `children` maps a character to the
/// child node index. `terminals` are the vocab ids of subwords that
/// end at this node.
#[derive(Default)]
struct TrieNode {
    children: Vec<(char, usize)>,
    terminals: Vec<u32>,
}

/// Subword trie for fast candidate matching. Built lazily from the
/// vocab on each `apply` call.
struct SubwordTrie {
    nodes: Vec<TrieNode>,
}

impl SubwordTrie {
    fn new() -> Self {
        Self {
            nodes: vec![TrieNode::default()],
        }
    }

    /// Insert a subword (as a sequence of chars) and its vocab id.
    fn insert(&mut self, chars: &[char], id: u32) {
        let mut cur = 0;
        for &c in chars {
            // Find or create the child for this char.
            let next =
                if let Some((_, idx)) = self.nodes[cur].children.iter().find(|(k, _)| *k == c) {
                    *idx
                } else {
                    let idx = self.nodes.len();
                    self.nodes[cur].children.push((c, idx));
                    self.nodes.push(TrieNode::default());
                    idx
                };
            cur = next;
        }
        self.nodes[cur].terminals.push(id);
    }

    /// Walk the trie along the input chars starting at `start`,
    /// collecting (end_position, vocab_id) matches.
    fn collect_matches(&self, chars: &[char], start: usize, out: &mut Vec<(usize, u32)>) {
        let mut cur = 0;
        for (offset, &c) in chars[start..].iter().enumerate() {
            let next = self.nodes[cur]
                .children
                .iter()
                .find(|(k, _)| *k == c)
                .map(|(_, idx)| *idx);
            let Some(next) = next else { break };
            cur = next;
            if !self.nodes[cur].terminals.is_empty() {
                let end = start + offset + 1;
                for &id in &self.nodes[cur].terminals {
                    out.push((end, id));
                }
            }
        }
    }
}

/// Unigram (SentencePiece) model.
#[derive(Debug, Clone)]
pub struct Unigram {
    /// Vocabulary: each entry is a subword string.
    pub vocab: Vocab,
    /// Log-probability of each subword, indexed parallel to `vocab`.
    pub log_probs: Vec<f64>,
    /// Whitespace marker. Pre-tokenized input is expected to have
    /// its leading whitespace converted to this marker before
    /// tokenization.
    pub whitespace_marker: String,
    /// Token to emit for subwords at or below `min_score`. Default
    /// `<unk>`.
    pub unk_token: String,
    /// Floor on the log-probability. Any subword with `log_prob <=
    /// min_score` is treated as unknown.
    pub min_score: f64,
}

impl Unigram {
    /// Construct a Unigram model with the given config.
    pub fn new(
        vocab: Vocab,
        log_probs: Vec<f64>,
        whitespace_marker: impl Into<String>,
        unk_token: impl Into<String>,
        min_score: f64,
    ) -> Self {
        Self {
            vocab,
            log_probs,
            whitespace_marker: whitespace_marker.into(),
            unk_token: unk_token.into(),
            min_score,
        }
    }

    /// Apply Viterbi to a single pre-token string.
    pub fn apply(&self, word: &str) -> Result<Vec<String>> {
        if word.is_empty() {
            return Ok(Vec::new());
        }
        // Ensure the word starts with the whitespace marker.
        let marker = self.whitespace_marker.chars().next().unwrap_or('\u{2581}');
        let normalized: String = if word.starts_with(marker) {
            word.to_string()
        } else {
            let mut s = String::with_capacity(word.len() + 1);
            s.push(marker);
            s.push_str(word);
            s
        };
        let chars: Vec<char> = normalized.chars().collect();
        let n = chars.len();

        // Build the subword trie from the active vocab.
        let mut trie = SubwordTrie::new();
        for (id, token) in self.vocab.iter() {
            if self.log_probs[id as usize] > self.min_score {
                let token_chars: Vec<char> = token.chars().collect();
                if !token_chars.is_empty() {
                    trie.insert(&token_chars, id);
                }
            }
        }

        // For each start position, collect matching (end, id) pairs.
        let mut matches_at: Vec<Vec<(usize, u32)>> = vec![Vec::new(); n];
        for (start, slot) in matches_at.iter_mut().enumerate() {
            trie.collect_matches(&chars, start, slot);
        }

        // Forward pass: best_score[i] = best total log-prob to reach
        // position i. Initialized to -inf except best_score[0] = 0.
        let mut best_score = vec![f64::NEG_INFINITY; n + 1];
        let mut backtrace: Vec<Option<(usize, u32)>> = vec![None; n + 1];
        best_score[0] = 0.0;

        for start in 0..n {
            if best_score[start] == f64::NEG_INFINITY {
                continue;
            }
            for &(end, id) in &matches_at[start] {
                let score = best_score[start] + self.log_probs[id as usize];
                if score > best_score[end] {
                    best_score[end] = score;
                    backtrace[end] = Some((start, id));
                }
            }
        }

        if best_score[n] == f64::NEG_INFINITY {
            return Err(SplinterError::UnknownToken(self.unk_token.clone()));
        }

        // Backtrace. Tokens are returned *as they appear in the vocab*,
        // including any leading whitespace marker. Use
        // [`MetaspaceDecoder`](crate::decoder) to convert back to
        // plain text.
        let mut out: Vec<String> = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let Some((prev, id)) = backtrace[pos] else {
                out.push(self.unk_token.clone());
                break;
            };
            let token = self
                .vocab
                .id_to_token(id)
                .map_err(|_| SplinterError::UnknownToken(self.unk_token.clone()))?
                .to_owned();
            out.push(token);
            pos = prev;
        }
        out.reverse();
        Ok(out)
    }
}

impl Model for Unigram {
    fn tokenize(&self, pre_token: PreToken<'_>) -> Result<Vec<String>> {
        self.apply(pre_token.text.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viterbi_picks_best_segmentation() {
        let tokens: Vec<String> = vec!["<unk>".into(), "▁a".into(), "▁b".into(), "▁ab".into()];
        let vocab = Vocab::from_tokens(tokens).unwrap();
        let log_probs: Vec<f64> = vec![
            f64::NEG_INFINITY, // <unk>
            -1.0,              // ▁a
            -1.0,              // ▁b
            -1.5,              // ▁ab (joint)
        ];
        let m = Unigram::new(vocab, log_probs, "\u{2581}", "<unk>", -10.0);
        let toks = m.apply("ab").unwrap();
        assert_eq!(toks, vec!["▁ab"]);
    }

    #[test]
    fn viterbi_falls_back_to_unigrams() {
        let tokens: Vec<String> = vec!["<unk>".into(), "▁a".into(), "▁b".into(), "▁ab".into()];
        let vocab = Vocab::from_tokens(tokens).unwrap();
        let log_probs: Vec<f64> = vec![f64::NEG_INFINITY, -1.0, -1.0, -1.5];
        let m = Unigram::new(vocab, log_probs, "\u{2581}", "<unk>", -10.0);
        let toks = m.apply("a").unwrap();
        assert_eq!(toks, vec!["▁a"]);
    }

    #[test]
    fn unknown_char_errors() {
        let m = build();
        assert!(m.apply("ax").is_err());
    }

    fn build() -> Unigram {
        let tokens: Vec<String> = vec!["<unk>".into(), "▁a".into(), "▁b".into(), "▁ab".into()];
        let vocab = Vocab::from_tokens(tokens).unwrap();
        let log_probs: Vec<f64> = vec![f64::NEG_INFINITY, -1.0, -1.0, -1.5];
        Unigram::new(vocab, log_probs, "\u{2581}", "<unk>", -10.0)
    }
}
