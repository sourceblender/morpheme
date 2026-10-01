//! The [`Encoding`] produced by [`crate::Tokenizer::encode`].

use std::collections::HashMap;
use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::{Offsets, Token};

/// Which side truncation removes tokens from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TruncationDirection {
    /// Remove tokens from the start.
    Left,
    /// Remove tokens from the end.
    #[default]
    Right,
}

/// Which side padding adds tokens to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PaddingDirection {
    /// Pad at the start.
    Left,
    /// Pad at the end.
    #[default]
    Right,
}

/// The output of encoding one input (a sentence or a sentence pair).
///
/// All the per-token vectors are parallel: index `i` describes token `i`.
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
/// use morpheme::models::WordLevel;
/// use morpheme::pre_tokenizers::Whitespace;
/// use morpheme::Tokenizer;
///
/// let vocab: HashMap<String, u32> = [("[UNK]", 0), ("héllo", 1), ("!", 2)].map(|(t, i)| (t.to_string(), i)).into();
/// let tokenizer = Tokenizer::new(WordLevel::builder().vocab(vocab).unk_token("[UNK]").build()?)
///     .with_pre_tokenizer(Whitespace);
///
/// let encoding = tokenizer.encode("héllo!", true)?;
/// assert_eq!(encoding.ids(), [1, 2]);
/// assert_eq!(encoding.tokens(), ["héllo", "!"]);
/// assert_eq!(encoding.offsets(), [(0, 6), (6, 7)]); // bytes ("é" is 2 bytes)
/// assert_eq!(encoding.word_ids(), [Some(0), Some(1)]);
/// assert_eq!(tokenizer.encode_char_offsets("héllo!", true)?.offsets(), [(0, 5), (5, 6)]);
/// # Ok::<(), morpheme::Error>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Encoding {
    ids: Vec<u32>,
    type_ids: Vec<u32>,
    tokens: Vec<String>,
    words: Vec<Option<u32>>,
    offsets: Vec<Offsets>,
    special_tokens_mask: Vec<u32>,
    attention_mask: Vec<u32>,
    overflowing: Vec<Encoding>,
    /// For multi-sequence encodings, the token range of each sequence.
    sequence_ranges: HashMap<usize, Range<usize>>,
}

impl Encoding {
    /// Build an encoding from its parts. All vectors must have the same
    /// length.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ids: Vec<u32>,
        type_ids: Vec<u32>,
        tokens: Vec<String>,
        words: Vec<Option<u32>>,
        offsets: Vec<Offsets>,
        special_tokens_mask: Vec<u32>,
        attention_mask: Vec<u32>,
        overflowing: Vec<Encoding>,
    ) -> Self {
        Self {
            ids,
            type_ids,
            tokens,
            words,
            offsets,
            special_tokens_mask,
            attention_mask,
            overflowing,
            sequence_ranges: HashMap::new(),
        }
    }

    /// An empty encoding with room for `len` tokens.
    pub fn with_capacity(len: usize) -> Self {
        Self {
            ids: Vec::with_capacity(len),
            type_ids: Vec::with_capacity(len),
            tokens: Vec::with_capacity(len),
            words: Vec::with_capacity(len),
            offsets: Vec::with_capacity(len),
            special_tokens_mask: Vec::with_capacity(len),
            attention_mask: Vec::with_capacity(len),
            overflowing: Vec::new(),
            sequence_ranges: HashMap::new(),
        }
    }

    /// Build an encoding from model tokens (no word ids, no specials).
    pub fn from_tokens(tokens: Vec<Token>, type_id: u32) -> Self {
        let mut e = Self::with_capacity(tokens.len());
        for t in tokens {
            e.push_token(t.id, t.value, t.offsets, None, type_id);
        }
        e
    }

    pub(crate) fn push_token(
        &mut self,
        id: u32,
        token: String,
        offsets: Offsets,
        word: Option<u32>,
        type_id: u32,
    ) {
        self.ids.push(id);
        self.tokens.push(token);
        self.offsets.push(offsets);
        self.words.push(word);
        self.type_ids.push(type_id);
        self.special_tokens_mask.push(0);
        self.attention_mask.push(1);
    }

    /// Number of tokens.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// True if there are no tokens.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Token ids.
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// Segment (type) ids.
    pub fn type_ids(&self) -> &[u32] {
        &self.type_ids
    }

    /// Replace the type ids.
    pub fn set_type_ids(&mut self, type_ids: Vec<u32>) {
        self.type_ids = type_ids;
    }

    /// Give this encoding and every overflowing part (recursively) the
    /// same type id. Post-processors use this so that overflow produced
    /// by truncation agrees with the main encoding.
    pub(crate) fn set_uniform_type_id(&mut self, type_id: u32) {
        self.type_ids = vec![type_id; self.len()];
        for o in &mut self.overflowing {
            o.set_uniform_type_id(type_id);
        }
    }

    /// Token strings.
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }

    /// For each token, the index of the word (pre-token) it came from.
    /// `None` for special tokens and padding.
    pub fn word_ids(&self) -> &[Option<u32>] {
        &self.words
    }

    /// Mutable access to the word ids.
    pub fn word_ids_mut(&mut self) -> &mut [Option<u32>] {
        &mut self.words
    }

    /// Offsets of each token in the original input (bytes or chars,
    /// depending on how the encoding was produced).
    pub fn offsets(&self) -> &[Offsets] {
        &self.offsets
    }

    /// Mutable access to the offsets.
    pub fn offsets_mut(&mut self) -> &mut [Offsets] {
        &mut self.offsets
    }

    /// `1` for special tokens (added by the post-processor or padding),
    /// `0` otherwise.
    pub fn special_tokens_mask(&self) -> &[u32] {
        &self.special_tokens_mask
    }

    /// `1` for real tokens, `0` for padding.
    pub fn attention_mask(&self) -> &[u32] {
        &self.attention_mask
    }

    /// Parts that did not fit when truncating with overflow.
    pub fn overflowing(&self) -> &[Encoding] {
        &self.overflowing
    }

    /// Mutable access to the overflowing parts.
    pub fn overflowing_mut(&mut self) -> &mut Vec<Encoding> {
        &mut self.overflowing
    }

    /// Take the overflowing parts out of this encoding.
    pub fn take_overflowing(&mut self) -> Vec<Encoding> {
        std::mem::take(&mut self.overflowing)
    }

    /// Number of input sequences merged into this encoding.
    pub fn n_sequences(&self) -> usize {
        if self.sequence_ranges.is_empty() {
            1
        } else {
            self.sequence_ranges.len()
        }
    }

    /// Mark the whole encoding and its overflowing parts as belonging
    /// to sequence `sequence_id`.
    pub fn set_sequence_id(&mut self, sequence_id: usize) {
        self.sequence_ranges.insert(sequence_id, 0..self.len());
        for overflow in &mut self.overflowing {
            overflow.set_sequence_id(sequence_id);
        }
    }

    /// For each token, the index of the sequence it belongs to.
    pub fn sequence_ids(&self) -> Vec<Option<usize>> {
        if self.sequence_ranges.is_empty() {
            return vec![Some(0); self.len()];
        }
        let mut out = vec![None; self.len()];
        for (&seq, range) in &self.sequence_ranges {
            for slot in &mut out[range.clone()] {
                *slot = Some(seq);
            }
        }
        out
    }

    fn sequence_range(&self, sequence_id: usize) -> Range<usize> {
        self.sequence_ranges
            .get(&sequence_id)
            .cloned()
            .unwrap_or_else(|| {
                if self.sequence_ranges.is_empty() && sequence_id == 0 {
                    0..self.len()
                } else {
                    0..0
                }
            })
    }

    /// The sequence that token `token` belongs to.
    pub fn token_to_sequence(&self, token: usize) -> Option<usize> {
        if token >= self.len() {
            None
        } else if self.sequence_ranges.is_empty() {
            Some(0)
        } else {
            self.sequence_ranges
                .iter()
                .find(|(_, r)| r.contains(&token))
                .map(|(seq, _)| *seq)
        }
    }

    /// Token range `[start, end)` covering `word` in `sequence_id`.
    pub fn word_to_tokens(&self, word: u32, sequence_id: usize) -> Option<(usize, usize)> {
        let range = self.sequence_range(sequence_id);
        let mut start = None;
        let mut end = None;
        for (i, w) in self.words[range.clone()].iter().enumerate() {
            match w {
                Some(w) if *w == word => {
                    if start.is_none() {
                        start = Some(range.start + i);
                    }
                    end = Some(range.start + i + 1);
                }
                Some(w) if *w > word => break,
                _ => {}
            }
        }
        Some((start?, end?))
    }

    /// `(sequence, offsets)` of token `token`.
    pub fn token_to_chars(&self, token: usize) -> Option<(usize, Offsets)> {
        Some((self.token_to_sequence(token)?, *self.offsets.get(token)?))
    }

    /// `(sequence, word)` of token `token`.
    pub fn token_to_word(&self, token: usize) -> Option<(usize, u32)> {
        Some((self.token_to_sequence(token)?, (*self.words.get(token)?)?))
    }

    /// Index of the token containing position `pos` of `sequence_id`.
    pub fn char_to_token(&self, pos: usize, sequence_id: usize) -> Option<usize> {
        let range = self.sequence_range(sequence_id);
        self.offsets[range.clone()]
            .iter()
            .position(|(s, e)| pos >= *s && pos < *e)
            .map(|i| range.start + i)
    }

    /// Truncate to `max_len` tokens; removed tokens become overflowing
    /// encodings of up to `max_len` tokens, each overlapping the
    /// previous one by `stride` tokens.
    ///
    /// # Panics
    /// Panics if `stride >= max_len` (when `max_len > 0`).
    pub fn truncate(&mut self, max_len: usize, stride: usize, direction: TruncationDirection) {
        let len = self.len();
        if max_len >= len {
            return;
        }
        if max_len == 0 {
            let whole = std::mem::replace(self, Encoding::with_capacity(0));
            self.overflowing.push(whole);
            return;
        }
        assert!(
            stride < max_len,
            "`stride` ({stride}) must be strictly less than `max_len` ({max_len})"
        );
        self.sequence_ranges.clear();

        let step = max_len - stride;
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        match direction {
            TruncationDirection::Right => {
                let mut start = 0;
                loop {
                    let stop = (start + max_len).min(len);
                    ranges.push((start, stop));
                    if stop == len {
                        break;
                    }
                    start += step;
                }
            }
            TruncationDirection::Left => {
                let mut stop = len;
                loop {
                    let start = stop.saturating_sub(max_len);
                    ranges.push((start, stop));
                    if start == 0 {
                        break;
                    }
                    stop -= step;
                }
            }
        }

        let part = |(s, e): (usize, usize)| Encoding {
            ids: self.ids[s..e].to_vec(),
            type_ids: self.type_ids[s..e].to_vec(),
            tokens: self.tokens[s..e].to_vec(),
            words: self.words[s..e].to_vec(),
            offsets: self.offsets[s..e].to_vec(),
            special_tokens_mask: self.special_tokens_mask[s..e].to_vec(),
            attention_mask: self.attention_mask[s..e].to_vec(),
            overflowing: Vec::new(),
            sequence_ranges: HashMap::new(),
        };
        let mut head = part(ranges[0]);
        head.overflowing = ranges[1..].iter().map(|r| part(*r)).collect();
        *self = head;
    }

    /// Concatenate several encodings into one.
    pub fn merge<I: IntoIterator<Item = Encoding>>(encodings: I, growing_offsets: bool) -> Self {
        let mut out = Encoding::default();
        for e in encodings {
            out.merge_with(e, growing_offsets);
        }
        out
    }

    /// Append `pair` to this encoding. Overflowing parts are combined
    /// pairwise. With `growing_offsets`, `pair`'s offsets are shifted to
    /// start after this encoding's last offset.
    pub fn merge_with(&mut self, pair: Encoding, growing_offsets: bool) {
        let mut overflowings = Vec::new();
        for self_o in &self.overflowing {
            let mut n = self_o.clone();
            n.merge_with(pair.clone(), growing_offsets);
            overflowings.push(n);
            for other_o in &pair.overflowing {
                let mut n = self_o.clone();
                n.merge_with(other_o.clone(), growing_offsets);
                overflowings.push(n);
            }
        }
        for other_o in &pair.overflowing {
            let mut n = self.clone();
            n.merge_with(other_o.clone(), growing_offsets);
            overflowings.push(n);
        }

        let base = self.len();
        self.sequence_ranges.extend(
            pair.sequence_ranges
                .into_iter()
                .map(|(seq, r)| (seq, base + r.start..base + r.end)),
        );
        let shift = if growing_offsets {
            self.offsets.last().map_or(0, |o| o.1)
        } else {
            0
        };
        self.ids.extend(pair.ids);
        self.type_ids.extend(pair.type_ids);
        self.tokens.extend(pair.tokens);
        self.words.extend(pair.words);
        self.offsets.extend(
            pair.offsets
                .into_iter()
                .map(|(s, e)| (s + shift, e + shift)),
        );
        self.special_tokens_mask.extend(pair.special_tokens_mask);
        self.attention_mask.extend(pair.attention_mask);
        self.overflowing = overflowings;
    }

    /// Pad to `target_length` tokens (no-op if already that long).
    pub fn pad(
        &mut self,
        target_length: usize,
        pad_id: u32,
        pad_type_id: u32,
        pad_token: &str,
        direction: PaddingDirection,
    ) {
        for o in &mut self.overflowing {
            o.pad(target_length, pad_id, pad_type_id, pad_token, direction);
        }
        if self.len() >= target_length {
            return;
        }
        let n = target_length - self.len();
        fn pad_vec<T: Clone>(v: &mut Vec<T>, n: usize, value: T, left: bool) {
            if left {
                v.splice(0..0, std::iter::repeat_n(value, n));
            } else {
                v.extend(std::iter::repeat_n(value, n));
            }
        }
        let left = direction == PaddingDirection::Left;
        pad_vec(&mut self.ids, n, pad_id, left);
        pad_vec(&mut self.type_ids, n, pad_type_id, left);
        pad_vec(&mut self.tokens, n, pad_token.to_owned(), left);
        pad_vec(&mut self.words, n, None, left);
        pad_vec(&mut self.attention_mask, n, 0, left);
        pad_vec(&mut self.special_tokens_mask, n, 1, left);
        pad_vec(&mut self.offsets, n, (0, 0), left);
        if left {
            for r in self.sequence_ranges.values_mut() {
                *r = r.start + n..r.end + n;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc(n: u32) -> Encoding {
        let tokens = (0..n)
            .map(|i| Token::new(i, format!("t{i}"), (i as usize, i as usize + 1)))
            .collect();
        Encoding::from_tokens(tokens, 0)
    }

    #[test]
    fn truncate_right_with_stride_produces_overflow() {
        let mut e = enc(5);
        e.truncate(3, 1, TruncationDirection::Right);
        assert_eq!(e.ids(), &[0, 1, 2]);
        let o: Vec<_> = e.overflowing().iter().map(|o| o.ids().to_vec()).collect();
        assert_eq!(o, vec![vec![2, 3, 4]]);
    }

    #[test]
    fn truncate_left() {
        let mut e = enc(5);
        e.truncate(2, 0, TruncationDirection::Left);
        assert_eq!(e.ids(), &[3, 4]);
        let o: Vec<_> = e.overflowing().iter().map(|o| o.ids().to_vec()).collect();
        assert_eq!(o, vec![vec![1, 2], vec![0]]);
    }

    #[test]
    fn pad_left_and_right() {
        let mut e = enc(2);
        e.pad(4, 9, 0, "[PAD]", PaddingDirection::Left);
        assert_eq!(e.ids(), &[9, 9, 0, 1]);
        assert_eq!(e.attention_mask(), &[0, 0, 1, 1]);
        assert_eq!(e.special_tokens_mask(), &[1, 1, 0, 0]);
    }

    #[test]
    fn merge_tracks_sequences() {
        let mut a = enc(2);
        a.set_sequence_id(0);
        let mut b = enc(3);
        b.set_sequence_id(1);
        let m = Encoding::merge([a, b], false);
        assert_eq!(m.len(), 5);
        assert_eq!(
            m.sequence_ids(),
            vec![Some(0), Some(0), Some(1), Some(1), Some(1)]
        );
    }
}
