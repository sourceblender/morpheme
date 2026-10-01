//! [`PreTokenizedString`]: the input split into pieces, each piece a
//! [`NormalizedString`] that may already carry its tokens.

use std::collections::HashMap;

use crate::encoding::Encoding;
use crate::error::{Error, Result};
use crate::normalized_string::{NormalizedString, OffsetRange};
use crate::{Offsets, Token};

/// How offsets in an [`Encoding`] are expressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffsetType {
    /// Byte offsets into the original input (the Rust default).
    Byte,
    /// Char (Unicode scalar) offsets into the original input — what the
    /// Python `tokenizers` API returns.
    Char,
}

/// One piece of a [`PreTokenizedString`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Split {
    /// The piece's text, with alignments back to the original input.
    pub normalized: NormalizedString,
    /// Tokens for this piece, once the model (or an added token match)
    /// has produced them. Pieces that already have tokens are skipped
    /// by later pre-tokenization and tokenization steps.
    pub tokens: Option<Vec<Token>>,
}

impl From<NormalizedString> for Split {
    fn from(normalized: NormalizedString) -> Self {
        Self {
            normalized,
            tokens: None,
        }
    }
}

impl From<(NormalizedString, Option<Vec<Token>>)> for Split {
    fn from((normalized, tokens): (NormalizedString, Option<Vec<Token>>)) -> Self {
        Self { normalized, tokens }
    }
}

/// A string being pre-tokenized: an ordered list of [`Split`]s that,
/// concatenated, cover the original input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreTokenizedString {
    original: String,
    splits: Vec<Split>,
}

impl From<NormalizedString> for PreTokenizedString {
    fn from(normalized: NormalizedString) -> Self {
        Self {
            original: normalized.get_original().to_owned(),
            splits: vec![Split::from(normalized)],
        }
    }
}

impl From<&str> for PreTokenizedString {
    fn from(s: &str) -> Self {
        NormalizedString::from(s).into()
    }
}

impl PreTokenizedString {
    /// Split every untokenized piece with `split_fn`.
    ///
    /// `split_fn(index, piece)` must return pieces that are slices of
    /// `piece` (so that offsets keep mapping to the original input).
    /// Empty pieces are dropped.
    pub fn split<F, U, R>(&mut self, mut split_fn: F) -> Result<()>
    where
        F: FnMut(usize, NormalizedString) -> Result<U>,
        U: IntoIterator<Item = R>,
        R: Into<Split>,
    {
        let mut new_splits = Vec::with_capacity(self.splits.len());
        for (i, split) in self.splits.drain(..).enumerate() {
            if split.tokens.is_some() {
                new_splits.push(split);
                continue;
            }
            for piece in split_fn(i, split.normalized)? {
                let piece: Split = piece.into();
                if !piece.normalized.is_empty() {
                    new_splits.push(piece);
                }
            }
        }
        self.splits = new_splits;
        Ok(())
    }

    /// Apply `normalize` to every untokenized piece.
    pub fn normalize<F>(&mut self, normalize: F) -> Result<()>
    where
        F: Fn(&mut NormalizedString) -> Result<()>,
    {
        for split in self.splits.iter_mut().filter(|s| s.tokens.is_none()) {
            normalize(&mut split.normalized)?;
        }
        Ok(())
    }

    /// Run `tokenize` on every untokenized piece.
    pub fn tokenize<F>(&mut self, tokenize: F) -> Result<()>
    where
        F: Fn(&NormalizedString) -> Result<Vec<Token>>,
    {
        for split in self.splits.iter_mut().filter(|s| s.tokens.is_none()) {
            split.tokens = Some(tokenize(&split.normalized)?);
        }
        Ok(())
    }

    /// The pieces, in order.
    pub fn splits(&self) -> &[Split] {
        &self.splits
    }

    /// The pieces as `(text, original offsets, tokens)`.
    pub fn get_splits(&self, offset_type: OffsetType) -> Vec<(&str, Offsets, &Option<Vec<Token>>)> {
        let converter = match offset_type {
            OffsetType::Char => Some(BytesToCharOffsetConverter::new(&self.original)),
            OffsetType::Byte => None,
        };
        self.splits
            .iter()
            .map(|split| {
                let mut offsets = split.normalized.offsets_original();
                if let Some(c) = &converter {
                    offsets = c.convert(offsets).unwrap_or(offsets);
                }
                (split.normalized.get(), offsets, &split.tokens)
            })
            .collect()
    }

    /// Build an [`Encoding`] from fully tokenized pieces.
    ///
    /// Token offsets (relative to their piece's normalized text) are
    /// mapped back to the original input. Each piece becomes one "word"
    /// unless `word_idx` forces a single word id for everything.
    pub fn into_encoding(
        self,
        word_idx: Option<u32>,
        type_id: u32,
        offset_type: OffsetType,
    ) -> Result<Encoding> {
        if self.splits.is_empty() {
            return Ok(Encoding::default());
        }
        if self.splits.iter().any(|s| s.tokens.is_none()) {
            return Err(Error::Model(
                "split has not been tokenized; call `tokenize` first".into(),
            ));
        }
        let converter = match offset_type {
            OffsetType::Char => Some(BytesToCharOffsetConverter::new(&self.original)),
            OffsetType::Byte => None,
        };

        let mut encoding = Encoding::default();
        for (idx, split) in self.splits.into_iter().enumerate() {
            let normalized = split.normalized;
            let (base, _) = normalized.offsets_original();
            for token in split.tokens.expect("checked above") {
                let mut offsets = normalized
                    .convert_offsets(OffsetRange::Normalized(token.offsets.0..token.offsets.1))
                    .map_or(token.offsets, |r| (base + r.start, base + r.end));
                if let Some(c) = &converter {
                    offsets = c.convert(offsets).unwrap_or(offsets);
                }
                encoding.push_token(
                    token.id,
                    token.value,
                    offsets,
                    word_idx.or(Some(idx as u32)),
                    type_id,
                );
            }
        }
        Ok(encoding)
    }
}

/// Converts byte offsets of a string to char offsets.
struct BytesToCharOffsetConverter {
    map: HashMap<usize, usize>,
}

impl BytesToCharOffsetConverter {
    fn new(s: &str) -> Self {
        let mut map = HashMap::with_capacity(s.len());
        for (i, (b, c)) in s.char_indices().enumerate() {
            for n in 0..c.len_utf8() {
                map.insert(b + n, i);
            }
        }
        Self { map }
    }

    fn convert(&self, offsets: Offsets) -> Option<Offsets> {
        match (self.map.get(&offsets.0), self.map.get(&offsets.1)) {
            (Some(&start), Some(&end)) => Some((start, end)),
            (Some(&start), None) => {
                let last = offsets
                    .1
                    .checked_sub(1)
                    .and_then(|b| self.map.get(&b).copied())
                    .unwrap_or(start + 1);
                Some((start, last + 1))
            }
            _ => None,
        }
    }
}
