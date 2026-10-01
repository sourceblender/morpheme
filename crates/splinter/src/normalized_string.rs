//! [`NormalizedString`]: a string that remembers where every byte came
//! from.
//!
//! Normalizers and pre-tokenizers rewrite text (lowercasing, Unicode
//! normalization, replacing spaces with `▁`, …). To report offsets into
//! the *original* input, every byte of the normalized string carries the
//! `(start, end)` byte range of the original text it was produced from.
//! The alignment model is the one used by Hugging Face `tokenizers`, so
//! offsets computed here match theirs exactly.

use std::ops::{Bound, RangeBounds};

use serde::{Deserialize, Serialize};
use unicode_normalization_alignments::UnicodeNormalization;

use crate::error::Result;
use crate::pattern::Pattern;
use crate::Offsets;

/// A byte range expressed in one of the two coordinate systems of a
/// [`NormalizedString`].
#[derive(Debug, Clone)]
pub enum OffsetRange<R: RangeBounds<usize> + Clone> {
    /// Byte range in the original string.
    Original(R),
    /// Byte range in the normalized string.
    Normalized(R),
}

impl<R: RangeBounds<usize> + Clone> OffsetRange<R> {
    fn bounds(&self) -> &R {
        match self {
            OffsetRange::Original(r) | OffsetRange::Normalized(r) => r,
        }
    }

    fn to_range(&self, max_len: usize) -> std::ops::Range<usize> {
        let r = self.bounds();
        let start = match r.start_bound() {
            Bound::Unbounded => 0,
            Bound::Included(i) => *i,
            Bound::Excluded(i) => *i + 1,
        };
        let end = match r.end_bound() {
            Bound::Unbounded => max_len,
            Bound::Included(i) => *i + 1,
            Bound::Excluded(i) => *i,
        };
        start..end
    }

    fn is_original(&self) -> bool {
        matches!(self, OffsetRange::Original(_))
    }
}

/// What to do with the delimiter when splitting a string.
///
/// Example splitting `"the-final--countdown"` on `-`:
///
/// - `Removed`            → `["the", "final", "countdown"]`
/// - `Isolated`           → `["the", "-", "final", "-", "-", "countdown"]`
/// - `MergedWithPrevious` → `["the-", "final-", "-", "countdown"]`
/// - `MergedWithNext`     → `["the", "-final", "-", "-countdown"]`
/// - `Contiguous`         → `["the", "-", "final", "--", "countdown"]`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SplitDelimiterBehavior {
    /// Drop the delimiter.
    Removed,
    /// Keep the delimiter as its own split.
    Isolated,
    /// Attach the delimiter to the preceding split.
    MergedWithPrevious,
    /// Attach the delimiter to the following split.
    MergedWithNext,
    /// Keep runs of consecutive delimiters together as one split.
    Contiguous,
}

/// A normalized string with byte-level alignments back to the original.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NormalizedString {
    original: String,
    normalized: String,
    /// For each byte of `normalized`, the original byte range it maps to.
    alignments: Vec<Offsets>,
    /// When this string is a slice of a larger one, the byte offset of
    /// `original` inside the larger original string.
    original_shift: usize,
}

impl From<&str> for NormalizedString {
    fn from(s: &str) -> Self {
        let mut alignments = Vec::with_capacity(s.len());
        for (b, c) in s.char_indices() {
            let len = c.len_utf8();
            alignments.extend(std::iter::repeat_n((b, b + len), len));
        }
        Self {
            original: s.to_owned(),
            normalized: s.to_owned(),
            alignments,
            original_shift: 0,
        }
    }
}

impl From<String> for NormalizedString {
    fn from(s: String) -> Self {
        Self::from(s.as_str())
    }
}

impl NormalizedString {
    /// The normalized text.
    pub fn get(&self) -> &str {
        &self.normalized
    }

    /// The original text.
    pub fn get_original(&self) -> &str {
        &self.original
    }

    /// Byte length of the normalized text.
    pub fn len(&self) -> usize {
        self.normalized.len()
    }

    /// Byte length of the original text.
    pub fn len_original(&self) -> usize {
        self.original.len()
    }

    /// True if the normalized text is empty.
    pub fn is_empty(&self) -> bool {
        self.normalized.is_empty()
    }

    /// The span this string covers in the top-level original input.
    pub fn offsets_original(&self) -> Offsets {
        (
            self.original_shift,
            self.original_shift + self.len_original(),
        )
    }

    /// Convert a range from one coordinate system to the other.
    ///
    /// Returns `None` if the range is reversed or out of bounds.
    pub fn convert_offsets<R>(&self, range: OffsetRange<R>) -> Option<std::ops::Range<usize>>
    where
        R: RangeBounds<usize> + Clone,
    {
        let original = range.is_original();
        let target = if original {
            range.to_range(self.len_original())
        } else {
            range.to_range(self.len())
        };

        if target.start == target.end {
            return Some(target);
        }
        if target.start > target.end {
            return None;
        }
        if original && self.original.is_empty() && target == (0..0) {
            return Some(0..self.len());
        }
        if !original && self.normalized.is_empty() && target == (0..0) {
            return Some(0..self.len_original());
        }

        if original {
            let mut start = None;
            let mut end = None;
            for (i, &(a_start, a_end)) in self.alignments.iter().enumerate() {
                if target.end < a_end {
                    break;
                }
                if start.is_none() && target.start <= a_start && a_start != a_end {
                    start = Some(i);
                }
                end = Some(i + 1);
            }
            match (start, end) {
                (Some(s), None) => Some(s..s),
                (None, Some(e)) => Some(e..e),
                (Some(s), Some(e)) => Some(s..e),
                (None, None) => None,
            }
        } else {
            let aligned = self.alignments.get(target)?;
            let first = aligned.first()?;
            let last = aligned.last()?;
            Some(first.0..last.1)
        }
    }

    /// Get a slice of the normalized text, addressed in either
    /// coordinate system.
    pub fn get_range<R>(&self, range: OffsetRange<R>) -> Option<&str>
    where
        R: RangeBounds<usize> + Clone,
    {
        match range {
            OffsetRange::Original(_) => self.normalized.get(self.convert_offsets(range)?),
            OffsetRange::Normalized(_) => self.normalized.get(range.to_range(self.len())),
        }
    }

    /// Get a slice of the original text, addressed in either coordinate
    /// system.
    pub fn get_range_original<R>(&self, range: OffsetRange<R>) -> Option<&str>
    where
        R: RangeBounds<usize> + Clone,
    {
        match range {
            OffsetRange::Original(_) => self.original.get(range.to_range(self.len_original())),
            OffsetRange::Normalized(_) => self.original.get(self.convert_offsets(range)?),
        }
    }

    /// Extract a sub-`NormalizedString`. Returns `None` if the range is
    /// not on char boundaries.
    pub fn slice<R>(&self, range: OffsetRange<R>) -> Option<NormalizedString>
    where
        R: RangeBounds<usize> + Clone,
    {
        let (normalized_range, original_range) = if range.is_original() {
            let r = range.to_range(self.len_original());
            if !(self.original.is_char_boundary(r.start) && self.original.is_char_boundary(r.end)) {
                return None;
            }
            (self.convert_offsets(OffsetRange::Original(r.clone()))?, r)
        } else {
            let r = range.to_range(self.len());
            if !(self.normalized.is_char_boundary(r.start)
                && self.normalized.is_char_boundary(r.end))
            {
                return None;
            }
            let o = self.convert_offsets(OffsetRange::Normalized(r.clone()))?;
            (r, o)
        };

        let shift = original_range.start;
        Some(NormalizedString {
            original: self.original.get(original_range.clone())?.to_owned(),
            normalized: self.normalized.get(normalized_range.clone())?.to_owned(),
            alignments: self
                .alignments
                .get(normalized_range)?
                .iter()
                .map(|&(s, e)| (s - shift, e - shift))
                .collect(),
            original_shift: self.original_shift + original_range.start,
        })
    }

    /// Rewrite part of the normalized string while keeping alignments.
    ///
    /// `dest` yields every char of the replacement text with a change
    /// marker:
    ///
    /// - `0`  — the char replaces the next existing char;
    /// - `1`  — the char is newly inserted (it inherits the alignment of
    ///   the char before it);
    /// - `-N` — the char replaces the next existing char, and the `N`
    ///   chars after that are removed.
    ///
    /// `initial_offset` is the number of chars removed at the very start
    /// of the range, before the first yielded char.
    pub fn transform_range<R, I>(&mut self, range: OffsetRange<R>, dest: I, initial_offset: usize)
    where
        R: RangeBounds<usize> + Clone,
        I: IntoIterator<Item = (char, isize)>,
    {
        let n_range = if range.is_original() {
            match self.convert_offsets(range) {
                Some(r) => r,
                None => return,
            }
        } else {
            range.to_range(self.len())
        };

        let mut replaced = self.normalized[n_range.clone()].chars();
        let initial_removed: usize = (&mut replaced)
            .take(initial_offset)
            .map(char::len_utf8)
            .sum();

        let mut offset = n_range.start + initial_removed;
        let mut new_alignments = Vec::with_capacity(n_range.len());
        let mut new_text = String::with_capacity(n_range.len());
        for (c, change) in dest {
            let align = if change > 0 {
                if offset == 0 {
                    (0, 0)
                } else {
                    self.alignments[offset - 1]
                }
            } else {
                self.alignments[offset]
            };
            if change <= 0 {
                if let Some(r) = replaced.next() {
                    offset += r.len_utf8();
                }
            }
            if change < 0 {
                offset += (&mut replaced)
                    .take(change.unsigned_abs())
                    .map(char::len_utf8)
                    .sum::<usize>();
            }
            new_alignments.extend(std::iter::repeat_n(align, c.len_utf8()));
            new_text.push(c);
        }

        self.alignments.splice(n_range.clone(), new_alignments);
        self.normalized.replace_range(n_range, &new_text);
    }

    /// [`transform_range`](Self::transform_range) over the whole string.
    pub fn transform<I>(&mut self, dest: I, initial_offset: usize)
    where
        I: IntoIterator<Item = (char, isize)>,
    {
        self.transform_range(OffsetRange::Original(..), dest, initial_offset);
    }

    /// Unicode NFD.
    pub fn nfd(&mut self) -> &mut Self {
        let dest: Vec<(char, isize)> = self.normalized.nfd().collect();
        self.transform(dest, 0);
        self
    }

    /// Unicode NFKD.
    pub fn nfkd(&mut self) -> &mut Self {
        let dest: Vec<(char, isize)> = self.normalized.nfkd().collect();
        self.transform(dest, 0);
        self
    }

    /// Unicode NFC.
    pub fn nfc(&mut self) -> &mut Self {
        let dest: Vec<(char, isize)> = self.normalized.nfc().collect();
        self.transform(dest, 0);
        self
    }

    /// Unicode NFKC.
    pub fn nfkc(&mut self) -> &mut Self {
        let dest: Vec<(char, isize)> = self.normalized.nfkc().collect();
        self.transform(dest, 0);
        self
    }

    /// Keep only the chars for which `keep` returns true.
    pub fn filter<F: Fn(char) -> bool>(&mut self, keep: F) -> &mut Self {
        let mut removed: isize = 0;
        let mut removed_start: usize = 0;
        let mut dest = Vec::with_capacity(self.normalized.len());
        let mut last_kept: Option<char> = None;
        for c in self.normalized.chars() {
            if keep(c) {
                match last_kept {
                    Some(lc) => dest.push((lc, -removed)),
                    None => removed_start = removed as usize,
                }
                last_kept = Some(c);
                removed = 0;
            } else {
                removed += 1;
            }
        }
        match last_kept {
            Some(lc) => dest.push((lc, -removed)),
            None => removed_start = removed as usize,
        }
        self.transform(dest, removed_start);
        self
    }

    /// Prepend `s` to the normalized text. No-op on an empty string.
    pub fn prepend(&mut self, s: &str) -> &mut Self {
        if let Some(first) = self.normalized.chars().next() {
            let dest = s
                .chars()
                .enumerate()
                .map(|(i, c)| (c, isize::from(i != 0)))
                .chain(std::iter::once((first, 1)));
            self.transform_range(OffsetRange::Normalized(0..first.len_utf8()), dest, 0);
        }
        self
    }

    /// Append `s` to the normalized text.
    pub fn append(&mut self, s: &str) -> &mut Self {
        if let Some((b, last)) = self.normalized.char_indices().last() {
            let dest = std::iter::once((last, 0)).chain(s.chars().map(|c| (c, 1)));
            self.transform_range(OffsetRange::Normalized(b..), dest, 0);
        } else {
            let dest = s.chars().map(|c| (c, 1));
            self.transform_range(OffsetRange::Normalized(..), dest, 0);
        }
        self
    }

    /// Map every char through `f` (one char in, one char out).
    pub fn map<F: Fn(char) -> char>(&mut self, f: F) -> &mut Self {
        let dest: Vec<(char, isize)> = self.normalized.chars().map(|c| (f(c), 0)).collect();
        self.transform(dest, 0);
        self
    }

    /// Unicode-aware lowercase (a char may expand to several).
    pub fn lowercase(&mut self) -> &mut Self {
        let mut dest = Vec::with_capacity(self.normalized.len());
        for c in self.normalized.chars() {
            for (i, lc) in c.to_lowercase().enumerate() {
                dest.push((lc, isize::from(i > 0)));
            }
        }
        self.transform(dest, 0);
        self
    }

    /// Unicode-aware uppercase (a char may expand to several).
    pub fn uppercase(&mut self) -> &mut Self {
        let mut dest = Vec::with_capacity(self.normalized.len());
        for c in self.normalized.chars() {
            for (i, uc) in c.to_uppercase().enumerate() {
                dest.push((uc, isize::from(i > 0)));
            }
        }
        self.transform(dest, 0);
        self
    }

    /// Replace every match of `pattern` with `content`. Inserted chars
    /// align to the last byte of the match they replace.
    pub fn replace<P: Pattern>(&mut self, pattern: P, content: &str) -> Result<()> {
        let mut new_text = String::with_capacity(self.normalized.len());
        let mut new_alignments = Vec::with_capacity(self.alignments.len());
        let mut last_end = 0;
        for ((start, end), is_match) in pattern.find_matches(&self.normalized)? {
            if !is_match {
                continue;
            }
            new_text.push_str(&self.normalized[last_end..start]);
            new_alignments.extend_from_slice(&self.alignments[last_end..start]);
            let align = if end == 0 {
                (0, 0)
            } else {
                self.alignments[end - 1]
            };
            for c in content.chars() {
                new_alignments.extend(std::iter::repeat_n(align, c.len_utf8()));
                new_text.push(c);
            }
            last_end = end;
        }
        new_text.push_str(&self.normalized[last_end..]);
        new_alignments.extend_from_slice(&self.alignments[last_end..]);
        self.normalized = new_text;
        self.alignments = new_alignments;
        Ok(())
    }

    /// Remove all normalized text. Returns the number of bytes removed.
    pub fn clear(&mut self) -> usize {
        let len = self.len();
        let chars = self.normalized.chars().count();
        self.transform(std::iter::empty(), chars);
        len
    }

    /// Split on `pattern` according to `behavior`. Every produced piece
    /// is a slice of `self`, so offsets keep working.
    pub fn split<P: Pattern>(
        &self,
        pattern: P,
        behavior: SplitDelimiterBehavior,
    ) -> Result<Vec<NormalizedString>> {
        let matches = pattern.find_matches(&self.normalized)?;
        use SplitDelimiterBehavior::*;
        // (offsets, should_remove)
        let spans: Vec<(Offsets, bool)> = match behavior {
            Isolated => matches.into_iter().map(|(o, _)| (o, false)).collect(),
            Removed => matches,
            Contiguous => {
                let mut prev_match = false;
                let mut acc: Vec<(Offsets, bool)> = Vec::new();
                for (o, is_match) in matches {
                    match acc.last_mut() {
                        Some(((_, end), _)) if is_match == prev_match => *end = o.1,
                        _ => acc.push((o, false)),
                    }
                    prev_match = is_match;
                }
                acc
            }
            MergedWithPrevious => {
                let mut prev_match = false;
                let mut acc: Vec<(Offsets, bool)> = Vec::new();
                for (o, is_match) in matches {
                    match acc.last_mut() {
                        Some(((_, end), _)) if is_match && !prev_match => *end = o.1,
                        _ => acc.push((o, false)),
                    }
                    prev_match = is_match;
                }
                acc
            }
            MergedWithNext => {
                let mut prev_match = false;
                let mut acc: Vec<(Offsets, bool)> = Vec::new();
                for (o, is_match) in matches.into_iter().rev() {
                    match acc.last_mut() {
                        Some(((start, _), _)) if is_match && !prev_match => *start = o.0,
                        _ => acc.push((o, false)),
                    }
                    prev_match = is_match;
                }
                acc.reverse();
                acc
            }
        };

        Ok(spans
            .into_iter()
            .filter(|(_, remove)| !remove)
            .map(|((s, e), _)| {
                self.slice(OffsetRange::Normalized(s..e))
                    .expect("pattern matches are on char boundaries")
            })
            .collect())
    }

    /// Remove leading whitespace.
    pub fn lstrip(&mut self) -> &mut Self {
        self.lrstrip(true, false)
    }

    /// Remove trailing whitespace.
    pub fn rstrip(&mut self) -> &mut Self {
        self.lrstrip(false, true)
    }

    /// Remove leading and trailing whitespace.
    pub fn strip(&mut self) -> &mut Self {
        self.lrstrip(true, true)
    }

    fn lrstrip(&mut self, left: bool, right: bool) -> &mut Self {
        let count = self.normalized.chars().count();
        let leading = if left {
            self.normalized
                .chars()
                .take_while(|c| c.is_whitespace())
                .count()
        } else {
            0
        };
        let trailing = if right {
            self.normalized
                .chars()
                .rev()
                .take_while(|c| c.is_whitespace())
                .count()
                .min(count - leading)
        } else {
            0
        };
        if leading == 0 && trailing == 0 {
            return self;
        }
        let keep_end = count - trailing;
        let dest: Vec<(char, isize)> = self
            .normalized
            .chars()
            .enumerate()
            .filter(|(i, _)| *i >= leading && *i < keep_end)
            .map(|(i, c)| {
                if i + 1 == keep_end {
                    (c, -(trailing as isize))
                } else {
                    (c, 0)
                }
            })
            .collect();
        let initial = if dest.is_empty() { count } else { leading };
        self.transform(dest, initial);
        self
    }
}

/// Convert a byte range of `s` to a char range. `None` if the range does
/// not fall on char boundaries.
pub fn bytes_to_char(s: &str, range: std::ops::Range<usize>) -> Option<std::ops::Range<usize>> {
    if range == (0..0) {
        return Some(0..0);
    }
    let mut start = None;
    let mut end = None;
    for (i, (b, c)) in s.char_indices().enumerate() {
        if b > range.end {
            break;
        }
        if b == range.start {
            start = Some(i);
        }
        if b == range.end {
            end = Some(i);
        }
        if b + c.len_utf8() == range.end {
            end = Some(i + 1);
        }
    }
    Some(start?..end?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_alignments() {
        let n = NormalizedString::from("héllo");
        assert_eq!(n.len(), 6);
        assert_eq!(n.convert_offsets(OffsetRange::Normalized(1..3)), Some(1..3));
    }

    #[test]
    fn nfd_then_filter_maps_back_to_original() {
        let mut n = NormalizedString::from("élégant");
        n.nfd().filter(|c| !('\u{300}'..='\u{36f}').contains(&c));
        assert_eq!(n.get(), "elegant");
        // "e" (normalized 0..1) came from "é" (original 0..2).
        assert_eq!(n.convert_offsets(OffsetRange::Normalized(0..1)), Some(0..2));
        assert_eq!(
            n.get_range_original(OffsetRange::Normalized(0..3)),
            Some("élé")
        );
    }

    #[test]
    fn lowercase_expansion_keeps_alignment() {
        let mut n = NormalizedString::from("İa");
        n.lowercase();
        assert_eq!(n.get(), "i\u{307}a");
        assert_eq!(
            n.get_range_original(OffsetRange::Normalized(0..3)),
            Some("İ")
        );
    }

    #[test]
    fn prepend_and_replace() {
        let mut n = NormalizedString::from("Hey friend");
        n.prepend("▁");
        n.replace(' ', "▁").unwrap();
        assert_eq!(n.get(), "▁Hey▁friend");
        let first = n.convert_offsets(OffsetRange::Normalized(0.."▁".len()));
        assert_eq!(first, Some(0..1));
    }

    #[test]
    fn strip_removes_whitespace_and_keeps_offsets() {
        let mut n = NormalizedString::from("  ab  ");
        n.strip();
        assert_eq!(n.get(), "ab");
        assert_eq!(n.convert_offsets(OffsetRange::Normalized(0..2)), Some(2..4));
        let mut all = NormalizedString::from("   ");
        all.strip();
        assert_eq!(all.get(), "");
    }

    #[test]
    fn split_behaviors() {
        let n = NormalizedString::from("the-final--countdown");
        let get = |b| {
            n.split('-', b)
                .unwrap()
                .iter()
                .map(|s| s.get().to_owned())
                .collect::<Vec<_>>()
        };
        use SplitDelimiterBehavior::*;
        assert_eq!(get(Removed), ["the", "final", "countdown"]);
        assert_eq!(get(Isolated), ["the", "-", "final", "-", "-", "countdown"]);
        assert_eq!(
            get(MergedWithPrevious),
            ["the-", "final-", "-", "countdown"]
        );
        assert_eq!(get(MergedWithNext), ["the", "-final", "-", "-countdown"]);
        assert_eq!(get(Contiguous), ["the", "-", "final", "--", "countdown"]);
    }

    #[test]
    fn slice_tracks_original_shift() {
        let n = NormalizedString::from("hello world");
        let s = n.slice(OffsetRange::Normalized(6..11)).unwrap();
        assert_eq!(s.get(), "world");
        assert_eq!(s.offsets_original(), (6, 11));
    }

    #[test]
    fn bytes_to_char_conversion() {
        assert_eq!(bytes_to_char("aéb", 1..3), Some(1..2));
        assert_eq!(bytes_to_char("aéb", 0..4), Some(0..3));
        assert_eq!(bytes_to_char("aéb", 2..3), None);
    }
}
