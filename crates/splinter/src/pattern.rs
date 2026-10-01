//! Patterns used to find matches inside a string (for splitting and
//! replacing), plus [`SysRegex`], the regex engine used throughout the
//! crate.

use crate::Offsets;
use crate::error::{Error, Result};

/// A compiled regular expression supporting the look-around syntax used
/// by GPT-2-style pre-tokenizer patterns (`\s+(?!\S)` and friends).
///
/// Backed by `fancy-regex`, which delegates to the `regex` crate when a
/// pattern needs no fancy features.
#[derive(Debug, Clone)]
pub struct SysRegex {
    pattern: String,
    regex: fancy_regex::Regex,
}

impl SysRegex {
    /// Compile `pattern`.
    pub fn new(pattern: &str) -> Result<Self> {
        Ok(Self {
            pattern: pattern.to_owned(),
            regex: fancy_regex::Regex::new(pattern)?,
        })
    }

    /// The source pattern.
    pub fn as_str(&self) -> &str {
        &self.pattern
    }

    /// Iterate over `(start, end)` byte offsets of non-overlapping
    /// matches. Errors from the backtracking engine (e.g. exceeding the
    /// backtrack limit) are surfaced instead of silently ending the
    /// iteration.
    pub fn find_iter(&self, inside: &str) -> Result<Vec<Offsets>> {
        self.regex
            .find_iter(inside)
            .map(|m| m.map(|m| (m.start(), m.end())).map_err(Error::from))
            .collect()
    }
}

/// Something that can find matches inside a string.
///
/// `find_matches` returns the whole input partitioned into consecutive
/// `(offsets, is_match)` spans, in order, covering every byte exactly
/// once.
pub trait Pattern {
    /// Partition `inside` into matching and non-matching spans.
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>>;
}

fn partition(inside: &str, matches: impl IntoIterator<Item = Offsets>) -> Vec<(Offsets, bool)> {
    if inside.is_empty() {
        return vec![((0, 0), false)];
    }
    let mut prev = 0;
    let mut out = Vec::new();
    for (start, end) in matches {
        if prev != start {
            out.push(((prev, start), false));
        }
        out.push(((start, end), true));
        prev = end;
    }
    if prev != inside.len() {
        out.push(((prev, inside.len()), false));
    }
    out
}

impl Pattern for char {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        let c = *self;
        (move |x: char| x == c).find_matches(inside)
    }
}

impl Pattern for &str {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        if self.is_empty() {
            return Ok(vec![((0, inside.len()), false)]);
        }
        let matches: Vec<Offsets> = inside
            .match_indices(*self)
            .map(|(i, m)| (i, i + m.len()))
            .collect();
        Ok(partition(inside, matches))
    }
}

impl Pattern for &String {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        self.as_str().find_matches(inside)
    }
}

impl Pattern for &SysRegex {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        if inside.is_empty() {
            return Ok(vec![((0, 0), false)]);
        }
        Ok(partition(inside, self.find_iter(inside)?))
    }
}

impl<F> Pattern for F
where
    F: Fn(char) -> bool,
{
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        let matches: Vec<Offsets> = inside
            .char_indices()
            .filter(|(_, c)| self(*c))
            .map(|(b, c)| (b, b + c.len_utf8()))
            .collect();
        Ok(partition(inside, matches))
    }
}

/// A pattern as stored in `tokenizer.json`: `{"String": "..."}` or
/// `{"Regex": "..."}`. Used by the `Split` pre-tokenizer and the
/// `Replace` normalizer / decoder.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SplitPattern {
    /// A literal string.
    String(String),
    /// A regular expression.
    Regex(String),
}

impl SplitPattern {
    /// Compile into a regex (literal strings are escaped).
    pub fn to_regex(&self) -> Result<SysRegex> {
        match self {
            SplitPattern::String(s) => SysRegex::new(&fancy_regex::escape(s)),
            SplitPattern::Regex(r) => SysRegex::new(r),
        }
    }
}

impl From<&str> for SplitPattern {
    fn from(s: &str) -> Self {
        SplitPattern::String(s.to_owned())
    }
}

impl From<String> for SplitPattern {
    fn from(s: String) -> Self {
        SplitPattern::String(s)
    }
}

/// Wrapper used to invert a pattern: matches become non-matches and
/// vice versa. Used by `Split` with `invert: true`.
pub struct Invert<P: Pattern>(pub P);

impl<P: Pattern> Pattern for Invert<P> {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        Ok(self
            .0
            .find_matches(inside)?
            .into_iter()
            .map(|(offsets, is_match)| (offsets, !is_match))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn str_pattern_partitions_whole_input() {
        let m = "-".find_matches("a-b--c").unwrap();
        assert_eq!(
            m,
            vec![
                ((0, 1), false),
                ((1, 2), true),
                ((2, 3), false),
                ((3, 4), true),
                ((4, 5), true),
                ((5, 6), false)
            ]
        );
    }

    #[test]
    fn regex_lookahead_supported() {
        let re = SysRegex::new(r"\s+(?!\S)|\s+").unwrap();
        assert_eq!(re.find_iter("a   b").unwrap(), vec![(1, 3), (3, 4)]);
    }

    #[test]
    fn fn_pattern_handles_multibyte() {
        let m = (|c: char| c == 'é').find_matches("aéb").unwrap();
        assert_eq!(m, vec![((0, 1), false), ((1, 3), true), ((3, 4), false)]);
    }
}
