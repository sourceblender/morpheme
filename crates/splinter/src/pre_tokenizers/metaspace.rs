//! SentencePiece-style metaspace pre-tokenizer and decoder: spaces
//! become `▁` and words keep their leading `▁`.

use serde::{Deserialize, Deserializer, Serialize, de};

use crate::error::Result;
use crate::normalized_string::SplitDelimiterBehavior;
use crate::pre_tokenized_string::PreTokenizedString;
use crate::traits::{Decoder, PreTokenizer};

/// When to prepend the replacement char to the input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrependScheme {
    /// Only on the first piece (the one starting at offset 0).
    First,
    /// Never.
    Never,
    /// On every piece that doesn't already start with it.
    Always,
}

/// Replaces spaces with `replacement` (default `▁`), optionally prepends
/// it, and optionally splits so each piece starts with it.
///
/// # Example
///
/// ```
/// use splinter::{OffsetType, PreTokenizedString, PreTokenizer};
///
/// fn split(pt: &impl PreTokenizer, text: &str) -> Vec<String> {
///     let mut s = PreTokenizedString::from(text);
///     pt.pre_tokenize(&mut s).unwrap();
///     s.get_splits(OffsetType::Byte).into_iter().map(|(w, _, _)| w.to_owned()).collect()
/// }
///
/// use splinter::pre_tokenizers::{Metaspace, PrependScheme};
///
/// // SentencePiece style: spaces become `▁` and start each word.
/// let m = Metaspace::new('▁', PrependScheme::Always, true);
/// assert_eq!(split(&m, "Hello world"), ["▁Hello", "▁world"]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub struct Metaspace {
    replacement: char,
    /// When to prepend `replacement`.
    pub prepend_scheme: PrependScheme,
    /// Split into pieces, each starting with `replacement`.
    pub split: bool,
    #[serde(skip)]
    str_rep: String,
}

impl<'de> Deserialize<'de> for Metaspace {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Helper {
            replacement: char,
            // Legacy (< 0.19) files use `add_prefix_space` instead of
            // `prepend_scheme`.
            add_prefix_space: Option<bool>,
            prepend_scheme: Option<PrependScheme>,
            split: Option<bool>,
        }
        let h = Helper::deserialize(deserializer)?;
        let mut scheme = h.prepend_scheme.unwrap_or(PrependScheme::Always);
        if h.add_prefix_space == Some(false) {
            if h.prepend_scheme.is_some_and(|s| s != PrependScheme::Never) {
                return Err(de::Error::custom(
                    "add_prefix_space does not match declared prepend_scheme",
                ));
            }
            scheme = PrependScheme::Never;
        }
        Ok(Self::new(h.replacement, scheme, h.split.unwrap_or(true)))
    }
}

impl Default for Metaspace {
    fn default() -> Self {
        Self::new('▁', PrependScheme::Always, true)
    }
}

impl Metaspace {
    /// Build a `Metaspace`.
    pub fn new(replacement: char, prepend_scheme: PrependScheme, split: bool) -> Self {
        Self {
            replacement,
            prepend_scheme,
            split,
            str_rep: replacement.to_string(),
        }
    }

    /// The replacement char.
    pub fn replacement(&self) -> char {
        self.replacement
    }

    /// Change the replacement char.
    pub fn set_replacement(&mut self, replacement: char) {
        self.replacement = replacement;
        self.str_rep = replacement.to_string();
    }
}

impl PreTokenizer for Metaspace {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, mut normalized| {
            normalized.replace(' ', &self.str_rep)?;
            let needs_prefix = !normalized.get().starts_with(self.replacement);
            match self.prepend_scheme {
                PrependScheme::Always if needs_prefix => {
                    normalized.prepend(&self.str_rep);
                }
                PrependScheme::First if needs_prefix && normalized.offsets_original().0 == 0 => {
                    normalized.prepend(&self.str_rep);
                }
                _ => {}
            }
            if self.split {
                normalized.split(self.replacement, SplitDelimiterBehavior::MergedWithNext)
            } else {
                Ok(vec![normalized])
            }
        })
    }
}

impl Decoder for Metaspace {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        Ok(tokens
            .iter()
            .enumerate()
            .map(|(i, token)| {
                token
                    .chars()
                    .filter_map(|c| {
                        if c == self.replacement {
                            if i == 0 && self.prepend_scheme != PrependScheme::Never {
                                None
                            } else {
                                Some(' ')
                            }
                        } else {
                            Some(c)
                        }
                    })
                    .collect()
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pre_tokenizers::tests::splits;

    #[test]
    fn basic() {
        let m = Metaspace::new('▁', PrependScheme::Always, true);
        assert_eq!(
            splits(&m, "Hey friend!"),
            vec![("▁Hey", (0, 3)), ("▁friend!", (3, 11))]
        );
    }

    #[test]
    fn multiple_spaces() {
        let m = Metaspace::new('▁', PrependScheme::Always, true);
        assert_eq!(
            splits(&m, "Hey   friend!"),
            vec![
                ("▁Hey", (0, 3)),
                ("▁", (3, 4)),
                ("▁", (4, 5)),
                ("▁friend!", (5, 13)),
            ]
        );
    }

    #[test]
    fn no_split() {
        let m = Metaspace::new('▁', PrependScheme::Always, false);
        assert_eq!(splits(&m, "Hey friend!"), vec![("▁Hey▁friend!", (0, 11))]);
    }

    #[test]
    fn serialization_and_legacy() {
        let m = Metaspace::new('_', PrependScheme::Always, true);
        let json = r#"{"replacement":"_","prepend_scheme":"always","split":true}"#;
        assert_eq!(serde_json::to_string(&m).unwrap(), json);
        assert_eq!(serde_json::from_str::<Metaspace>(json).unwrap(), m);

        let legacy = r#"{"replacement":"_","str_rep":"_","add_prefix_space":true}"#;
        assert_eq!(serde_json::from_str::<Metaspace>(legacy).unwrap(), m);

        let legacy_never = r#"{"replacement":"_","add_prefix_space":false}"#;
        assert_eq!(
            serde_json::from_str::<Metaspace>(legacy_never).unwrap(),
            Metaspace::new('_', PrependScheme::Never, true)
        );

        let mismatch = r#"{"replacement":"_","add_prefix_space":false,"prepend_scheme":"always"}"#;
        assert!(serde_json::from_str::<Metaspace>(mismatch).is_err());
    }

    #[test]
    fn decode() {
        let m = Metaspace::default();
        let out = m
            .decode_chain(vec!["▁Hey".into(), "▁friend!".into()])
            .unwrap();
        assert_eq!(out, vec!["Hey", " friend!"]);
        let never = Metaspace::new('▁', PrependScheme::Never, true);
        assert_eq!(
            never.decode_chain(vec!["▁Hey".into()]).unwrap(),
            vec![" Hey"]
        );
    }
}
