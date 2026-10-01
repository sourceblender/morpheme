//! Template post-processor: describe where special tokens go with a
//! template such as `"[CLS] $A [SEP] $B:1 [SEP]:1"`.
//!
//! Template syntax (pieces separated by whitespace):
//!
//! - `$A` / `$a` / `$` — the first sequence; `$B` / `$b` — the second.
//! - `$0`, `$1`, … — the first sequence with that type id.
//! - anything else — a special token, which must be listed in the
//!   processor's special tokens.
//! - a `:N` suffix sets the piece's type id (`$B:1`, `[SEP]:1`).

use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::special_encoding;
use crate::encoding::Encoding;
use crate::error::{Error, Result};
use crate::traits::PostProcessor;

fn err(msg: impl Into<String>) -> Error {
    Error::PostProcessor(msg.into())
}

/// Which input sequence a template piece refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Sequence {
    /// The first sequence.
    A,
    /// The second sequence (pairs only).
    B,
}

/// One piece of a [`Template`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Piece {
    /// An input sequence, with the type id its tokens get.
    Sequence {
        /// Which sequence.
        id: Sequence,
        /// Type id for its tokens.
        type_id: u32,
    },
    /// A special token (looked up in [`Tokens`] by `id`).
    SpecialToken {
        /// Key into the processor's special tokens.
        id: String,
        /// Type id for the inserted tokens.
        type_id: u32,
    },
}

impl Piece {
    fn parse_id(s: &str) -> Option<Self> {
        let Some(rest) = s.strip_prefix('$') else {
            return Some(Piece::SpecialToken {
                id: s.to_owned(),
                type_id: 0,
            });
        };
        let (id, type_id) = match rest {
            "" | "A" | "a" => (Sequence::A, 0),
            "B" | "b" => (Sequence::B, 0),
            n => (Sequence::A, n.parse::<u32>().ok()?),
        };
        Some(Piece::Sequence { id, type_id })
    }

    fn with_type_id(self, type_id: u32) -> Self {
        match self {
            Piece::Sequence { id, .. } => Piece::Sequence { id, type_id },
            Piece::SpecialToken { id, .. } => Piece::SpecialToken { id, type_id },
        }
    }
}

impl TryFrom<&str> for Piece {
    type Error = Error;

    fn try_from(s: &str) -> Result<Self> {
        let bad = || err(format!("cannot build a template piece from {s:?}"));
        let parts: Vec<&str> = s.split(':').collect();
        match parts.as_slice() {
            [id] => Piece::parse_id(id).ok_or_else(bad),
            [id, type_id] => {
                let type_id: u32 = type_id.parse().map_err(|_| bad())?;
                Ok(Piece::parse_id(id).ok_or_else(bad)?.with_type_id(type_id))
            }
            _ => Err(bad()),
        }
    }
}

impl TryFrom<String> for Piece {
    type Error = Error;

    fn try_from(s: String) -> Result<Self> {
        Piece::try_from(s.as_str())
    }
}

/// A special token: a key (`id`) that expands to one or more tokens.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "RawSpecialToken")]
pub struct SpecialToken {
    id: String,
    ids: Vec<u32>,
    tokens: Vec<String>,
}

#[derive(Deserialize)]
struct RawSpecialToken {
    id: String,
    ids: Vec<u32>,
    tokens: Vec<String>,
}

impl TryFrom<RawSpecialToken> for SpecialToken {
    type Error = Error;

    fn try_from(r: RawSpecialToken) -> Result<Self> {
        SpecialToken::new(r.id, r.ids, r.tokens)
    }
}

impl SpecialToken {
    /// Build a special token that expands to `tokens` with `ids`.
    pub fn new(id: String, ids: Vec<u32>, tokens: Vec<String>) -> Result<Self> {
        if ids.len() != tokens.len() {
            return Err(err(format!(
                "special token {id:?}: ids and tokens must have the same length"
            )));
        }
        Ok(Self { id, ids, tokens })
    }

    /// The key used in templates.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The token ids inserted.
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// The token strings inserted.
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }
}

impl From<(String, u32)> for SpecialToken {
    fn from((token, id): (String, u32)) -> Self {
        Self {
            id: token.clone(),
            ids: vec![id],
            tokens: vec![token],
        }
    }
}

impl From<(&str, u32)> for SpecialToken {
    fn from((token, id): (&str, u32)) -> Self {
        Self::from((token.to_owned(), id))
    }
}

impl From<(u32, String)> for SpecialToken {
    fn from((id, token): (u32, String)) -> Self {
        Self::from((token, id))
    }
}

impl From<(u32, &str)> for SpecialToken {
    fn from((id, token): (u32, &str)) -> Self {
        Self::from((token.to_owned(), id))
    }
}

/// A template: an ordered list of [`Piece`]s.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Template(Vec<Piece>);

impl Template {
    /// The pieces.
    pub fn pieces(&self) -> &[Piece] {
        &self.0
    }

    fn special_ids(&self) -> impl Iterator<Item = &str> {
        self.0.iter().filter_map(|p| match p {
            Piece::SpecialToken { id, .. } => Some(id.as_str()),
            Piece::Sequence { .. } => None,
        })
    }

    fn uses(&self, seq: Sequence) -> bool {
        self.0
            .iter()
            .any(|p| matches!(p, Piece::Sequence { id, .. } if *id == seq))
    }
}

impl From<Vec<Piece>> for Template {
    fn from(pieces: Vec<Piece>) -> Self {
        Self(pieces)
    }
}

impl TryFrom<&str> for Template {
    type Error = Error;

    fn try_from(s: &str) -> Result<Self> {
        s.split_whitespace()
            .map(Piece::try_from)
            .collect::<Result<Vec<_>>>()
            .map(Template)
    }
}

impl TryFrom<String> for Template {
    type Error = Error;

    fn try_from(s: String) -> Result<Self> {
        Template::try_from(s.as_str())
    }
}

impl TryFrom<Vec<&str>> for Template {
    type Error = Error;

    fn try_from(v: Vec<&str>) -> Result<Self> {
        v.into_iter()
            .map(Piece::try_from)
            .collect::<Result<Vec<_>>>()
            .map(Template)
    }
}

impl TryFrom<Vec<String>> for Template {
    type Error = Error;

    fn try_from(v: Vec<String>) -> Result<Self> {
        v.into_iter()
            .map(Piece::try_from)
            .collect::<Result<Vec<_>>>()
            .map(Template)
    }
}

/// The special tokens available to a template, keyed by id. Serialized
/// sorted by key.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Tokens(pub BTreeMap<String, SpecialToken>);

impl<T: Into<SpecialToken>> From<Vec<T>> for Tokens {
    fn from(v: Vec<T>) -> Self {
        Self(
            v.into_iter()
                .map(|t| {
                    let t: SpecialToken = t.into();
                    (t.id.clone(), t)
                })
                .collect(),
        )
    }
}

impl From<HashMap<String, SpecialToken>> for Tokens {
    fn from(m: HashMap<String, SpecialToken>) -> Self {
        Self(m.into_iter().collect())
    }
}

impl From<BTreeMap<String, SpecialToken>> for Tokens {
    fn from(m: BTreeMap<String, SpecialToken>) -> Self {
        Self(m)
    }
}

/// Post-processor driven by a [`Template`] for single inputs and another
/// for pairs.
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
/// use morpheme::models::WordLevel;
/// use morpheme::pre_tokenizers::WhitespaceSplit;
/// use morpheme::processors::TemplateProcessing;
/// use morpheme::Tokenizer;
///
/// let template = TemplateProcessing::builder()
///     .try_single("[CLS] $A [SEP]")?
///     .try_pair("[CLS] $A [SEP] $B:1 [SEP]:1")?
///     .special_tokens(vec![("[CLS]", 101), ("[SEP]", 102)])
///     .build()?;
///
/// let vocab: HashMap<String, u32> =
///     [("[UNK]", 0), ("hi", 7), ("there", 8)].map(|(t, i)| (t.to_string(), i)).into();
/// let tokenizer = Tokenizer::new(WordLevel::builder().vocab(vocab).unk_token("[UNK]").build()?)
///     .with_pre_tokenizer(WhitespaceSplit)
///     .with_post_processor(template);
///
/// assert_eq!(tokenizer.encode("hi", true)?.ids(), [101, 7, 102]);
/// assert_eq!(tokenizer.encode(("hi", "there"), true)?.type_ids(), [0, 0, 0, 1, 1]);
/// # Ok::<(), morpheme::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemplateProcessing {
    single: Template,
    pair: Template,
    #[serde(default)]
    special_tokens: Tokens,
}

fn default_single() -> Template {
    Template(vec![Piece::Sequence {
        id: Sequence::A,
        type_id: 0,
    }])
}

fn default_pair() -> Template {
    Template(vec![
        Piece::Sequence {
            id: Sequence::A,
            type_id: 0,
        },
        Piece::Sequence {
            id: Sequence::B,
            type_id: 1,
        },
    ])
}

impl Default for TemplateProcessing {
    fn default() -> Self {
        Self {
            single: default_single(),
            pair: default_pair(),
            special_tokens: Tokens::default(),
        }
    }
}

impl TemplateProcessing {
    pub(crate) fn rebind_token_ids(&mut self, lookup: &impl Fn(&str) -> Result<u32>) -> Result<()> {
        for token in self.special_tokens.0.values_mut() {
            token.ids = token
                .tokens
                .iter()
                .map(|text| lookup(text))
                .collect::<Result<_>>()?;
        }
        Ok(())
    }

    /// Start building a template processor.
    pub fn builder() -> TemplateProcessingBuilder {
        TemplateProcessingBuilder::default()
    }

    /// Template for single inputs.
    pub fn single(&self) -> &Template {
        &self.single
    }

    /// Template for pairs.
    pub fn pair(&self) -> &Template {
        &self.pair
    }

    /// The special tokens.
    pub fn special_tokens(&self) -> &Tokens {
        &self.special_tokens
    }

    fn count_added(&self, template: &Template) -> usize {
        template
            .special_ids()
            .map(|id| self.special_tokens.0.get(id).map_or(0, |t| t.ids.len()))
            .sum()
    }

    fn apply(
        &self,
        template: &Template,
        mut encodings: Vec<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        let mut out = Vec::with_capacity(template.0.len());
        for piece in &template.0 {
            match piece {
                Piece::Sequence { id, type_id } => {
                    let i = usize::from(*id == Sequence::B);
                    let n = encodings.len();
                    let encoding = encodings.get_mut(i).ok_or_else(|| {
                        err(format!(
                            "template uses sequence {id:?} but only {n} sequence(s) were given"
                        ))
                    })?;
                    // Overflow parts take the template's type id too
                    // (Hugging Face keeps their original id).
                    encoding.set_uniform_type_id(*type_id);
                    encoding.set_sequence_id(i);
                    out.push(encoding.clone());
                }
                Piece::SpecialToken { id, type_id } => {
                    if add_special_tokens {
                        let tok = self.special_tokens.0.get(id).ok_or_else(|| {
                            err(format!("missing special token {id:?} used in template"))
                        })?;
                        out.push(special_encoding(
                            tok.ids.clone(),
                            tok.tokens.clone(),
                            *type_id,
                        ));
                    }
                }
            }
        }
        Ok(out)
    }
}

impl PostProcessor for TemplateProcessing {
    fn added_tokens(&self, is_pair: bool) -> usize {
        if is_pair {
            self.count_added(&self.pair)
        } else {
            self.count_added(&self.single)
        }
    }

    fn process_encodings(
        &self,
        encodings: Vec<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        let template = match encodings.len() {
            1 => &self.single,
            2 => &self.pair,
            n => {
                return Err(err(format!(
                    "TemplateProcessing expects 1 or 2 encodings, got {n}"
                )));
            }
        };
        self.apply(template, encodings, add_special_tokens)
    }
}

/// Builder for [`TemplateProcessing`]. Defaults: single `$0`, pair
/// `$A:0 $B:1`, no special tokens.
#[derive(Debug, Clone, Default)]
pub struct TemplateProcessingBuilder {
    single: Option<Template>,
    pair: Option<Template>,
    special_tokens: Option<Tokens>,
}

impl TemplateProcessingBuilder {
    /// Set the single-input template.
    pub fn single(mut self, template: Template) -> Self {
        self.single = Some(template);
        self
    }

    /// Set the pair template.
    pub fn pair(mut self, template: Template) -> Self {
        self.pair = Some(template);
        self
    }

    /// Parse and set the single-input template.
    pub fn try_single<T: TryInto<Template, Error = Error>>(mut self, template: T) -> Result<Self> {
        self.single = Some(template.try_into()?);
        Ok(self)
    }

    /// Parse and set the pair template.
    pub fn try_pair<T: TryInto<Template, Error = Error>>(mut self, template: T) -> Result<Self> {
        self.pair = Some(template.try_into()?);
        Ok(self)
    }

    /// Set the special tokens.
    pub fn special_tokens(mut self, tokens: impl Into<Tokens>) -> Self {
        self.special_tokens = Some(tokens.into());
        self
    }

    /// Validate and build. Fails if the pair template does not use both
    /// sequences, or a template uses a special token that is not
    /// provided.
    pub fn build(self) -> Result<TemplateProcessing> {
        let single = self.single.unwrap_or_else(default_single);
        let pair = self.pair.unwrap_or_else(default_pair);
        let special_tokens = self.special_tokens.unwrap_or_default();

        if !(pair.uses(Sequence::A) && pair.uses(Sequence::B)) {
            return Err(err("template for `pair` must use both sequences"));
        }
        if single.uses(Sequence::B) {
            return Err(err("template for `single` cannot use sequence B"));
        }
        let mut missing: Vec<&str> = single
            .special_ids()
            .chain(pair.special_ids())
            .filter(|id| !special_tokens.0.contains_key(*id))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        if !missing.is_empty() {
            missing.sort_unstable();
            return Err(err(format!(
                "missing special token(s) with id(s) `{}`",
                missing.join(", ")
            )));
        }
        Ok(TemplateProcessing {
            single,
            pair,
            special_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Token;

    fn enc(tokens: &[(u32, &str, (usize, usize))]) -> Encoding {
        Encoding::from_tokens(
            tokens
                .iter()
                .map(|(id, v, o)| Token::new(*id, (*v).into(), *o))
                .collect(),
            0,
        )
    }

    fn bert_template() -> TemplateProcessing {
        TemplateProcessing::builder()
            .try_single("[CLS] $A [SEP]")
            .unwrap()
            .try_pair("[CLS] $A [SEP] $B:1 [SEP]:1")
            .unwrap()
            .special_tokens(vec![("[CLS]", 1), ("[SEP]", 0)])
            .build()
            .unwrap()
    }

    #[test]
    fn piece_serde() {
        let cases = [
            (
                Piece::Sequence {
                    id: Sequence::A,
                    type_id: 0,
                },
                r#"{"Sequence":{"id":"A","type_id":0}}"#,
            ),
            (
                Piece::Sequence {
                    id: Sequence::B,
                    type_id: 1,
                },
                r#"{"Sequence":{"id":"B","type_id":1}}"#,
            ),
            (
                Piece::SpecialToken {
                    id: "[CLS]".into(),
                    type_id: 0,
                },
                r#"{"SpecialToken":{"id":"[CLS]","type_id":0}}"#,
            ),
        ];
        for (piece, json) in cases {
            assert_eq!(serde_json::to_string(&piece).unwrap(), json);
            assert_eq!(serde_json::from_str::<Piece>(json).unwrap(), piece);
        }
    }

    #[test]
    fn piece_parsing() {
        let seq = |id, type_id| Piece::Sequence { id, type_id };
        assert_eq!(Piece::try_from("$").unwrap(), seq(Sequence::A, 0));
        assert_eq!(Piece::try_from("$a").unwrap(), seq(Sequence::A, 0));
        assert_eq!(Piece::try_from("$B").unwrap(), seq(Sequence::B, 0));
        assert_eq!(Piece::try_from("$1").unwrap(), seq(Sequence::A, 1));
        assert_eq!(Piece::try_from("$B:2").unwrap(), seq(Sequence::B, 2));
        assert_eq!(Piece::try_from("$:1").unwrap(), seq(Sequence::A, 1));
        assert_eq!(
            Piece::try_from("[SEP]:1").unwrap(),
            Piece::SpecialToken {
                id: "[SEP]".into(),
                type_id: 1
            }
        );
        assert!(Piece::try_from("$C:1").is_err());
        assert!(Piece::try_from("$A:").is_err());
        assert!(Piece::try_from("a:b:c").is_err());
        assert!(Piece::try_from("[X]:-1").is_err());
    }

    #[test]
    fn template_parsing_tolerates_repeated_spaces() {
        let t = Template::try_from("[CLS]  $A   [SEP]").unwrap();
        assert_eq!(t.pieces().len(), 3);
        let t = Template::try_from(vec!["[CLS]", "$0"]).unwrap();
        assert_eq!(t.pieces().len(), 2);
    }

    #[test]
    fn special_token_serde_and_validation() {
        let simple = SpecialToken::from(("[CLS]", 0));
        let json = r#"{"id":"[CLS]","ids":[0],"tokens":["[CLS]"]}"#;
        assert_eq!(serde_json::to_string(&simple).unwrap(), json);
        assert_eq!(serde_json::from_str::<SpecialToken>(json).unwrap(), simple);

        assert!(SpecialToken::new("x".into(), vec![1, 2], vec!["a".into()]).is_err());
        assert!(
            serde_json::from_str::<SpecialToken>(r#"{"id":"x","ids":[1,2],"tokens":["a"]}"#)
                .is_err()
        );
    }

    #[test]
    fn serde_matches_hf_layout() {
        let t = bert_template();
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(
            json,
            r#"{"single":[{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"[SEP]","type_id":0}}],"pair":[{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"[SEP]","type_id":0}},{"Sequence":{"id":"B","type_id":1}},{"SpecialToken":{"id":"[SEP]","type_id":1}}],"special_tokens":{"[CLS]":{"id":"[CLS]","ids":[1],"tokens":["[CLS]"]},"[SEP]":{"id":"[SEP]","ids":[0],"tokens":["[SEP]"]}}}"#
        );
        assert_eq!(
            serde_json::from_str::<TemplateProcessing>(&json).unwrap(),
            t
        );
    }

    #[test]
    fn builder_validation() {
        let missing = TemplateProcessing::builder()
            .try_single("[CLS] $A [SEP]")
            .unwrap()
            .special_tokens(vec![("[CLS]", 0)])
            .build();
        match missing {
            Err(Error::PostProcessor(m)) => assert!(m.contains("[SEP]"), "{m}"),
            other => panic!("expected error, got {other:?}"),
        }
        assert!(
            TemplateProcessing::builder()
                .try_pair("$A $A")
                .unwrap()
                .build()
                .is_err()
        );
        assert!(
            TemplateProcessing::builder()
                .try_single("$A $B")
                .unwrap()
                .build()
                .is_err()
        );
        assert!(TemplateProcessing::builder().build().is_ok());
    }

    #[test]
    fn processes_single_and_pair() {
        // Same expectations as HF's `template_processing` unit test.
        let p = bert_template();
        assert_eq!(p.added_tokens(false), 2);
        assert_eq!(p.added_tokens(true), 3);

        let a = enc(&[(12, "Hello", (0, 5)), (14, "there", (6, 11))]);
        let b = enc(&[(15, "pair", (0, 4))]);

        let single = p.process(a.clone(), None, true).unwrap();
        assert_eq!(single.ids(), &[1, 12, 14, 0]);
        assert_eq!(single.type_ids(), &[0, 0, 0, 0]);
        assert_eq!(single.tokens(), &["[CLS]", "Hello", "there", "[SEP]"]);
        assert_eq!(single.offsets(), &[(0, 0), (0, 5), (6, 11), (0, 0)]);
        assert_eq!(single.special_tokens_mask(), &[1, 0, 0, 1]);
        assert_eq!(single.word_ids(), &[None, None, None, None]);
        assert_eq!(single.token_to_sequence(2), Some(0));
        assert_eq!(single.token_to_sequence(3), None);

        let pair = p.process(a.clone(), Some(b.clone()), true).unwrap();
        assert_eq!(pair.ids(), &[1, 12, 14, 0, 15, 0]);
        assert_eq!(pair.type_ids(), &[0, 0, 0, 0, 1, 1]);
        assert_eq!(pair.special_tokens_mask(), &[1, 0, 0, 1, 0, 1]);
        assert_eq!(
            pair.offsets(),
            &[(0, 0), (0, 5), (6, 11), (0, 0), (0, 4), (0, 0)]
        );
        assert_eq!(pair.token_to_sequence(4), Some(1));
        assert_eq!(pair.token_to_sequence(5), None);

        let raw = p.process(a, Some(b), false).unwrap();
        assert_eq!(raw.ids(), &[12, 14, 15]);
        assert_eq!(raw.type_ids(), &[0, 0, 1]);
        assert_eq!(raw.special_tokens_mask(), &[0, 0, 0]);
    }

    #[test]
    fn multi_token_specials_and_type_id_pieces() {
        // Python: TemplateProcessing(single="[2FR] $1 [END]:2", ...)
        let p = TemplateProcessing::builder()
            .try_single("[2FR] $1 [END]:2")
            .unwrap()
            .special_tokens(vec![
                SpecialToken::new("[2FR]".into(), vec![7, 8], vec!["to".into(), "FR".into()])
                    .unwrap(),
                SpecialToken::from(("[END]", 9)),
            ])
            .build()
            .unwrap();
        assert_eq!(p.added_tokens(false), 3);
        let out = p.process(enc(&[(5, "x", (0, 1))]), None, true).unwrap();
        assert_eq!(out.ids(), &[7, 8, 5, 9]);
        assert_eq!(out.tokens(), &["to", "FR", "x", "[END]"]);
        assert_eq!(out.type_ids(), &[0, 0, 1, 2]);
        assert_eq!(out.special_tokens_mask(), &[1, 1, 0, 1]);
    }

    #[test]
    fn overflowing_encodings_are_processed() {
        let p = bert_template();
        let mut a = enc(&[(1, "a", (0, 1)), (2, "b", (1, 2)), (3, "c", (2, 3))]);
        a.truncate(2, 0, crate::TruncationDirection::Right);
        let out = p.process(a, None, true).unwrap();
        assert_eq!(out.ids(), &[1, 1, 2, 0]);
        let o: Vec<_> = out.overflowing().iter().map(|o| o.ids().to_vec()).collect();
        assert_eq!(o, vec![vec![1, 3, 0]]);
        assert_eq!(out.overflowing()[0].special_tokens_mask(), &[1, 0, 1]);
    }

    #[test]
    fn missing_special_token_in_deserialized_config_errors() {
        let json = r#"{"single":[{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{}}"#;
        let p: TemplateProcessing = serde_json::from_str(json).unwrap();
        assert_eq!(p.added_tokens(false), 0);
        assert!(p.process(enc(&[(5, "x", (0, 1))]), None, true).is_err());
        // Without special tokens requested, the missing entry is irrelevant.
        assert!(p.process(enc(&[(5, "x", (0, 1))]), None, false).is_ok());
    }
}
