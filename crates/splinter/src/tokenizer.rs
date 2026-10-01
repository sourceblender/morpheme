//! The top-level `Tokenizer` — wires the pipeline together.

pub mod hf;
pub mod json;

use std::borrow::Cow;
use std::path::Path;

use crate::decoder::{Decoder, WordPieceDecoder};
use crate::encoding::Encoding;
use crate::error::{Error, Result};
use crate::model::bpe::Bpe;
use crate::model::unigram::Unigram;
use crate::model::wordpiece::WordPiece;
use crate::model::Model;
use crate::normalizer::{IdentityNormalizer, Normalizer};
use crate::post_processor::PostProcessor;
use crate::pre_tokenizer::{PreTokenizer, Whitespace};
use crate::vocab::Vocab;

/// Concrete model variants the `Tokenizer` can wrap. Public so that
/// downstream callers can pattern-match, but the `TokenizerBuilder`
/// also accepts any variant directly via `Into`.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum ModelKind {
    /// Byte-Pair Encoding model.
    Bpe(Bpe),
    /// WordPiece greedy longest-match model.
    WordPiece(WordPiece),
    /// Unigram (SentencePiece) Viterbi-best-path model.
    Unigram(Unigram),
}

impl From<Bpe> for ModelKind {
    fn from(m: Bpe) -> Self {
        ModelKind::Bpe(m)
    }
}

impl From<WordPiece> for ModelKind {
    fn from(m: WordPiece) -> Self {
        ModelKind::WordPiece(m)
    }
}

impl From<Unigram> for ModelKind {
    fn from(m: Unigram) -> Self {
        ModelKind::Unigram(m)
    }
}

impl ModelKind {
    /// Borrow the model's vocabulary.
    pub fn vocab(&self) -> &Vocab {
        match self {
            ModelKind::Bpe(m) => &m.vocab,
            ModelKind::WordPiece(m) => &m.vocab,
            ModelKind::Unigram(m) => &m.vocab,
        }
    }
}

/// Glues a normalizer, pre-tokenizer, model, decoder, and
/// post-processor into a runnable pipeline.
#[derive(Debug)]
pub struct Tokenizer {
    normalizer: Box<dyn Normalizer>,
    pre_tokenizer: Box<dyn PreTokenizer>,
    model: ModelKind,
    decoder: Box<dyn Decoder>,
    post_processor: Option<Box<dyn PostProcessor>>,
}

impl Tokenizer {
    /// Build a tokenizer around a [`Bpe`] model with sensible
    /// defaults: `IdentityNormalizer`, `Whitespace` pre-tokenizer,
    /// `WordPieceDecoder` (no-op on GPT-2 BPE output).
    pub fn new(model: Bpe) -> Self {
        Self::builder(ModelKind::Bpe(model))
            .normalizer(Box::new(IdentityNormalizer))
            .pre_tokenizer(Box::new(Whitespace))
            .decoder(Box::new(WordPieceDecoder::default()))
            .build()
    }

    /// Build a tokenizer around a [`WordPiece`] model with sensible
    /// defaults: `IdentityNormalizer`, `Whitespace` pre-tokenizer,
    /// `WordPieceDecoder` (strips `##` continuation markers).
    pub fn wordpiece(model: WordPiece) -> Self {
        Self::builder(ModelKind::WordPiece(model))
            .normalizer(Box::new(IdentityNormalizer))
            .pre_tokenizer(Box::new(Whitespace))
            .decoder(Box::new(WordPieceDecoder::default()))
            .build()
    }

    /// Build a tokenizer around a [`Unigram`] model with sensible
    /// defaults: `IdentityNormalizer`, `Whitespace` pre-tokenizer,
    /// `WordPieceDecoder` (pass-through for Unigram tokens).
    pub fn unigram(model: Unigram) -> Self {
        Self::builder(ModelKind::Unigram(model))
            .normalizer(Box::new(IdentityNormalizer))
            .pre_tokenizer(Box::new(Whitespace))
            .decoder(Box::new(WordPieceDecoder::default()))
            .build()
    }

    /// Start building a tokenizer with explicit components. Accepts any
    /// model variant.
    pub fn builder(model: impl Into<ModelKind>) -> TokenizerBuilder {
        TokenizerBuilder {
            normalizer: Box::new(IdentityNormalizer),
            pre_tokenizer: Box::new(Whitespace),
            model: model.into(),
            decoder: Box::new(WordPieceDecoder::default()),
            post_processor: None,
        }
    }

    /// Replace the normalizer.
    pub fn with_normalizer(mut self, n: Box<dyn Normalizer>) -> Self {
        self.normalizer = n;
        self
    }

    /// Replace the pre-tokenizer.
    pub fn with_pre_tokenizer(mut self, p: Box<dyn PreTokenizer>) -> Self {
        self.pre_tokenizer = p;
        self
    }

    /// Replace the decoder.
    pub fn with_decoder(mut self, d: Box<dyn Decoder>) -> Self {
        self.decoder = d;
        self
    }

    /// Replace the post-processor.
    pub fn with_post_processor(mut self, p: Box<dyn PostProcessor>) -> Self {
        self.post_processor = Some(p);
        self
    }

    /// Borrow the underlying BPE model if the tokenizer uses one.
    pub fn bpe_model(&self) -> Option<&Bpe> {
        match &self.model {
            ModelKind::Bpe(m) => Some(m),
            _ => None,
        }
    }

    /// Borrow the underlying WordPiece model if the tokenizer uses one.
    pub fn wordpiece_model(&self) -> Option<&WordPiece> {
        match &self.model {
            ModelKind::WordPiece(m) => Some(m),
            _ => None,
        }
    }

    /// Borrow the underlying Unigram model if the tokenizer uses one.
    pub fn unigram_model(&self) -> Option<&Unigram> {
        match &self.model {
            ModelKind::Unigram(m) => Some(m),
            _ => None,
        }
    }

    /// Whether a post-processor is currently configured. Useful for
    /// tooling that rebuilds the `Tokenizer` and needs to preserve
    /// components that the public builder doesn't expose.
    pub fn has_post_processor(&self) -> bool {
        self.post_processor.is_some()
    }

    /// Borrow the vocabulary regardless of model type.
    pub fn vocab(&self) -> &Vocab {
        self.model.vocab()
    }

    /// Load a tokenizer from a JSON string in splinter's own format.
    pub fn from_json(s: &str) -> Result<Self> {
        json::from_str(s)
    }

    /// Serialize this tokenizer to a JSON string in splinter's own
    /// format.
    pub fn to_json(&self) -> Result<String> {
        json::to_string(self)
    }

    /// Load a tokenizer from a JSON file on disk.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let s = std::fs::read_to_string(path)?;
        json::from_str(&s)
    }

    /// Save this tokenizer to a JSON file on disk.
    pub fn to_file(&self, path: impl AsRef<Path>, pretty: bool) -> Result<()> {
        let s = if pretty {
            json::to_string_pretty(self)?
        } else {
            json::to_string(self)?
        };
        std::fs::write(path, s)?;
        Ok(())
    }

    /// Encode `text` into an `Encoding`.
    pub fn encode(&self, text: &str) -> Result<Encoding> {
        let encoding = self.encode_raw(text)?;

        if let Some(pp) = &self.post_processor {
            pp.apply(encoding)
        } else {
            Ok(encoding)
        }
    }

    /// Run normalizer → pre-tokenizer → model → vocab lookup, but
    /// skip the post-processor. Used internally by `encode_pair` so
    /// the pair-level post-processor doesn't double-apply.
    fn encode_raw(&self, text: &str) -> Result<Encoding> {
        let normalized: Cow<'_, str> = self.normalizer.normalize(text)?;
        let pre_tokens = self.pre_tokenizer.pre_tokenize(normalized.as_ref())?;

        let mut ids: Vec<u32> = Vec::new();
        let mut tokens: Vec<String> = Vec::new();
        let mut offsets: Vec<(usize, usize)> = Vec::new();

        let model = &self.model;
        for pt in pre_tokens {
            let ts: Vec<String> = match model {
                ModelKind::Bpe(m) => m.tokenize(pt.clone())?,
                ModelKind::WordPiece(m) => m.tokenize(pt.clone())?,
                ModelKind::Unigram(m) => m.tokenize(pt.clone())?,
            };
            let (s, e) = pt.span;
            for t in ts {
                let id = self
                    .vocab()
                    .token_to_id(&t)
                    .ok_or_else(|| Error::UnknownToken(t.clone()))?;
                ids.push(id);
                offsets.push((s, e));
                tokens.push(t);
            }
        }

        Ok(Encoding {
            type_ids: vec![0; ids.len()],
            ids,
            tokens,
            offsets,
        })
    }

    /// Encode a pair `(text_a, text_b)` through a sentence-pair post-processor.
    pub fn encode_pair(&self, text_a: &str, text_b: &str) -> Result<Encoding> {
        let enc_a = self.encode_raw(text_a)?;
        let enc_b = self.encode_raw(text_b)?;
        let Some(pp) = &self.post_processor else {
            return Err(Error::Model(
                "encode_pair called but no post-processor is configured".into(),
            ));
        };
        pp.apply_pair(enc_a, enc_b)
    }

    /// Decode a list of ids back into a string.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        let mut tokens = Vec::with_capacity(ids.len());
        for &id in ids {
            let tok = self.vocab().id_to_token(id)?;
            if skip_special && (tok.starts_with('<') && tok.ends_with('>')) {
                continue;
            }
            tokens.push(tok.to_owned());
        }
        self.decoder.decode(&tokens)
    }
}

/// Builder for [`Tokenizer`].
pub struct TokenizerBuilder {
    normalizer: Box<dyn Normalizer>,
    pre_tokenizer: Box<dyn PreTokenizer>,
    model: ModelKind,
    decoder: Box<dyn Decoder>,
    post_processor: Option<Box<dyn PostProcessor>>,
}

impl TokenizerBuilder {
    /// Set the normalizer.
    pub fn normalizer(mut self, n: Box<dyn Normalizer>) -> Self {
        self.normalizer = n;
        self
    }

    /// Set the pre-tokenizer.
    pub fn pre_tokenizer(mut self, p: Box<dyn PreTokenizer>) -> Self {
        self.pre_tokenizer = p;
        self
    }

    /// Set the decoder.
    pub fn decoder(mut self, d: Box<dyn Decoder>) -> Self {
        self.decoder = d;
        self
    }

    /// Set the post-processor.
    pub fn post_processor(mut self, p: Box<dyn PostProcessor>) -> Self {
        self.post_processor = Some(p);
        self
    }

    /// Build the [`Tokenizer`].
    pub fn build(self) -> Tokenizer {
        Tokenizer {
            normalizer: self.normalizer,
            pre_tokenizer: self.pre_tokenizer,
            model: self.model,
            decoder: self.decoder,
            post_processor: self.post_processor,
        }
    }
}
