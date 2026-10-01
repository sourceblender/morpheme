//! The [`Tokenizer`]: the full pipeline, plus loading and saving in the
//! Hugging Face `tokenizer.json` format.

mod decode_stream;
#[cfg(feature = "hub")]
mod hub;
mod serialization;

pub use decode_stream::DecodeStream;
#[cfg(feature = "hub")]
#[cfg_attr(docsrs, doc(cfg(feature = "hub")))]
pub use hub::FromPretrainedParameters;

use std::borrow::Cow;
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::Path;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::added_vocabulary::{AddedToken, AddedVocabulary};
use crate::decoders::DecoderWrapper;
use crate::encoding::{Encoding, PaddingDirection, TruncationDirection};
use crate::error::{Error, Result};
use crate::models::ModelWrapper;
use crate::normalized_string::NormalizedString;
use crate::normalizers::NormalizerWrapper;
use crate::pre_tokenized_string::{OffsetType, PreTokenizedString};
use crate::pre_tokenizers::PreTokenizerWrapper;
use crate::processors::PostProcessorWrapper;
use crate::trainers::TrainerWrapper;
use crate::traits::{Decoder, Model, Normalizer, PostProcessor, PreTokenizer, Trainer};

/// One input sequence: raw text, or text already split into words.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum InputSequence<'s> {
    /// Raw text.
    Raw(Cow<'s, str>),
    /// Pre-split words. Each word is tokenized on its own and gets its
    /// index as word id; offsets are relative to each word.
    PreTokenized(Vec<Cow<'s, str>>),
}

impl<'s> From<&'s str> for InputSequence<'s> {
    fn from(s: &'s str) -> Self {
        InputSequence::Raw(Cow::Borrowed(s))
    }
}

impl<'s> From<&'s String> for InputSequence<'s> {
    fn from(s: &'s String) -> Self {
        InputSequence::Raw(Cow::Borrowed(s.as_str()))
    }
}

impl From<String> for InputSequence<'_> {
    fn from(s: String) -> Self {
        InputSequence::Raw(Cow::Owned(s))
    }
}

impl<'s> From<&'s [&'s str]> for InputSequence<'s> {
    fn from(words: &'s [&'s str]) -> Self {
        InputSequence::PreTokenized(words.iter().map(|w| Cow::Borrowed(*w)).collect())
    }
}

impl<'s> From<Vec<&'s str>> for InputSequence<'s> {
    fn from(words: Vec<&'s str>) -> Self {
        InputSequence::PreTokenized(words.into_iter().map(Cow::Borrowed).collect())
    }
}

impl From<Vec<String>> for InputSequence<'_> {
    fn from(words: Vec<String>) -> Self {
        InputSequence::PreTokenized(words.into_iter().map(Cow::Owned).collect())
    }
}

/// What to encode: a single sequence or a pair.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum EncodeInput<'s> {
    /// One sequence.
    Single(InputSequence<'s>),
    /// A sequence pair (e.g. question/context).
    Dual(InputSequence<'s>, InputSequence<'s>),
}

impl<'s, I: Into<InputSequence<'s>>> From<I> for EncodeInput<'s> {
    fn from(input: I) -> Self {
        EncodeInput::Single(input.into())
    }
}

impl<'s, I1, I2> From<(I1, I2)> for EncodeInput<'s>
where
    I1: Into<InputSequence<'s>>,
    I2: Into<InputSequence<'s>>,
{
    fn from((a, b): (I1, I2)) -> Self {
        EncodeInput::Dual(a.into(), b.into())
    }
}

/// How to truncate a pair of sequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TruncationStrategy {
    /// Remove tokens from the longest sequence first.
    #[default]
    LongestFirst,
    /// Only truncate the first sequence.
    OnlyFirst,
    /// Only truncate the second sequence.
    OnlySecond,
}

/// Truncation settings (`tokenizer.json` field `"truncation"`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TruncationParams {
    /// Side to truncate from.
    #[serde(default)]
    pub direction: TruncationDirection,
    /// Maximum length, special tokens included.
    pub max_length: usize,
    /// Pair strategy.
    pub strategy: TruncationStrategy,
    /// Overlap between overflowing parts.
    pub stride: usize,
}

impl Default for TruncationParams {
    fn default() -> Self {
        Self {
            direction: TruncationDirection::Right,
            max_length: 512,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
        }
    }
}

/// Target length for padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PaddingStrategy {
    /// Pad every encoding of a batch to the longest one.
    #[default]
    BatchLongest,
    /// Pad to a fixed length.
    Fixed(usize),
}

/// Padding settings (`tokenizer.json` field `"padding"`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PaddingParams {
    /// Target length.
    pub strategy: PaddingStrategy,
    /// Side to pad.
    pub direction: PaddingDirection,
    /// Round the target length up to a multiple of this.
    pub pad_to_multiple_of: Option<usize>,
    /// Id of the padding token.
    pub pad_id: u32,
    /// Type id of padding tokens.
    pub pad_type_id: u32,
    /// The padding token string.
    pub pad_token: String,
}

impl Default for PaddingParams {
    fn default() -> Self {
        Self {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".to_owned(),
        }
    }
}

/// Truncate a sequence (and optional pair) to `params.max_length`.
pub fn truncate_encodings(
    mut encoding: Encoding,
    mut pair: Option<Encoding>,
    params: &TruncationParams,
) -> Result<(Encoding, Option<Encoding>)> {
    if params.max_length == 0 {
        encoding.truncate(0, params.stride, params.direction);
        if let Some(p) = pair.as_mut() {
            p.truncate(0, params.stride, params.direction);
        }
        return Ok((encoding, pair));
    }
    let total = encoding.len() + pair.as_ref().map_or(0, Encoding::len);
    if total <= params.max_length {
        return Ok((encoding, pair));
    }
    let to_remove = total - params.max_length;
    let check_stride = |len: usize| {
        if len > 0 && params.stride >= len {
            Err(Error::Truncation(format!(
                "stride ({}) must be less than the truncated length ({len})",
                params.stride
            )))
        } else {
            Ok(())
        }
    };

    match params.strategy {
        TruncationStrategy::LongestFirst => match pair.as_mut() {
            Some(p) => {
                let (mut n1, mut n2) = (encoding.len(), p.len());
                let swap = n1 > n2;
                if swap {
                    std::mem::swap(&mut n1, &mut n2);
                }
                n2 = if n1 > params.max_length {
                    n1
                } else {
                    n1.max(params.max_length - n1)
                };
                if n1 + n2 > params.max_length {
                    n1 = params.max_length / 2;
                    n2 = n1 + params.max_length % 2;
                }
                if swap {
                    std::mem::swap(&mut n1, &mut n2);
                }
                if n1 < encoding.len() {
                    check_stride(n1)?;
                }
                if n2 < p.len() {
                    check_stride(n2)?;
                }
                encoding.truncate(n1, params.stride, params.direction);
                p.truncate(n2, params.stride, params.direction);
            }
            None => {
                check_stride(total - to_remove)?;
                encoding.truncate(total - to_remove, params.stride, params.direction);
            }
        },
        TruncationStrategy::OnlyFirst | TruncationStrategy::OnlySecond => {
            let target = if params.strategy == TruncationStrategy::OnlyFirst {
                &mut encoding
            } else {
                pair.as_mut().ok_or_else(|| {
                    Error::Truncation("OnlySecond strategy requires a pair".into())
                })?
            };
            let len = target.len();
            if len <= to_remove {
                return Err(Error::Truncation(
                    "sequence to truncate is too short to respect max_length".into(),
                ));
            }
            check_stride(len - to_remove)?;
            target.truncate(len - to_remove, params.stride, params.direction);
        }
    }
    Ok((encoding, pair))
}

/// Pad a batch of encodings according to `params`.
pub fn pad_encodings(encodings: &mut [Encoding], params: &PaddingParams) -> Result<()> {
    if encodings.is_empty() {
        return Ok(());
    }
    let mut target = match params.strategy {
        PaddingStrategy::Fixed(n) => n,
        PaddingStrategy::BatchLongest => encodings.iter().map(Encoding::len).max().unwrap_or(0),
    };
    if let Some(m) = params.pad_to_multiple_of {
        if m > 0 && target % m > 0 {
            target += m - target % m;
        }
    }
    encodings.par_iter_mut().for_each(|e| {
        e.pad(
            target,
            params.pad_id,
            params.pad_type_id,
            &params.pad_token,
            params.direction,
        )
    });
    Ok(())
}

/// A tokenizer: normalizer, pre-tokenizer, model, post-processor and
/// decoder, plus added tokens and truncation/padding settings.
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
/// use morpheme::models::WordLevel;
/// use morpheme::pre_tokenizers::Whitespace;
/// use morpheme::Tokenizer;
///
/// let vocab: HashMap<String, u32> = [("[UNK]", 0), ("[PAD]", 1), ("hello", 2), ("world", 3), ("!", 4)]
///     .map(|(t, i)| (t.to_string(), i))
///     .into();
/// let model = WordLevel::builder().vocab(vocab).unk_token("[UNK]").build()?;
/// let tokenizer = Tokenizer::new(model).with_pre_tokenizer(Whitespace);
///
/// let encoding = tokenizer.encode("hello world!", false)?;
/// assert_eq!(encoding.ids(), [2, 3, 4]);
/// assert_eq!(tokenizer.decode(encoding.ids(), false)?, "hello world !");
///
/// // Save and reload as a Hugging Face tokenizer.json.
/// let reloaded = Tokenizer::from_json(&tokenizer.to_json(true)?)?;
/// assert_eq!(reloaded.encode("hello world!", false)?, encoding);
/// # Ok::<(), morpheme::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct Tokenizer {
    normalizer: Option<NormalizerWrapper>,
    pre_tokenizer: Option<PreTokenizerWrapper>,
    model: ModelWrapper,
    post_processor: Option<PostProcessorWrapper>,
    decoder: Option<DecoderWrapper>,
    added_vocabulary: AddedVocabulary,
    truncation: Option<TruncationParams>,
    padding: Option<PaddingParams>,
}

impl Tokenizer {
    /// A tokenizer with just a model (no normalizer, pre-tokenizer,
    /// post-processor or decoder).
    pub fn new(model: impl Into<ModelWrapper>) -> Self {
        Self {
            normalizer: None,
            pre_tokenizer: None,
            model: model.into(),
            post_processor: None,
            decoder: None,
            added_vocabulary: AddedVocabulary::new(),
            truncation: None,
            padding: None,
        }
    }

    /// Load from a `tokenizer.json` string.
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    /// Load from `tokenizer.json` bytes.
    pub fn from_bytes(bytes: impl AsRef<[u8]>) -> Result<Self> {
        Ok(serde_json::from_slice(bytes.as_ref())?)
    }

    /// Load from a `tokenizer.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(bytes)
    }

    /// Download `tokenizer.json` for `identifier` (`name` or `org/name`)
    /// from the Hugging Face Hub and load it. Requires the `hub` feature.
    ///
    /// Files are cached in the standard Hugging Face cache (shared with
    /// Python), honoring `HF_HOME`, `HF_HUB_CACHE`, `HF_TOKEN`,
    /// `HF_ENDPOINT` and `HF_HUB_OFFLINE`. When the Hub is unreachable a
    /// cached copy is used if there is one. See [`FromPretrainedParameters`] for the options.
    ///
    /// ```no_run
    /// use morpheme::{FromPretrainedParameters, Tokenizer};
    ///
    /// let tokenizer = Tokenizer::from_pretrained("google-bert/bert-base-uncased", None)?;
    /// let pinned = Tokenizer::from_pretrained(
    ///     "openai-community/gpt2",
    ///     Some(FromPretrainedParameters::default().revision("607a30d783dfa663caf39e06633721c8d4cfcd7e")),
    /// )?;
    /// # Ok::<(), morpheme::Error>(())
    /// ```
    ///
    /// # Example
    ///
    /// ```no_run
    /// use morpheme::{FromPretrainedParameters, Tokenizer};
    ///
    /// // Downloads into (or reuses) the Hugging Face cache shared with Python.
    /// let bert = Tokenizer::from_pretrained("google-bert/bert-base-uncased", None)?;
    ///
    /// // Pin a revision; use a token for gated models (or set HF_TOKEN).
    /// let params = FromPretrainedParameters::default().revision("v1.0");
    /// let pinned = Tokenizer::from_pretrained("my-org/my-model", Some(params))?;
    /// # Ok::<(), morpheme::Error>(())
    /// ```
    #[cfg(feature = "hub")]
    #[cfg_attr(docsrs, doc(cfg(feature = "hub")))]
    pub fn from_pretrained(
        identifier: &str,
        params: Option<hub::FromPretrainedParameters>,
    ) -> Result<Self> {
        Self::from_file(hub::from_pretrained(identifier, params)?)
    }

    /// Serialize to `tokenizer.json`.
    pub fn to_json(&self, pretty: bool) -> Result<String> {
        Ok(if pretty {
            serde_json::to_string_pretty(self)?
        } else {
            serde_json::to_string(self)?
        })
    }

    /// Save as `tokenizer.json` using atomic replacement. Readers see the
    /// previous complete file or the new complete file. A destination
    /// symlink is replaced, rather than writing through it.
    pub fn save(&self, path: impl AsRef<Path>, pretty: bool) -> Result<()> {
        let contents = self.to_json(pretty)?;
        let path = path.as_ref();
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let mut file = tempfile::NamedTempFile::new_in(parent)?;
        file.write_all(contents.as_bytes())?;
        // Keep an existing file's permissions. A new file would otherwise
        // inherit tempfile's 0600, unreadable by other users of a shared
        // directory (or a later container uid), so give it the usual 0644.
        match std::fs::metadata(path) {
            Ok(metadata) => file.as_file().set_permissions(metadata.permissions())?,
            #[cfg(unix)]
            Err(_) => {
                use std::os::unix::fs::PermissionsExt;
                file.as_file()
                    .set_permissions(std::fs::Permissions::from_mode(0o644))?;
            }
            #[cfg(not(unix))]
            Err(_) => {}
        }
        file.as_file().sync_all()?;
        // tempfile's Windows implementation uses MoveFileExW with
        // MOVEFILE_REPLACE_EXISTING; persist replaces on both platforms.
        file.persist(path).map_err(|e| e.error)?;
        Ok(())
    }

    // ----- components -------------------------------------------------

    /// Set the normalizer (builder style).
    #[must_use]
    pub fn with_normalizer(mut self, normalizer: impl Into<NormalizerWrapper>) -> Self {
        self.set_normalizer(Some(normalizer.into()))
            .expect("refreshing added tokens cannot fail for a valid normalizer");
        self
    }

    /// Set the pre-tokenizer (builder style).
    #[must_use]
    pub fn with_pre_tokenizer(mut self, pre_tokenizer: impl Into<PreTokenizerWrapper>) -> Self {
        self.pre_tokenizer = Some(pre_tokenizer.into());
        self
    }

    /// Set the post-processor (builder style).
    #[must_use]
    pub fn with_post_processor(mut self, post_processor: impl Into<PostProcessorWrapper>) -> Self {
        self.post_processor = Some(post_processor.into());
        self
    }

    /// Set the decoder (builder style).
    #[must_use]
    pub fn with_decoder(mut self, decoder: impl Into<DecoderWrapper>) -> Self {
        self.decoder = Some(decoder.into());
        self
    }

    /// Replace the normalizer. Added tokens that match normalized text
    /// are re-normalized.
    pub fn set_normalizer(&mut self, normalizer: Option<NormalizerWrapper>) -> Result<()> {
        self.normalizer = normalizer;
        self.added_vocabulary
            .refresh(self.normalizer.as_ref().map(|n| n as &dyn Normalizer))
    }

    /// Replace the pre-tokenizer.
    pub fn set_pre_tokenizer(&mut self, pre_tokenizer: Option<PreTokenizerWrapper>) {
        self.pre_tokenizer = pre_tokenizer;
    }

    /// Replace the model.
    pub fn set_model(&mut self, model: impl Into<ModelWrapper>) {
        self.model = model.into();
    }

    /// Replace the post-processor.
    pub fn set_post_processor(&mut self, post_processor: Option<PostProcessorWrapper>) {
        self.post_processor = post_processor;
    }

    /// Replace the decoder.
    pub fn set_decoder(&mut self, decoder: Option<DecoderWrapper>) {
        self.decoder = decoder;
    }

    /// Set truncation. Fails if the stride is not smaller than the
    /// effective max length (max length minus added special tokens).
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use morpheme::models::WordLevel;
    /// use morpheme::pre_tokenizers::Whitespace;
    /// use morpheme::Tokenizer;
    ///
    /// let vocab: HashMap<String, u32> = [("[UNK]", 0), ("[PAD]", 1), ("hello", 2), ("world", 3), ("!", 4)]
    ///     .map(|(t, i)| (t.to_string(), i))
    ///     .into();
    /// let model = WordLevel::builder().vocab(vocab).unk_token("[UNK]").build()?;
    /// use morpheme::TruncationParams;
    ///
    /// let mut tokenizer = Tokenizer::new(model).with_pre_tokenizer(Whitespace);
    /// tokenizer.set_truncation(Some(TruncationParams { max_length: 2, ..Default::default() }))?;
    ///
    /// let encoding = tokenizer.encode("hello world !", false)?;
    /// assert_eq!(encoding.ids(), [2, 3]);
    /// assert_eq!(encoding.overflowing()[0].ids(), [4]); // the rest
    /// # Ok::<(), morpheme::Error>(())
    /// ```
    pub fn set_truncation(&mut self, truncation: Option<TruncationParams>) -> Result<()> {
        if let Some(t) = &truncation {
            let effective = t.max_length.saturating_sub(self.n_added_tokens(false));
            if t.stride >= effective && effective > 0 {
                return Err(Error::Truncation(format!(
                    "stride {} must be less than the effective max length {effective} \
                     ({} minus {} special tokens)",
                    t.stride,
                    t.max_length,
                    self.n_added_tokens(false)
                )));
            }
        }
        self.truncation = truncation;
        Ok(())
    }

    /// Set padding.
    pub fn set_padding(&mut self, padding: Option<PaddingParams>) {
        self.padding = padding;
    }

    /// The normalizer.
    pub fn normalizer(&self) -> Option<&NormalizerWrapper> {
        self.normalizer.as_ref()
    }

    /// The pre-tokenizer.
    pub fn pre_tokenizer(&self) -> Option<&PreTokenizerWrapper> {
        self.pre_tokenizer.as_ref()
    }

    /// The model.
    pub fn model(&self) -> &ModelWrapper {
        &self.model
    }

    /// The post-processor.
    pub fn post_processor(&self) -> Option<&PostProcessorWrapper> {
        self.post_processor.as_ref()
    }

    /// The decoder.
    pub fn decoder(&self) -> Option<&DecoderWrapper> {
        self.decoder.as_ref()
    }

    /// Truncation settings.
    pub fn truncation(&self) -> Option<&TruncationParams> {
        self.truncation.as_ref()
    }

    /// Padding settings.
    pub fn padding(&self) -> Option<&PaddingParams> {
        self.padding.as_ref()
    }

    /// The added tokens.
    pub fn added_vocabulary(&self) -> &AddedVocabulary {
        &self.added_vocabulary
    }

    /// When true, special tokens in the input are tokenized as plain
    /// text instead of being matched.
    pub fn set_encode_special_tokens(&mut self, value: bool) {
        self.added_vocabulary.set_encode_special_tokens(value);
    }

    // ----- vocabulary -------------------------------------------------

    /// The vocabulary, optionally including added tokens.
    pub fn vocab(&self, with_added_tokens: bool) -> HashMap<String, u32> {
        let mut vocab = self.model.vocab();
        if with_added_tokens {
            for (token, id) in self.added_vocabulary.vocab() {
                vocab.insert(token.clone(), *id);
            }
        }
        vocab
    }

    /// Vocabulary size, optionally counting added tokens that are not
    /// already in the model's vocabulary.
    pub fn vocab_size(&self, with_added_tokens: bool) -> usize {
        let base = self.model.vocab_size();
        if !with_added_tokens {
            return base;
        }
        let added = self.added_vocabulary.vocab();
        let overlapping = added
            .keys()
            .filter(|t| self.model.token_to_id(t).is_some())
            .count();
        base + added.len() - overlapping
    }

    /// Id of `token` (added tokens take precedence).
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.added_vocabulary.token_to_id(token, &self.model)
    }

    /// Token for `id` (added tokens take precedence).
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.added_vocabulary
            .simple_id_to_token(id)
            .or_else(|| self.model.id_to_token(id))
    }

    /// Add regular tokens. Returns how many were added.
    pub fn add_tokens(&mut self, tokens: &[AddedToken]) -> Result<usize> {
        self.added_vocabulary.add_tokens(
            tokens,
            &self.model,
            self.normalizer.as_ref().map(|n| n as &dyn Normalizer),
        )
    }

    /// Add special tokens (marked `special`). Returns how many were
    /// added.
    pub fn add_special_tokens(&mut self, tokens: &[AddedToken]) -> Result<usize> {
        let tokens: Vec<AddedToken> = tokens.iter().cloned().map(|t| t.special(true)).collect();
        self.add_tokens(&tokens)
    }

    // ----- encoding ---------------------------------------------------

    fn n_added_tokens(&self, is_pair: bool) -> usize {
        self.post_processor
            .as_ref()
            .map_or(0, |pp| pp.added_tokens(is_pair))
    }

    /// Run the normalizer (only) on `text`.
    pub fn normalize(&self, text: &str) -> Result<NormalizedString> {
        let mut n = NormalizedString::from(text);
        if let Some(normalizer) = &self.normalizer {
            normalizer.normalize(&mut n)?;
        }
        Ok(n)
    }

    fn pre_tokenize(&self, mut pretokenized: PreTokenizedString) -> Result<PreTokenizedString> {
        if let Some(pt) = &self.pre_tokenizer {
            pt.pre_tokenize(&mut pretokenized)?;
        }
        Ok(pretokenized)
    }

    fn encode_text(
        &self,
        text: &str,
        word_idx: Option<u32>,
        type_id: u32,
        offset_type: OffsetType,
    ) -> Result<Encoding> {
        let pretokenized = self
            .added_vocabulary
            .extract_and_normalize(self.normalizer.as_ref().map(|n| n as &dyn Normalizer), text)?;
        let mut pretokenized = self.pre_tokenize(pretokenized)?;
        pretokenized.tokenize(|n| self.model.tokenize(n.get()))?;
        pretokenized.into_encoding(word_idx, type_id, offset_type)
    }

    fn encode_sequence(
        &self,
        sequence: &InputSequence<'_>,
        type_id: u32,
        offset_type: OffsetType,
    ) -> Result<Encoding> {
        match sequence {
            InputSequence::Raw(text) => self.encode_text(text, None, type_id, offset_type),
            InputSequence::PreTokenized(words) => {
                let parts = words
                    .iter()
                    .enumerate()
                    .map(|(i, w)| self.encode_text(w, Some(i as u32), type_id, offset_type))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Encoding::merge(parts, false))
            }
        }
    }

    fn encode_with<'s>(
        &self,
        input: impl Into<EncodeInput<'s>>,
        add_special_tokens: bool,
        offset_type: OffsetType,
    ) -> Result<Encoding> {
        let (first, second) = match input.into() {
            EncodeInput::Single(s) => (s, None),
            EncodeInput::Dual(a, b) => (a, Some(b)),
        };
        let encoding = self.encode_sequence(&first, 0, offset_type)?;
        let pair = second
            .map(|s| self.encode_sequence(&s, 1, offset_type))
            .transpose()?;
        self.post_process(encoding, pair, add_special_tokens)
    }

    /// Encode a sequence or a pair. Offsets are **byte** offsets into
    /// the original input(s).
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use morpheme::models::WordLevel;
    /// use morpheme::pre_tokenizers::Whitespace;
    /// use morpheme::Tokenizer;
    ///
    /// let vocab: HashMap<String, u32> = [("[UNK]", 0), ("[PAD]", 1), ("hello", 2), ("world", 3), ("!", 4)]
    ///     .map(|(t, i)| (t.to_string(), i))
    ///     .into();
    /// let model = WordLevel::builder().vocab(vocab).unk_token("[UNK]").build()?;
    /// let tokenizer = Tokenizer::new(model).with_pre_tokenizer(Whitespace);
    ///
    /// // A single sequence, a pair, or pre-split words.
    /// assert_eq!(tokenizer.encode("hello world", true)?.ids(), [2, 3]);
    /// let pair = tokenizer.encode(("hello", "world"), true)?;
    /// assert_eq!(pair.type_ids(), [0, 1]);
    /// let words = tokenizer.encode(vec!["hello", "world"], true)?;
    /// assert_eq!(words.word_ids(), [Some(0), Some(1)]);
    /// # Ok::<(), morpheme::Error>(())
    /// ```
    pub fn encode<'s>(
        &self,
        input: impl Into<EncodeInput<'s>>,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        self.encode_with(input, add_special_tokens, OffsetType::Byte)
    }

    /// Like [`encode`](Self::encode), but offsets are **char** offsets
    /// (what the Python `tokenizers` API returns).
    pub fn encode_char_offsets<'s>(
        &self,
        input: impl Into<EncodeInput<'s>>,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        self.encode_with(input, add_special_tokens, OffsetType::Char)
    }

    fn encode_batch_with<'s, E>(
        &self,
        inputs: Vec<E>,
        add_special_tokens: bool,
        offset_type: OffsetType,
    ) -> Result<Vec<Encoding>>
    where
        E: Into<EncodeInput<'s>> + Send,
    {
        let mut encodings = inputs
            .into_par_iter()
            .map(|i| self.encode_with(i, add_special_tokens, offset_type))
            .collect::<Result<Vec<_>>>()?;
        if let Some(params) = &self.padding {
            pad_encodings(&mut encodings, params)?;
        }
        Ok(encodings)
    }

    /// Encode several inputs in parallel. With `BatchLongest` padding,
    /// all encodings are padded to the longest one. Byte offsets.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use morpheme::models::WordLevel;
    /// use morpheme::pre_tokenizers::Whitespace;
    /// use morpheme::Tokenizer;
    ///
    /// let vocab: HashMap<String, u32> = [("[UNK]", 0), ("[PAD]", 1), ("hello", 2), ("world", 3), ("!", 4)]
    ///     .map(|(t, i)| (t.to_string(), i))
    ///     .into();
    /// let model = WordLevel::builder().vocab(vocab).unk_token("[UNK]").build()?;
    /// use morpheme::PaddingParams;
    ///
    /// let mut tokenizer = Tokenizer::new(model).with_pre_tokenizer(Whitespace);
    /// tokenizer.set_padding(Some(PaddingParams { pad_id: 1, ..Default::default() }));
    ///
    /// let batch = tokenizer.encode_batch(vec!["hello", "hello world !"], false)?;
    /// assert_eq!(batch[0].ids(), [2, 1, 1]); // padded to the longest
    /// assert_eq!(batch[0].attention_mask(), [1, 0, 0]);
    /// # Ok::<(), morpheme::Error>(())
    /// ```
    pub fn encode_batch<'s, E>(
        &self,
        inputs: Vec<E>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>>
    where
        E: Into<EncodeInput<'s>> + Send,
    {
        self.encode_batch_with(inputs, add_special_tokens, OffsetType::Byte)
    }

    /// [`encode_batch`](Self::encode_batch) with char offsets.
    pub fn encode_batch_char_offsets<'s, E>(
        &self,
        inputs: Vec<E>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>>
    where
        E: Into<EncodeInput<'s>> + Send,
    {
        self.encode_batch_with(inputs, add_special_tokens, OffsetType::Char)
    }

    /// Truncate, post-process (add special tokens, merge pairs) and pad.
    pub fn post_process(
        &self,
        encoding: Encoding,
        pair: Option<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        let (encoding, pair) = match &self.truncation {
            Some(trunc) => {
                let n_added = self.n_added_tokens(pair.is_some());
                if add_special_tokens && n_added > 0 {
                    let max_length = trunc.max_length.checked_sub(n_added).ok_or_else(|| {
                        Error::Truncation(format!(
                            "max_length {} is smaller than the {n_added} required special tokens",
                            trunc.max_length
                        ))
                    })?;
                    if max_length == 0
                        && (!encoding.is_empty() || pair.as_ref().is_some_and(|p| !p.is_empty()))
                    {
                        return Err(Error::Truncation(
                            "no space for input tokens after reserving special tokens".into(),
                        ));
                    }
                    let params = TruncationParams {
                        max_length,
                        ..trunc.clone()
                    };
                    truncate_encodings(encoding, pair, &params)?
                } else {
                    truncate_encodings(encoding, pair, trunc)?
                }
            }
            None => (encoding, pair),
        };

        let encoding = match &self.post_processor {
            Some(pp) => pp.process(encoding, pair, add_special_tokens)?,
            None => match pair {
                None => encoding,
                Some(pair) => {
                    let mut first = encoding;
                    first.set_sequence_id(0);
                    let mut second = pair;
                    second.set_sequence_id(1);
                    Encoding::merge([first, second], false)
                }
            },
        };

        let mut out = [encoding];
        if let Some(params) = &self.padding {
            pad_encodings(&mut out, params)?;
        }
        let [encoding] = out;
        Ok(encoding)
    }

    // ----- decoding ---------------------------------------------------

    /// Decode ids back to text. Unknown ids are skipped.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use morpheme::models::WordLevel;
    /// use morpheme::pre_tokenizers::Whitespace;
    /// use morpheme::Tokenizer;
    ///
    /// let vocab: HashMap<String, u32> = [("[UNK]", 0), ("[PAD]", 1), ("hello", 2), ("world", 3), ("!", 4)]
    ///     .map(|(t, i)| (t.to_string(), i))
    ///     .into();
    /// let model = WordLevel::builder().vocab(vocab).unk_token("[UNK]").build()?;
    /// use morpheme::decoders::WordPiece;
    /// use morpheme::AddedToken;
    ///
    /// let mut tokenizer = Tokenizer::new(model)
    ///     .with_pre_tokenizer(Whitespace)
    ///     .with_decoder(WordPiece::new("##", true));
    /// tokenizer.add_special_tokens(&[AddedToken::new("[PAD]", true)])?;
    ///
    /// assert_eq!(tokenizer.decode(&[2, 3, 4, 1], false)?, "hello world! [PAD]");
    /// assert_eq!(tokenizer.decode(&[2, 3, 4, 1], true)?, "hello world!");
    /// # Ok::<(), morpheme::Error>(())
    /// ```
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let tokens: Vec<String> = ids
            .iter()
            .filter_map(|&id| self.id_to_token(id))
            .filter(|t| !skip_special_tokens || !self.added_vocabulary.is_special_token(t))
            .collect();
        match &self.decoder {
            Some(d) => d.decode(tokens),
            None => Ok(tokens.join(" ")),
        }
    }

    /// Decode several id sequences in parallel.
    pub fn decode_batch(
        &self,
        sequences: &[&[u32]],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        sequences
            .par_iter()
            .map(|ids| self.decode(ids, skip_special_tokens))
            .collect()
    }

    /// Start incremental decoding for ids that arrive one at a time
    /// (e.g. during generation). See [`DecodeStream`].
    pub fn decode_stream(&self, skip_special_tokens: bool) -> DecodeStream<'_> {
        DecodeStream::new(self, skip_special_tokens)
    }

    // ----- training ---------------------------------------------------

    /// Train the model on `sequences`. Each sequence is normalized and
    /// pre-tokenized with this tokenizer's components before counting.
    /// The trainer's special tokens are registered as added tokens. If
    /// the trainer is for a different kind of model, the model is
    /// replaced. Existing added tokens retain their options and are
    /// assigned ids against the new vocabulary. Post-processor and padding
    /// ids are rebound by token text. If a configured token is missing,
    /// training fails without changing this tokenizer.
    pub fn train<T, I, S>(&mut self, trainer: T, sequences: I) -> Result<&mut Self>
    where
        T: Into<TrainerWrapper>,
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
    {
        self.train_fallible(trainer, sequences.map(Ok))
    }

    fn train_fallible<T, I, S>(&mut self, trainer: T, sequences: I) -> Result<&mut Self>
    where
        T: Into<TrainerWrapper>,
        I: Iterator<Item = Result<S>> + Send,
        S: AsRef<str> + Send,
    {
        let mut trainer: TrainerWrapper = trainer.into();
        let mut read_error = None;
        let sequences = sequences
            .map_while(|sequence| match sequence {
                Ok(sequence) => Some(sequence),
                Err(error) => {
                    read_error = Some(error);
                    None
                }
            })
            .fuse();
        trainer.feed(sequences, |seq| {
            let normalized = self.normalize(seq)?;
            let pretokenized = self.pre_tokenize(PreTokenizedString::from(normalized))?;
            Ok(pretokenized
                .get_splits(OffsetType::Byte)
                .into_iter()
                .map(|(s, _, _)| s.to_owned())
                .collect())
        })?;
        if let Some(error) = read_error {
            return Err(error);
        }
        let mut model = self.model.clone();
        let special = trainer.train(&mut model)?;
        let mut added_vocabulary = AddedVocabulary::new();
        added_vocabulary.set_encode_special_tokens(self.added_vocabulary.encode_special_tokens());
        let mut tokens: Vec<AddedToken> = self
            .added_vocabulary
            .tokens_with_ids()
            .into_iter()
            .map(|t| t.token)
            .collect();
        for token in special {
            if let Some(existing) = tokens.iter_mut().find(|t| t.content == token.content) {
                existing.special = true;
            } else {
                tokens.push(token.special(true));
            }
        }
        added_vocabulary.add_tokens(
            &tokens,
            &model,
            self.normalizer.as_ref().map(|n| n as &dyn Normalizer),
        )?;
        let lookup = |token: &str| {
            added_vocabulary.token_to_id(token, &model).ok_or_else(|| {
                Error::Training(format!(
                    "configured token {token:?} is missing from the trained vocabulary; include it in the trainer's special tokens"
                ))
            })
        };
        let mut post_processor = self.post_processor.clone();
        if let Some(processor) = &mut post_processor {
            processor.rebind_token_ids(&lookup)?;
        }
        let mut padding = self.padding.clone();
        if let Some(padding) = &mut padding {
            padding.pad_id = lookup(&padding.pad_token)?;
        }
        self.model = model;
        self.added_vocabulary = added_vocabulary;
        self.post_processor = post_processor;
        self.padding = padding;
        Ok(self)
    }

    /// Train on the lines of text files (line endings are kept, like
    /// Hugging Face). Lines are streamed rather than retaining the full
    /// corpus. I/O and UTF-8 errors are reported, not skipped.
    pub fn train_from_files<T, P>(&mut self, trainer: T, files: &[P]) -> Result<&mut Self>
    where
        T: Into<TrainerWrapper>,
        P: AsRef<Path>,
    {
        let mut paths = files
            .iter()
            .map(|p| p.as_ref().to_owned())
            .collect::<Vec<_>>()
            .into_iter();
        let mut active: Option<(std::path::PathBuf, std::io::BufReader<std::fs::File>)> = None;
        let lines = std::iter::from_fn(move || {
            loop {
                if active.is_none() {
                    let path = paths.next()?;
                    match std::fs::File::open(&path) {
                        Ok(file) => active = Some((path, std::io::BufReader::new(file))),
                        Err(e) => {
                            return Some(Err(Error::Training(format!("{}: {e}", path.display()))));
                        }
                    }
                }
                let (path, reader) = active.as_mut().expect("file opened above");
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => active = None,
                    Ok(_) => return Some(Ok(line)),
                    Err(e) => {
                        return Some(Err(Error::Training(format!("{}: {e}", path.display()))));
                    }
                }
            }
        });
        self.train_fallible(trainer, lines)
    }
}

impl std::str::FromStr for Tokenizer {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::from_json(s)
    }
}
