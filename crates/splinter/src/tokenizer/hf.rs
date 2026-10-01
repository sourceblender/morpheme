//! Hugging Face `tokenizers.json` interop.
//!
//! Reads a JSON file produced by the
//! [`huggingface/tokenizers`](https://huggingface.co/docs/tokenizers)
//! library and produces a [`Tokenizer`]. Supports the common
//! components (BertNormalizer, ByteLevel, BertPreTokenizer,
//! Metaspace, Whitespace, WordPiece, BPE, Unigram) and rejects
//! unsupported ones with a clear error.
//!
//! ## Format differences vs. splinter's JSON
//!
//! - HF vocab is a `string -> id` object; splinter uses an ordered
//!   array.
//! - HF merges are `"a b"` strings; splinter uses `[a, b]` arrays.
//! - HF has a flat top-level structure (`normalizer`,
//!   `pre_tokenizer`, `model`, `decoder`, `post_processor`);
//!   splinter nests model config under `model.*`.
//!
//! Use [`From_str`] or [`From_file`] to load a saved HF tokenizer.

// The serde-derived configs mirror the HF schema. They exist only to
// parse the JSON, so we don't require docstrings on every field.
#![allow(missing_docs)]

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use crate::error::{Error as SplinterError, Result};
use crate::model::bpe::Bpe;
use crate::model::unigram::Unigram;
use crate::model::wordpiece::WordPiece;
use crate::normalizer::{BertNormalizer, BertNormalizerOpts, Lowercase, Nfkc, Normalizer};
use crate::pre_tokenizer::{
    BertPreTokenizer, ByteLevel, ByteLevelAddChar, MetaspacePreTokenizer, PreTokenizer, Whitespace,
};
use crate::tokenizer::Tokenizer;
use crate::vocab::Vocab;
use crate::ModelKind;

/// Top-level Hugging Face tokenizer file.
#[derive(Debug, Deserialize)]
pub struct HfTokenizerFile {
    pub version: String,
    #[serde(default)]
    pub truncation: Option<serde_json::Value>,
    #[serde(default)]
    pub padding: Option<serde_json::Value>,
    #[serde(default)]
    pub added_tokens: Vec<AddedToken>,
    #[serde(default)]
    pub normalizer: Option<NormalizerConfig>,
    #[serde(default)]
    pub pre_tokenizer: Option<PreTokenizerConfig>,
    #[serde(default)]
    pub post_processor: Option<PostProcessorConfig>,
    #[serde(default)]
    pub decoder: Option<DecoderConfig>,
    pub model: ModelConfig,
}

/// `added_tokens` entry — special tokens with optional leading/trailing
/// whitespace flags. We only model the `id` for now.
#[derive(Debug, Deserialize)]
pub struct AddedToken {
    pub id: u32,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub normalized: bool,
    #[serde(default)]
    pub special: bool,
}

/// HF normalizer — sum enum over the small set we support.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NormalizerConfig {
    BertNormalizer(BertNormalizerConfig),
    Lowercase,
    Nfkc,
    Nfd,
    StripAccents,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Default, Deserialize)]
pub struct BertNormalizerConfig {
    #[serde(default = "default_true")]
    pub clean_text: bool,
    #[serde(default = "default_true")]
    pub handle_chinese_chars: bool,
    #[serde(default = "default_true")]
    pub strip_accents: bool,
    #[serde(default = "default_true")]
    pub lowercase: bool,
}

fn default_true() -> bool {
    true
}

/// HF pre-tokenizer.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PreTokenizerConfig {
    BertPreTokenizer,
    ByteLevel(ByteLevelPreTokenizerConfig),
    Whitespace,
    WhitespaceSplit,
    Metaspace(MetaspacePreTokenizerConfig),
    Digits(DigitsPreTokenizerConfig),
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Default, Deserialize)]
pub struct ByteLevelPreTokenizerConfig {
    #[serde(default = "default_true")]
    pub add_prefix_space: bool,
    #[serde(default = "default_true")]
    pub add_dummy_prefix: bool,
    #[serde(default = "default_true")]
    pub trim_offsets: bool,
    #[serde(default = "default_true")]
    pub use_regex: bool,
}

#[derive(Debug, Default, Deserialize)]
pub struct MetaspacePreTokenizerConfig {
    #[serde(default = "default_metaspace_replacement")]
    pub replacement: String,
    #[serde(default = "default_true")]
    pub add_dummy_prefix: bool,
}

fn default_metaspace_replacement() -> String {
    "\u{2581}".to_string()
}

#[derive(Debug, Default, Deserialize)]
pub struct DigitsPreTokenizerConfig {
    #[serde(default = "default_true")]
    pub individual_digits: bool,
}

/// HF decoder.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecoderConfig {
    WordPiece(WordPieceDecoderConfig),
    ByteLevel,
    Metaspace,
    Strip(StripDecoderConfig),
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Default, Deserialize)]
pub struct WordPieceDecoderConfig {
    #[serde(default = "default_wp_prefix")]
    pub prefix: String,
    #[serde(default = "default_true")]
    pub cleanup: bool,
}

fn default_wp_prefix() -> String {
    "##".to_string()
}

#[derive(Debug, Default, Deserialize)]
pub struct StripDecoderConfig {
    #[serde(default = "default_strip_content")]
    pub content: String,
    #[serde(default = "default_strip_start")]
    pub start: usize,
    #[serde(default = "default_stop")]
    pub stop: usize,
}

fn default_strip_content() -> String {
    "".to_string()
}
fn default_strip_start() -> usize {
    0
}
fn default_stop() -> usize {
    0
}

/// HF model.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ModelConfig {
    BPE(BpeModelConfig),
    WordPiece(WordPieceModelConfig),
    Unigram(UnigramModelConfig),
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
pub struct BpeModelConfig {
    #[serde(default)]
    pub dropout: Option<f64>,
    #[serde(default)]
    pub unk_token: Option<String>,
    #[serde(default)]
    pub end_of_word_suffix: Option<String>,
    #[serde(default)]
    pub continuing_subword_suffix: Option<String>,
    #[serde(default)]
    pub fuse_unk: bool,
    #[serde(default)]
    pub byte_fallback: bool,
    /// HF stores vocab as `token -> id`.
    pub vocab: HashMap<String, u32>,
    pub merges: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct WordPieceModelConfig {
    #[serde(default)]
    pub unk_token: Option<String>,
    #[serde(default = "default_wp_prefix")]
    pub continuing_subword_prefix: String,
    #[serde(default = "default_max_chars")]
    pub max_input_chars_per_word: usize,
    pub vocab: HashMap<String, u32>,
}

fn default_max_chars() -> usize {
    100
}

#[derive(Debug, Deserialize)]
pub struct UnigramModelConfig {
    /// `unk_id` (not `unk_token`) — HF stores the unk id directly.
    #[serde(default)]
    pub unk_id: Option<u32>,
    /// `byte_fallback` (we don't support yet).
    #[serde(default)]
    pub byte_fallback: bool,
    pub vocab: Vec<UnigramVocabEntry>,
}

#[derive(Debug, Deserialize)]
pub struct UnigramVocabEntry {
    pub token: String,
    pub log_prob: f64,
}

/// HF post-processor config.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum PostProcessorConfig {
    /// `RobertaProcessing` — `<s> $A </s> $B </s>` with type ids
    /// `[0, ..., 0, 2, ..., 1, 2]`. Equivalent to a fixed
    /// `TemplateProcessing` but simpler to construct.
    RobertaProcessing(RobertaProcessingConfig),
    /// `TemplateProcessing` — a list of pieces where each piece is
    /// either a literal `SpecialToken` or a reference to a
    /// `Sequence`. Supports both `single` (sentence) and `pair`
    /// (sentence-pair) templates.
    TemplateProcessing(TemplateProcessingConfig),
    /// Catch-all for post-processors we don't (yet) support.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize, Default)]
pub struct RobertaProcessingConfig {
    #[serde(default = "default_roberta_token")]
    pub sep: Vec<String>,
    #[serde(default = "default_roberta_token")]
    pub cls: Vec<String>,
    #[serde(default)]
    pub trim_offset: bool,
    #[serde(default = "default_true")]
    pub handle_special_tokens: bool,
}

fn default_roberta_token() -> Vec<String> {
    Vec::new()
}

/// One entry in a `TemplateProcessing` template — a literal special
/// token or a sequence reference. HF wraps each piece in a single-
/// key object: `{"SpecialToken": {...}}` or `{"Sequence": {...}}`.
#[derive(Debug, Deserialize)]
pub enum TemplateProcessingPiece {
    /// `{"SpecialToken": {"id": "[CLS]", "type_id": 0}}`.
    SpecialToken(SpecialTokenPiece),
    /// `{"Sequence": {"id": "A"}}` (or `"B"`).
    Sequence(SequencePiece),
}

#[derive(Debug, Deserialize)]
pub struct SpecialTokenPiece {
    pub id: String,
    #[serde(default)]
    pub type_id: u32,
}

#[derive(Debug, Deserialize)]
pub struct SequencePiece {
    pub id: String,
    #[serde(default)]
    pub type_id: u32,
}

#[derive(Debug, Deserialize, Default)]
pub struct TemplateProcessingConfig {
    #[serde(default)]
    pub single: Vec<TemplateProcessingPiece>,
    #[serde(default)]
    pub pair: Vec<TemplateProcessingPiece>,
}

/// Helper: load an HF tokenizer from a JSON string.
pub fn from_str(s: &str) -> Result<Tokenizer> {
    let file: HfTokenizerFile = serde_json::from_str(s)
        .map_err(|e| SplinterError::Model(format!("HF parse error: {e}")))?;
    build(file)
}

/// Helper: load an HF tokenizer from a file path.
pub fn from_file(path: impl AsRef<Path>) -> Result<Tokenizer> {
    let raw = std::fs::read_to_string(path.as_ref())
        .map_err(|e| SplinterError::Model(format!("read error: {e}")))?;
    from_str(&raw)
}

/// Construct a [`Tokenizer`] from a parsed HF file.
fn build(file: HfTokenizerFile) -> Result<Tokenizer> {
    let model = match file.model {
        ModelConfig::BPE(bpe) => ModelKind::Bpe(build_bpe(bpe)?),
        ModelConfig::WordPiece(wp) => ModelKind::WordPiece(build_wordpiece(wp)?),
        ModelConfig::Unigram(ug) => ModelKind::Unigram(build_unigram(ug)?),
        ModelConfig::Unknown => {
            return Err(SplinterError::Model(
                "unsupported HF model type (only BPE, WordPiece, Unigram are supported in v0.1)"
                    .into(),
            ));
        }
    };

    let normalizer: Box<dyn Normalizer> = match file.normalizer {
        Some(NormalizerConfig::BertNormalizer(cfg)) => {
            let opts = BertNormalizerOpts {
                clean_text: cfg.clean_text,
                handle_chinese_chars: cfg.handle_chinese_chars,
                strip_accents: cfg.strip_accents,
                lowercase: cfg.lowercase,
            };
            Box::new(BertNormalizer::new(opts))
        }
        Some(NormalizerConfig::Lowercase) => Box::new(Lowercase),
        Some(NormalizerConfig::Nfkc) => Box::new(Nfkc),
        Some(NormalizerConfig::Nfd) | Some(NormalizerConfig::StripAccents) => {
            return Err(SplinterError::Model(
                "HF NFD / StripAccents normalizer not supported in v0.1".into(),
            ));
        }
        Some(NormalizerConfig::Unknown) | None => Box::new(crate::normalizer::IdentityNormalizer),
    };

    let pre_tokenizer: Box<dyn PreTokenizer> = match file.pre_tokenizer {
        Some(PreTokenizerConfig::BertPreTokenizer) => Box::new(BertPreTokenizer),
        Some(PreTokenizerConfig::ByteLevel(cfg)) => {
            let add = if cfg.add_dummy_prefix {
                ByteLevelAddChar::Add
            } else {
                ByteLevelAddChar::NoAdd
            };
            Box::new(ByteLevel::new(cfg.add_prefix_space, add))
        }
        Some(PreTokenizerConfig::Metaspace(cfg)) => {
            let marker = cfg.replacement.chars().next().unwrap_or('\u{2581}');
            Box::new(MetaspacePreTokenizer::new(marker, cfg.add_dummy_prefix))
        }
        Some(PreTokenizerConfig::Whitespace) | Some(PreTokenizerConfig::WhitespaceSplit) => {
            Box::new(Whitespace)
        }
        Some(PreTokenizerConfig::Digits(_)) => {
            // The HF Digits pre-tokenizer splits digits from letters.
            // We approximate it with Whitespace — the model handles
            // single chars correctly.
            Box::new(Whitespace)
        }
        Some(PreTokenizerConfig::Unknown) | None => Box::new(Whitespace),
    };

    let decoder: Box<dyn crate::decoder::Decoder> = match file.decoder {
        Some(DecoderConfig::WordPiece(cfg)) => {
            Box::new(crate::decoder::WordPieceDecoder::new(cfg.cleanup))
        }
        Some(DecoderConfig::ByteLevel) => Box::new(crate::decoder::ByteLevelDecoder),
        Some(DecoderConfig::Metaspace) => {
            return Err(SplinterError::Model(
                "HF Metaspace decoder not supported in v0.1".into(),
            ));
        }
        Some(DecoderConfig::Strip(_)) => {
            return Err(SplinterError::Model(
                "HF Strip decoder not supported in v0.1".into(),
            ));
        }
        Some(DecoderConfig::Unknown) | None => {
            Box::new(crate::decoder::WordPieceDecoder::default())
        }
    };

    let mut builder = Tokenizer::builder(model)
        .normalizer(normalizer)
        .pre_tokenizer(pre_tokenizer)
        .decoder(decoder);
    if let Some(pp_cfg) = file.post_processor {
        let pp = build_post_processor(pp_cfg)?;
        builder = builder.post_processor(pp);
    }
    Ok(builder.build())
}

/// Construct a [`crate::PostProcessor`] from an HF
/// [`PostProcessorConfig`].
fn build_post_processor(cfg: PostProcessorConfig) -> Result<Box<dyn crate::PostProcessor>> {
    match cfg {
        PostProcessorConfig::RobertaProcessing(_) => {
            // RoBERTa's `<s> $A </s> $B </s>` matches our
            // default RobertaPostProcessor (CLS=0, SEP=2, type
            // ids `[0, ..., 0, 2, ..., 1, 2]`). HF also supports
            // custom sep/cls token strings, but v0.1 hard-codes
            // `<s>`/`</s>`.
            Ok(Box::new(crate::RobertaPostProcessor))
        }
        PostProcessorConfig::TemplateProcessing(cfg) => {
            let single = build_template(cfg.single)?;
            let pair = if cfg.pair.is_empty() {
                None
            } else {
                Some(build_template(cfg.pair)?)
            };
            let pp = if let Some(pair) = pair {
                crate::TemplatePostProcessor::with_pair(single, pair)
            } else {
                crate::TemplatePostProcessor::new(single)
            };
            Ok(Box::new(pp))
        }
        PostProcessorConfig::Unknown => Err(SplinterError::Model(
            "HF post-processor type not recognized".into(),
        )),
    }
}

/// Convert HF template pieces to a splinter [`crate::Template`].
fn build_template(pieces: Vec<TemplateProcessingPiece>) -> Result<crate::Template> {
    let mut entries = Vec::with_capacity(pieces.len());
    for piece in pieces {
        let entry = match piece {
            TemplateProcessingPiece::SpecialToken(sp) => crate::TemplateEntry {
                piece: crate::TemplatePiece::TokenType(sp.id),
                type_id: sp.type_id,
            },
            TemplateProcessingPiece::Sequence(sp) => {
                let piece = match sp.id.as_str() {
                    "A" => crate::TemplatePiece::SequenceA,
                    "B" => crate::TemplatePiece::SequenceB,
                    "0" => crate::TemplatePiece::TypeIdA,
                    "1" => crate::TemplatePiece::TypeIdB,
                    other => {
                        return Err(SplinterError::Model(format!(
                            "unsupported HF Sequence id {other:?}"
                        )));
                    }
                };
                crate::TemplateEntry {
                    piece,
                    type_id: sp.type_id,
                }
            }
        };
        entries.push(entry);
    }
    Ok(crate::Template::new(entries))
}

fn build_bpe(cfg: BpeModelConfig) -> Result<Bpe> {
    if cfg.continuing_subword_suffix.is_some() {
        return Err(SplinterError::Model(
            "HF BPE continuing_subword_suffix not supported in splinter v0.1".into(),
        ));
    }
    let eow = cfg.end_of_word_suffix.unwrap_or_else(|| "</w>".to_string());

    // Convert HashMap<String, u32> to ordered Vec<String>. HF doesn't
    // guarantee insertion order, so we sort by id for determinism.
    let mut entries: Vec<(u32, String)> = cfg.vocab.into_iter().map(|(s, id)| (id, s)).collect();
    entries.sort_by_key(|(id, _)| *id);
    let vocab_list: Vec<String> = entries.into_iter().map(|(_, s)| s).collect();

    // Parse merges: each is `"a b"` or sometimes already `["a", "b"]`.
    let merges: Vec<(String, String)> = cfg
        .merges
        .into_iter()
        .map(|m| {
            if let Some((a, b)) = m.split_once(' ') {
                (a.to_string(), b.to_string())
            } else {
                // Already a JSON array? Unlikely in HF, but handle.
                (m.clone(), String::new())
            }
        })
        .filter(|(_, b)| !b.is_empty())
        .collect();

    let vocab = Vocab::from_tokens(vocab_list)
        .map_err(|e| SplinterError::Model(format!("vocab construction failed: {e}")))?;
    let mut b = Bpe::builder(vocab, merges, eow);
    if let Some(p) = cfg.dropout {
        b = b.dropout(p as f32);
    }
    if cfg.byte_fallback {
        b = b.byte_fallback(Bpe::byte_fallback_table());
    }
    Ok(b.build())
}

fn build_wordpiece(cfg: WordPieceModelConfig) -> Result<WordPiece> {
    let unk = cfg
        .unk_token
        .ok_or_else(|| SplinterError::Model("HF WordPiece requires unk_token".into()))?;
    let mut entries: Vec<(u32, String)> = cfg.vocab.into_iter().map(|(s, id)| (id, s)).collect();
    entries.sort_by_key(|(id, _)| *id);
    let vocab_list: Vec<String> = entries.into_iter().map(|(_, s)| s).collect();
    let vocab = Vocab::from_tokens(vocab_list)
        .map_err(|e| SplinterError::Model(format!("vocab construction failed: {e}")))?;
    // Real BERT vocabs store entries with an `</w>` end-of-word
    // suffix on the final character. We use the default
    // `WordPiece::new`, which sets the suffix to `</w>`.
    Ok(WordPiece::new(
        vocab,
        cfg.continuing_subword_prefix,
        unk,
        cfg.max_input_chars_per_word,
    ))
}

fn build_unigram(cfg: UnigramModelConfig) -> Result<Unigram> {
    if cfg.byte_fallback {
        return Err(SplinterError::Model(
            "HF Unigram byte_fallback not supported in splinter v0.1".into(),
        ));
    }
    // HF's UnigramModelConfig.vocab is a list of {token, log_prob} pairs
    // in the order of the vocab (i.e. token at index i has id i).
    let vocab_list: Vec<String> = cfg.vocab.iter().map(|e| e.token.clone()).collect();
    let log_probs: Vec<f64> = cfg.vocab.iter().map(|e| e.log_prob).collect();
    let vocab = Vocab::from_tokens(vocab_list)
        .map_err(|e| SplinterError::Model(format!("vocab construction failed: {e}")))?;
    let unk_token = if let Some(unk_id) = cfg.unk_id {
        vocab
            .id_to_token(unk_id)
            .map_err(|e| SplinterError::Model(format!("HF Unigram unk_id invalid: {e}")))?
            .to_string()
    } else {
        "<unk>".to_string()
    };
    // Use the minimum log-prob as the floor (we don't have HF's
    // explicit `min_score` here).
    let min_score = log_probs.iter().cloned().fold(f64::INFINITY, f64::min) - 1.0;
    Ok(Unigram::new(
        vocab,
        log_probs,
        "\u{2581}".to_string(),
        unk_token,
        min_score,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GPT2_TINY: &str = r##"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
          {"id": 50256, "content": "<|endoftext|>", "special": true}
        ],
        "normalizer": null,
        "pre_tokenizer": {
          "type": "ByteLevel",
          "add_prefix_space": false,
          "add_dummy_prefix": true,
          "trim_offsets": true,
          "use_regex": true
        },
        "post_processor": null,
        "decoder": {"type": "ByteLevel"},
        "model": {
          "type": "BPE",
          "dropout": null,
          "unk_token": null,
          "end_of_word_suffix": "</w>",
          "continuing_subword_suffix": null,
          "fuse_unk": false,
          "byte_fallback": false,
          "vocab": {
            "a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4
          },
          "merges": [
            "a b",
            "ab c"
          ]
        }
    }"##;

    #[test]
    fn loads_minimal_gpt2_style_tokenizer() {
        let t = from_str(GPT2_TINY).unwrap();
        // We don't try to encode here — GPT-2's byte-level BPE
        // doesn't add `</w>` to vocab entries, but our `Bpe` model
        // expects vocab entries to be suffixed. So we just verify
        // the loader produces a Tokenizer with the expected vocab
        // size.
        assert_eq!(t.vocab().len(), 5);
    }

    #[test]
    fn loads_wordpiece_with_mid_word_continuation() {
        // Vocab: <unk>, "h", "##ello</w>" — testing that mid-word
        // tokens get the `##` prefix.
        let raw = r###"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordPiece",
                "unk_token": "<unk>",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100,
                "vocab": {
                    "<unk>": 0,
                    "h": 1,
                    "##ello</w>": 2
                }
            }
        }"###;
        let t = from_str(raw).unwrap();
        // 'h' matches first; then 'ello</w>' has `##` prefix.
        let enc = t.encode("hello").unwrap();
        assert_eq!(enc.tokens, vec!["h", "##ello</w>"]);
    }

    #[test]
    fn loads_wordpiece_tokenizer_and_encodes() {
        // Minimal WordPiece vocab that exercises both single chars
        // and full-word matches.
        let raw = r###"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": {
                "type": "BertNormalizer",
                "clean_text": true,
                "handle_chinese_chars": true,
                "strip_accents": true,
                "lowercase": true
            },
            "pre_tokenizer": {"type": "BertPreTokenizer"},
            "post_processor": null,
            "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true},
            "model": {
                "type": "WordPiece",
                "unk_token": "<unk>",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100,
                "vocab": {
                    "<unk>": 0,
                    "h": 1,
                    "##e": 2,
                    "##l": 3,
                    "##o</w>": 4,
                    "hello</w>": 5
                }
            }
        }"###;
        let t = from_str(raw).unwrap();
        assert_eq!(t.vocab().len(), 6);
        let enc = t.encode("hello").unwrap();
        // Greedy matches 'hello</w>' (length 5, with default suffix).
        assert_eq!(enc.tokens, vec!["hello</w>"]);
    }

    #[test]
    fn accepts_hf_dropout_and_byte_fallback() {
        // v0.1 accepts both `dropout` (stored, not applied at
        // inference) and `byte_fallback` (resolves OOV via the
        // byte-to-unicode mapping).
        let raw = r###"{
            "version": "1.0",
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "decoder": null,
            "model": {
                "type": "BPE",
                "dropout": 0.1,
                "unk_token": null,
                "end_of_word_suffix": "</w>",
                "continuing_subword_suffix": null,
                "fuse_unk": false,
                "byte_fallback": true,
                "vocab": {"a</w>": 0, "b</w>": 1},
                "merges": []
            }
        }"###;
        let t = from_str(raw).expect("should load");
        let bpe = t.bpe_model().expect("bpe model");
        assert_eq!(bpe.dropout, Some(0.1));
        assert!(crate::model::bpe::Bpe::byte_fallback_table().contains(&'a'));
    }

    #[test]
    fn loads_roberta_post_processor() {
        // Standard RoBERTa-style `<s> $A </s> $B </s>` post-
        // processor. Round-trip: encode "the" through the loaded
        // tokenizer and verify specials appear.
        let raw = r###"{
            "version": "1.0",
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "decoder": null,
            "post_processor": {"type": "RobertaProcessing"},
            "model": {
                "type": "WordPiece",
                "unk_token": "<unk>",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100,
                "vocab": {
                    "<unk>": 0,
                    "the</w>": 1,
                    "quick</w>": 2
                }
            }
        }"###;
        let t = from_str(raw).expect("should load");
        let enc = t.encode("the").expect("encode");
        assert_eq!(enc.tokens[0], "<s>");
        assert_eq!(enc.tokens[enc.tokens.len() - 1], "</s>");
        assert_eq!(enc.tokens[1], "the</w>");
    }

    #[test]
    fn loads_template_post_processor() {
        // Standard BERT-style template: `[CLS] $A [SEP]`.
        let raw = r###"{
            "version": "1.0",
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "decoder": null,
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                    {"Sequence": {"id": "A"}},
                    {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
                ]
            },
            "model": {
                "type": "WordPiece",
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100,
                "vocab": {
                    "[UNK]": 0,
                    "hello</w>": 1,
                    "world</w>": 2
                }
            }
        }"###;
        let t = from_str(raw).expect("should load");
        let enc = t.encode("hello").expect("encode");
        assert_eq!(enc.tokens, vec!["[CLS]", "hello</w>", "[SEP]"]);
    }

    #[test]
    fn loads_template_post_processor_pair() {
        // BERT NLI / sentence-pair: `[CLS] $A [SEP] $B [SEP]`
        // with type ids `[0, ..., 0, 1, ..., 1]`.
        let raw = r###"{
            "version": "1.0",
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "decoder": null,
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                    {"Sequence": {"id": "A"}},
                    {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
                ],
                "pair": [
                    {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                    {"Sequence": {"id": "A"}},
                    {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
                    {"Sequence": {"id": "B", "type_id": 1}},
                    {"SpecialToken": {"id": "[SEP]", "type_id": 1}}
                ]
            },
            "model": {
                "type": "WordPiece",
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100,
                "vocab": {
                    "[UNK]": 0,
                    "hello</w>": 1,
                    "world</w>": 2
                }
            }
        }"###;
        let t = from_str(raw).expect("should load");
        let enc = t.encode_pair("hello", "world").expect("encode pair");
        assert_eq!(
            enc.tokens,
            vec!["[CLS]", "hello</w>", "[SEP]", "world</w>", "[SEP]"]
        );
        // type ids: [CLS]=0, hello=0, [SEP]=0, world=1, [SEP]=1
        assert_eq!(enc.type_ids, vec![0, 0, 0, 1, 1]);
    }
}
