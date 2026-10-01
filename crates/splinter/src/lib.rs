//! `splinter` — a Rust tokenizer library.
//!
//! See `docs/architecture.md` for the design. This file is the entry
//! point of the library and re-exports the public API.

#![deny(missing_docs)]

/// Library version, mirrored from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod decoder;
pub mod encoding;
pub mod error;
pub mod model;
pub mod normalizer;
pub mod post_processor;
pub mod pre_tokenizer;
pub mod tokenizer;
pub mod trainer;
pub mod vocab;

pub use decoder::{ByteLevelDecoder, Decoder, WordPieceDecoder};
pub use encoding::Encoding;
pub use error::{Error, Result};
pub use model::{bpe::Bpe, unigram::Unigram, wordpiece::WordPiece, Model};
pub use normalizer::{
    BertNormalizer, BertNormalizerOpts, IdentityNormalizer, Lowercase, Nfd, Nfkc, Normalizer,
    Replace, StripAccents,
};
pub use post_processor::{
    PostProcessor, PostProcessorKind, RobertaPostProcessor, Template, TemplateEntry, TemplatePiece,
    TemplatePostProcessor,
};
pub use pre_tokenizer::{
    BertPreTokenizer, ByteLevel, ByteLevelAddChar, MetaspacePreTokenizer, PreToken, PreTokenText,
    PreTokenizer, Whitespace,
};
pub use tokenizer::{ModelKind, Tokenizer, TokenizerBuilder};
pub use trainer::{
    BpeTrainer, BpeTrainerBuilder, Corpus, TrainedModel, Trainer, UnigramTrainedModel,
    UnigramTrainer, UnigramTrainerBuilder, WordPieceTrainedModel, WordPieceTrainer,
    WordPieceTrainerBuilder,
};
pub use vocab::Vocab;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_set() {
        assert!(!VERSION.is_empty());
    }
}
