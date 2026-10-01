//! Unigram language model (SentencePiece-style).

pub(crate) mod lattice;
mod model;
mod trie;

pub(crate) use lattice::Lattice;
pub use model::Unigram;
