//! Unigram language model (SentencePiece-style).

pub mod lattice;
mod model;
mod trie;

pub use lattice::Lattice;
pub use model::Unigram;
