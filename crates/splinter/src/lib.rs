//! `splinter` — a Rust tokenizer library.
//!
//! See `docs/architecture.md` for the design. This file is a stub.

#![deny(missing_docs)]

/// Library version, mirrored from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Placeholder. Real tokenizers will live in submodules.
pub fn placeholder() -> &'static str {
    "splinter is still being carved out."
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_set() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn placeholder_returns_expected_text() {
        assert!(placeholder().contains("splinter"));
    }
}