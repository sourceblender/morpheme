//! Decoders — token strings back into text.
//!
//! v0.1 ships `WordPieceDecoder` (strips `##` continuation markers)
//! and `ByteLevelDecoder` (inverse of
//! [`crate::pre_tokenizer::ByteLevel`]'s byte-to-unicode mapping).

use crate::error::Result;
use crate::pre_tokenizer::byte_level::ByteLevel as ByteLevelPre;

/// Anything that turns a list of token strings back into text.
pub trait Decoder: Send + Sync + std::fmt::Debug {
    /// Decode a sequence of token strings into a single string.
    fn decode(&self, tokens: &[String]) -> Result<String>;
}

/// BERT-style decoder: joins consecutive tokens and strips a leading
/// `##` from continuation tokens.
#[derive(Debug, Default, Clone, Copy)]
pub struct WordPieceDecoder {
    /// If true, emit a single space between tokens (true is what BERT
    /// needs since the pre-tokenizer already emits spaces as their
    /// own pre-tokens).
    pub cleanup: bool,
}

impl WordPieceDecoder {
    /// Construct a `WordPieceDecoder` with the given `cleanup` flag.
    pub fn new(cleanup: bool) -> Self {
        Self { cleanup }
    }
}

impl Decoder for WordPieceDecoder {
    fn decode(&self, tokens: &[String]) -> Result<String> {
        // Decode each token to its raw chars (strip `##` between
        // every char, strip trailing `</w>`), then optionally insert
        // spaces between word boundaries.
        let mut out = String::new();
        for (i, t) in tokens.iter().enumerate() {
            // Split on `##` to get the raw chars, then strip the
            // `</w>` end-of-word suffix from the final char.
            let mut pieces: Vec<&str> = t.split("##").collect();
            if let Some(last) = pieces.last_mut() {
                if let Some(stripped) = last.strip_suffix("</w>") {
                    *last = stripped;
                }
            }
            let raw = pieces.join("");

            if i > 0 && self.cleanup && !out.ends_with(' ') {
                let prev_ended_word = tokens[i - 1].ends_with("</w>");
                if prev_ended_word {
                    out.push(' ');
                }
            }
            out.push_str(&raw);
        }
        Ok(out)
    }
}

/// Inverse of [`ByteLevelPre`]. Joins tokens, then maps the joined
/// string back through the unicode→byte mapping to recover the
/// original bytes.
#[derive(Debug, Default, Clone, Copy)]
pub struct ByteLevelDecoder;

impl Decoder for ByteLevelDecoder {
    fn decode(&self, tokens: &[String]) -> Result<String> {
        let joined = tokens.join("");
        let mut out = String::with_capacity(joined.len());
        for c in joined.chars() {
            if let Some(b) = ByteLevelPre::unmap_char(c) {
                let buf = [b];
                let s = std::str::from_utf8(&buf).unwrap_or("?");
                out.push_str(s);
            } else {
                // Unmapped char — keep it.
                out.push(c);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wordpiece_strips_double_hash() {
        let d = WordPieceDecoder::default();
        let s = d
            .decode(&["hello".into(), "##world".into(), "!".into()])
            .unwrap();
        assert_eq!(s, "helloworld!");
    }

    #[test]
    fn wordpiece_strips_end_of_word_suffix() {
        let d = WordPieceDecoder::default();
        let s = d
            .decode(&[
                "h".into(),
                "##e##l##l##o</w>".into(),
                "w".into(),
                "##orld</w>".into(),
            ])
            .unwrap();
        assert_eq!(s, "helloworld");
    }

    #[test]
    fn byte_level_round_trip() {
        let d = ByteLevelDecoder;
        let input = "hello world";
        let table = crate::pre_tokenizer::byte_level::bytes_to_unicode();
        let mapped: String = input.bytes().map(|b| table[b as usize]).collect();
        // No ASCII space in `mapped` (the space byte was mapped to
        // U+0120 Ġ), so splitting on ' ' yields one token.
        let tokens: Vec<String> = mapped.split(' ').map(String::from).collect();
        assert_eq!(tokens.len(), 1);
        let decoded = d.decode(&tokens).unwrap();
        assert_eq!(decoded, "hello world");
    }
}
