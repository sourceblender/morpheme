//! Incremental decoding for token-by-token generation.

use super::Tokenizer;
use crate::error::{Error, Result};

/// Decodes ids one at a time, yielding text as soon as it is complete.
///
/// Decoding each id on its own breaks text that spans several tokens: a
/// UTF-8 character split across byte-fallback tokens, or decoders that
/// strip a leading space only at the start of a sequence. `DecodeStream`
/// keeps just enough context to emit exactly the new text each step, so
/// concatenating every returned chunk equals [`Tokenizer::decode`] of all
/// ids. Semantics match Hugging Face `tokenizers`' `DecodeStream`.
///
/// ```no_run
/// # use splinter::Tokenizer;
/// let tokenizer = Tokenizer::from_file("tokenizer.json")?;
/// let mut stream = tokenizer.decode_stream(true);
/// for id in tokenizer.encode("Hello, streaming world! 😀", false)?.ids() {
///     if let Some(chunk) = stream.step(*id)? {
///         print!("{chunk}");
///     }
/// }
/// # Ok::<(), splinter::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct DecodeStream<'tok> {
    tokenizer: &'tok Tokenizer,
    skip_special_tokens: bool,
    /// Ids still needed to decode the next chunk: the ones behind
    /// `prefix` plus any not yet emitted.
    ids: Vec<u32>,
    /// Text of `ids[..prefix_index]`, already emitted; trimmed off the
    /// next decode.
    prefix: String,
    prefix_index: usize,
}

impl<'tok> DecodeStream<'tok> {
    pub(crate) fn new(tokenizer: &'tok Tokenizer, skip_special_tokens: bool) -> Self {
        Self {
            tokenizer,
            skip_special_tokens,
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
        }
    }

    /// Start from ids whose text was already shown (e.g. the prompt), so
    /// the next chunk is decoded in their context but not re-emitted.
    #[must_use]
    pub fn prefill(mut self, ids: &[u32]) -> Self {
        self.ids.extend_from_slice(ids);
        self
    }

    /// Add the next id. Returns the newly completed text, or `None` if the
    /// id does not complete any text yet (e.g. half of a UTF-8 character).
    pub fn step(&mut self, id: u32) -> Result<Option<String>> {
        self.step_many(&[id])
    }

    /// Add several ids at once; returns all text they complete.
    pub fn step_many(&mut self, ids: &[u32]) -> Result<Option<String>> {
        let (tokenizer, skip) = (self.tokenizer, self.skip_special_tokens);
        let decode = |ids: &[u32]| tokenizer.decode(ids, skip);

        if self.prefix.is_empty() && !self.ids.is_empty() {
            let prefix = decode(&self.ids)?;
            if !prefix.ends_with('\u{FFFD}') {
                self.prefix = prefix;
                self.prefix_index = self.ids.len();
            }
        }

        self.ids.extend_from_slice(ids);
        let text = decode(&self.ids)?;
        if text.len() <= self.prefix.len() || text.ends_with('\u{FFFD}') {
            return Ok(None);
        }
        let Some(new_text) = text.strip_prefix(self.prefix.as_str()) else {
            return Err(Error::Decoder(format!(
                "stream decode: text {text:?} for id {} does not start with the \
                 previously decoded {:?}",
                self.ids.last().copied().unwrap_or_default(),
                self.prefix
            )));
        };
        let new_text = new_text.to_owned();
        let new_prefix_index = self.ids.len() - self.prefix_index;
        self.ids.drain(..self.prefix_index);
        self.prefix = decode(&self.ids)?;
        self.prefix_index = new_prefix_index;
        Ok(Some(new_text))
    }
}
